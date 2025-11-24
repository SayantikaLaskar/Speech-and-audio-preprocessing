import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Josh Talks Hindi ASR data and build a Whisper-ready manifest."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the provided CSV (e.g. 'FT Data - data.csv').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/joshtalks_hi",
        help="Directory to store processed assets.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for audio files.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Parallel download workers.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=5,
        help="Number of retries for failed downloads.",
    )
    return parser.parse_args()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalise_text(text: str) -> str:
    """Lowercase, remove duplicate whitespace, and strip unsupported chars."""
    if not text:
        return ""
    text = text.replace("’", "'").replace("`", "'")
    text = text.lower()
    # Keep Devanagari, basic latin letters, digits, and spaces
    text = re.sub(r"[^0-9\u0900-\u097fa-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_json(url: str, retry: int) -> Any:
    for attempt in range(1, retry + 1):
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retry} attempts.")


def flatten_transcript(obj: Any) -> str:
    """
    The JSON sometimes appears either as:
    1. {"segments": [{"text": ...}, ...]}
    2. [{"text": ..., "start": ..., "end": ...}, ...]
    3. {"text": "..."}  (single blob)
    """
    if isinstance(obj, dict):
        if "segments" in obj and isinstance(obj["segments"], list):
            return " ".join(normalise_text(seg.get("text", "")) for seg in obj["segments"])
        if "text" in obj:
            return normalise_text(obj["text"])
    if isinstance(obj, list):
        return " ".join(normalise_text(seg.get("text", "")) for seg in obj if isinstance(seg, dict))
    raise ValueError("Unrecognised transcription schema.")


def download_file(url: str, dest: Path, retry: int) -> None:
    if dest.exists():
        return
    for attempt in range(1, retry + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except requests.RequestException as exc:  # pragma: no cover
            if attempt == retry:
                raise RuntimeError(f"Failed downloading {url}: {exc}") from exc
            time.sleep(2 * attempt)


def rewrite_bucket(url: str) -> str:
    """
    The brief notes that assets were migrated from
    `joshtalks-data-collection/hq_data/hi/...` to `upload_goai/...`.
    Convert legacy URLs on the fly so the provided CSV stays untouched.
    """
    old_prefix = "https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/"
    new_prefix = "https://storage.googleapis.com/upload_goai/"
    if url.startswith(old_prefix):
        return new_prefix + url[len(old_prefix) :]
    return url


def process_row(
    row: Dict[str, Any],
    audio_dir: Path,
    transcript_dir: Path,
    metadata_dir: Path,
    retry: int,
) -> Optional[Dict[str, Any]]:
    recording_id = row["recording_id"]

    audio_path = audio_dir / f"{recording_id}.wav"
    transcript_path = transcript_dir / f"{recording_id}.json"
    metadata_path = metadata_dir / f"{recording_id}.json"

    audio_url = rewrite_bucket(row["rec_url_gcp"])
    transcript_url = rewrite_bucket(row["transcription_url_gcp"])
    metadata_url = rewrite_bucket(row["metadata_url_gcp"])

    download_file(audio_url, audio_path, retry=retry)
    transcript_json = fetch_json(transcript_url, retry=retry)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_json, f, ensure_ascii=False, indent=2)

    try:
        metadata_json = fetch_json(metadata_url, retry=retry)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, ensure_ascii=False, indent=2)
    except Exception:
        metadata_json = None

    transcript_text = flatten_transcript(transcript_json)
    if not transcript_text:
        return None

    # verify audio file is readable + capture duration (retry once on corruption)
    duration = None
    for attempt in range(2):
        try:
            audio, sr = sf.read(str(audio_path))
            duration = len(audio) / sr
            break
        except Exception:
            if attempt == 0:
                if audio_path.exists():
                    audio_path.unlink()
                download_file(audio_url, audio_path, retry=retry)
            else:
                raise
    if duration is None:
        return None

    manifest_entry = {
        "audio_filepath": str(audio_path.resolve()),
        "duration": duration,
        "text": transcript_text,
        "language": row["language"],
        "user_id": row["user_id"],
        "recording_id": recording_id,
        "metadata_path": str(metadata_path.resolve()) if metadata_json else None,
    }
    return manifest_entry


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    transcript_dir = output_dir / "transcripts"
    metadata_dir = output_dir / "metadata"
    for path in (audio_dir, transcript_dir, metadata_dir):
        safe_mkdir(path)

    rows = df.to_dict(orient="records")
    manifest: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_row, row, audio_dir, transcript_dir, metadata_dir, args.retry)
            for row in rows
        ]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result:
                manifest.append(result)

    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats = {
        "num_records": len(manifest),
        "total_hours": round(sum(m["duration"] for m in manifest) / 3600, 2),
        "min_duration": round(min(m["duration"] for m in manifest), 2),
        "max_duration": round(max(m["duration"] for m in manifest), 2),
        "languages": sorted({m["language"] for m in manifest}),
    }
    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Wrote manifest to", manifest_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()


