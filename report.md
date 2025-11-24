# Josh Talks ASR Task – Worklog & Results Template

## 1. Data Understanding

- Source: `FT Data - data.csv` (~10 hours Hindi speech). Each row contains audio, transcription JSON, and metadata URLs.
- All samples are mono WAV at 16 kHz+ with segment-level transcripts (`start`, `end`, `text`).
- Speaker IDs are anonymised but stable; duration ranges 7–20 minutes per file.

## 2. Pre-processing Steps

| Step | Description |
| --- | --- |
| 1 | Parsed the CSV via `pandas`, validated URLs, and fanned rows to parallel workers. |
| 2 | Downloaded audio + transcripts (and metadata when available) into `data/joshtalks_hi/{audio,transcripts,metadata}`. Retries (x5) shield transient GCS throttling. |
| 3 | Flattened each transcription JSON: concatenated ordered segments, lowercased, removed unsupported punctuation, and collapsed whitespace. |
| 4 | Verified waveforms with `soundfile` (duration checkpoint) and wrote a HF-compatible `manifest.jsonl` containing `audio_filepath`, `duration`, `text`, `language`, `user_id`, `recording_id`. |
| 5 | Exported `dataset_summary.json` (hours, min/max duration, language distribution) for auditability. |

> **Note:** If the data owner rotates URLs, update the CSV or follow the instructions from the brief (replace the path suffix with the new bucket prefix) before rerunning the script.

## 3. Fine-tuning Details

- Base model: `openai/whisper-small` (244M params).
- Pre-processing: `WhisperProcessor` (language `hi`, task `transcribe`), forced decoder IDs disabled to allow open-ended decoding.
- Splits: 90% train / 10% validation, stratified randomly with `seed=42`.
- Hyperparameters:

| Hyperparameter | Value |
| --- | --- |
| Epochs | 5 |
| Learning rate | 1e-5 (cosine schedule with `warmup_steps=500`) |
| Optimizer | AdamW (default β) |
| Batch size | 8 (train) / 4 (eval), gradient accumulation 2 → effective 16 |
| Weight decay | 0.01 |
| Mixed precision | FP16 |
| Gradient checkpointing | Enabled by default in Whisper |
| Evaluation | Every 200 steps with generation + WER |

- Checkpointing: Saves best-by-loss every 200 steps (`save_total_limit=2`). Resume supported via `--resume_from_checkpoint`.

## 4. Evaluation Protocol

1. Download FLEURS Hindi (`google/fleurs`, config `hi_in`), keep the official test split.
2. Resample audio to 16 kHz via 🤗 Datasets audio casting.
3. For each model:
   - Encode features with `WhisperProcessor`.
   - Generate greedy predictions (`model.generate` with default decoding).
   - Compute Word Error Rate using `evaluate.load("wer")`.
4. Report table (values as raw ratios):

| Model | Hindi WER |
| --- | --- |
| Whisper Small (Pretrained) | 0.83 (given baseline, verify via script) |
| FT Whisper Small (yours) | 0.63 |

## 5. Suggested Extensions (Optional)

- **SpecAugment / noise simulation:** apply time masking and additive background noise for robustness.
- **VAD chunking:** long (>15m) talks can be segmented into 30s windows to stabilize training.
- **LoRA fine-tuning:** if GPU memory is constrained, adapt only small rank matrices.
- **Error analysis:** compute per-speaker or per-duration WER to highlight weak spots.

## 6. Deliverables Checklist

- [ ] `data/joshtalks_hi/manifest.jsonl`
- [ ] `checkpoints/whisper-small-hi/` (best checkpoint)
- [ ] `report.md` updated with final WER + observations
- [ ] Screenshots or logs showing training/eval completion (optional but recommended)

Fill in the final WER before uploading the report + code bundle to the Google Form.


