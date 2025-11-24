import argparse
from typing import Dict

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoProcessor, WhisperForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper checkpoints on FLEURS hi-IN.")
    parser.add_argument("--checkpoint_pretrained", type=str, default="openai/whisper-small")
    parser.add_argument("--checkpoint_ft", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_eval_records", type=int, default=None)
    return parser.parse_args()


def prepare_dataset():
    dataset = load_dataset("google/fleurs", "hi_in", split="test")
    dataset = dataset.cast_column("audio", dataset.features["audio"].clone().cast_audio(sampling_rate=16000))
    return dataset


@torch.no_grad()
def run_eval(model_name: str, processor_name: str, dataset, batch_size: int, device: str, limit: int):
    processor = AutoProcessor.from_pretrained(processor_name, language="hi", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    wer_metric = evaluate.load("wer")
    predictions, references = [], []

    for batch_start in range(0, len(dataset) if not limit else min(limit, len(dataset)), batch_size):
        batch = dataset[batch_start : batch_start + batch_size]
        inputs = processor.feature_extractor(
            [sample["array"] for sample in batch["audio"]],
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device)
        generated_ids = model.generate(inputs)
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        ref_text = batch["transcription"]
        predictions.extend(pred_text)
        references.extend(ref_text)

    wer_value = wer_metric.compute(predictions=predictions, references=references)
    return wer_value


def main():
    args = parse_args()
    dataset = prepare_dataset()
    limit = args.max_eval_records

    results: Dict[str, float] = {}
    results["Whisper Small (Pretrained)"] = run_eval(
        args.checkpoint_pretrained,
        args.checkpoint_pretrained,
        dataset,
        args.batch_size,
        args.device,
        limit,
    )
    results["FT Whisper Small (yours)"] = run_eval(
        args.checkpoint_ft,
        args.checkpoint_ft,
        dataset,
        args.batch_size,
        args.device,
        limit,
    )

    print("\nValues are raw WER ratios (e.g., 0.30 = 30%)\n")
    print(f"{'Model':40s}WER")
    for name, value in results.items():
        print(f"{name:40s}{value:.4f}")


if __name__ == "__main__":
    main()


