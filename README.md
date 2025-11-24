# Josh Talks ASR Take-Home

This repository contains a completely reproducible workflow to solve the Josh Talks AI Research Intern task:

1. **Pre-process the ~10hrs Hindi ASR dataset** shipped as `FT Data - data.csv`.
2. **Fine-tune Whisper-small** on the cleaned data.
3. **Evaluate** the pretrained baseline and the fine-tuned checkpoint on the Hindi FLEURS test split and report WER.

The code is written to be executed on a CUDA machine (>=24 GB VRAM recommended for full-batch fine-tuning). Replace any placeholders (e.g. Hugging Face token) before running.

## Repository Layout

```
.
├─ FT Data - data.csv              # Provided manifest (do not edit)
├─ scripts/
│  ├─ preprocess_dataset.py        # Downloads audio + labels, builds HF-ready manifest
│  ├─ train_whisper_small.py       # Fine-tunes Whisper-small w/ transformers Trainer
│  └─ eval_whisper_small.py        # Benchmarks baseline + fine-tuned checkpoints
├─ requirements.txt                # Minimal Python dependencies
└─ report.md                       # Methodology, hyper-params, and result template
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # (use .venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 1. Pre-processing

```bash
python scripts/preprocess_dataset.py \
  --csv_path "FT Data - data.csv" \
  --output_dir data/joshtalks_hi \
  --sample_rate 16000 \
  --max_workers 8
```

Outputs:

- `data/joshtalks_hi/audio/*.wav`
- `data/joshtalks_hi/transcripts/*.json`
- `data/joshtalks_hi/manifest.jsonl` (training manifest)
- `data/joshtalks_hi/dataset_summary.json`

### 2. Fine-tuning Whisper-small

```bash
python scripts/train_whisper_small.py \
  --manifest_path data/joshtalks_hi/manifest.jsonl \
  --output_dir checkpoints/whisper-small-hi \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --warmup_steps 500
```

This script performs an 90/10 train-validation split, logs to TensorBoard, and saves the best checkpoint by validation loss.

### 3. Evaluation (FLEURS hi-IN)

```bash
python scripts/eval_whisper_small.py \
  --checkpoint_pretrained openai/whisper-small \
  --checkpoint_ft checkpoints/whisper-small-hi \
  --batch_size 4 \
  --device cuda
```

The script downloads the Hindi (`hi_in`) FLEURS test split via 🤗 Datasets, evaluates both checkpoints, and prints a WER table that can be pasted into the submission form.

## Submission Guidance

Fill `report.md` with the final WER (pretrained baseline already provided as 0.83 in the assignment) and attach any additional observations or error analysis you perform. Upload the report plus the code (or a GitHub link) alongside the Google Form.

## Notes

- The storage bucket occasionally throttles public downloads. The preprocessing script retries failed downloads up to 5 times and supports partial re-runs (existing files are skipped).
- If you need to resume training, point `--resume_from_checkpoint` to the last saved directory.
- For experimentation on lower memory GPUs, reduce `--per_device_train_batch_size`, enable `--gradient_checkpointing`, or switch to 8-bit optimizers (Accelerate / bitsandbytes).

Good luck! 🎧


