import argparse
from pathlib import Path

import evaluate
from datasets import Audio, load_dataset
from transformers import (
    AutoProcessor,
    DataCollatorSpeechSeq2SeqWithPadding,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-small on Josh Talks data.")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest.jsonl.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/whisper-small-hi")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))
    dataset = dataset.rename_column("audio_filepath", "audio")
    return dataset


def prepare_dataset(batch, processor: WhisperProcessor):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


def main() -> None:
    args = parse_args()
    dataset = load_manifest(args.manifest_path)
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)

    processor: WhisperProcessor = AutoProcessor.from_pretrained(
        "openai/whisper-small", language="hi", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    remove_cols = dataset["train"].column_names
    train_dataset = dataset["train"].map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=remove_cols,
    )
    eval_dataset = dataset["test"].map(
        prepare_dataset,
        fn_kwargs={"processor": processor},
        remove_columns=remove_cols,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer_score}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is None:
        last_ckpt = get_last_checkpoint(args.output_dir) if Path(args.output_dir).exists() else None
        resume_ckpt = last_ckpt

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=None,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()


