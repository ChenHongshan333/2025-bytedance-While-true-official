#!/usr/bin/env python
# benchmark_eval.py
# Evaluate fine-tuned BERT review classifier

import argparse
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

CATEGORIES = ["valid", "advertisement", "irrelevant", "rants without visit"]

def evaluate_model(model_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load dataset (replace with your own dataset path if needed)
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files={"validation": "val.csv"})  # expects val.csv
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer
    )

    print("Running evaluation...")
    predictions = trainer.predict(tokenized_dataset["validation"])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # Save per-sample predictions
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    eval_df = pd.DataFrame({
        "true_label": [CATEGORIES[i] for i in y_true],
        "pred_label": [CATEGORIES[i] for i in y_pred],
        "confidence": probs.max(axis=1),
        "correct": (y_true == y_pred)
    })
    eval_path = os.path.join(output_dir, "validation_predictions.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved predictions to {eval_path}")

    # Save classification metrics
    report = classification_report(y_true, y_pred, target_names=CATEGORIES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    metrics_path = os.path.join(output_dir, "classification_metrics.csv")
    report_df.to_csv(metrics_path)
    print(f"Saved metrics to {metrics_path}")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES)
    cm_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    accuracy = (y_true == y_pred).mean()
    print(f"\nFinal Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark evaluation script")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Where to save results")
    args = parser.parse_args()

    evaluate_model(args.model_name, args.output_dir)
