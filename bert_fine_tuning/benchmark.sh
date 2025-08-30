#!/bin/bash
# benchmark.sh
# Script to benchmark fine-tuned BERT review classifier
# Usage: bash benchmark.sh

set -e  # Exit on error
set -o pipefail

MODEL_NAME="laikexi/bert-review-classifier"
OUTPUT_DIR="benchmark_results"
EVAL_SCRIPT="benchmark_eval.py"
LOG_FILE="${OUTPUT_DIR}/benchmark.log"

mkdir -p $OUTPUT_DIR

echo "============================================" | tee $LOG_FILE
echo " BERT Review Classifier Benchmark" | tee -a $LOG_FILE
echo " Model: $MODEL_NAME" | tee -a $LOG_FILE
echo " Timestamp: $(date)" | tee -a $LOG_FILE
echo " Output Dir: $OUTPUT_DIR" | tee -a $LOG_FILE
echo "============================================" | tee -a $LOG_FILE

# Step 1: Run evaluation (classification report, confusion matrix, etc.)
echo -e "\n[1/3] Running evaluation..." | tee -a $LOG_FILE
python $EVAL_SCRIPT \
  --model_name $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  2>&1 | tee -a $LOG_FILE

# Step 2: Run inference latency benchmark
echo -e "\n[2/3] Measuring inference latency..." | tee -a $LOG_FILE
python - <<'EOF' 2>&1 | tee -a $LOG_FILE
import time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "laikexi/bert-review-classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

sample_texts = ["This place was amazing! Highly recommended." for _ in range(128)]
inputs = tokenizer(sample_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model(**inputs)

# Measure latency
batch_sizes = [1, 8, 32, 64, 128]
print("\nBatch Size | Avg Latency (ms/sample) | Throughput (samples/s)")
print("-----------------------------------------------------------")
for bs in batch_sizes:
    batch = sample_texts[:bs]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    
    N = 50
    start = time.time()
    for _ in range(N):
        with torch.no_grad():
            _ = model(**inputs)
    end = time.time()
    
    total_time = (end - start)
    avg_latency = (total_time / (N * bs)) * 1000
    throughput = (N * bs) / total_time
    print(f"{bs:<10} | {avg_latency:>20.2f} | {throughput:>22.2f}")
EOF

# Step 3: Summary
echo -e "\n[3/3] Benchmark complete!" | tee -a $LOG_FILE
echo "Results stored in $OUTPUT_DIR and log file $LOG_FILE"
