
# Configuration
# INPUT_CSV_PATH="data/collect_data/test_gt.csv"
# TRAIN_CSV_PATH="data/google_local-Alabama/train_gt.csv"
# OUTPUT_FOLDER="results/collect_data/test_gt/Qwen3-8B"
# VLLM_HOST="localhost"
# VLLM_PORT=8501
# VLLM_MODEL="Qwen/Qwen3-8B"
# BACKEND="vllm"
# STORE_COLUMN="store_info"
# COMMENT_COLUMN="text_info"
# WORKERS=64
# BENCHMARK_FOLDER="results/collect_data/test_gt/Qwen3-8B_benchmark"
# RAG_INDEX_FILE="embeddings/train_index.faiss"
# RAG_K=5

INPUT_CSV_PATH="data/collect_data/test_gt.csv"
TRAIN_CSV_PATH="data/google_local-Alabama/train_gt.csv"
OUTPUT_FOLDER="results/collect_data/test_gt/Qwen2.5-0.5B-Instruct"
VLLM_HOST="localhost"
VLLM_PORT=8501
VLLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
BACKEND="vllm"
STORE_COLUMN="store_info"
COMMENT_COLUMN="text_info"
WORKERS=64
BENCHMARK_FOLDER="results/collect_data/test_gt/Qwen2.5-0.5B-Instruct_benchmark"
RAG_INDEX_FILE="embeddings/train_index.faiss"
RAG_K=5

# Create output directories if they don't exist
mkdir -p "$OUTPUT_FOLDER"
mkdir -p "$BENCHMARK_FOLDER"
mkdir -p "embeddings"

echo "🏗️  Building FAISS index for RAG..."
# Check if RAG index already exists
if [ ! -f "$RAG_INDEX_FILE" ]; then
    echo "📊 Creating FAISS embedding index from training data..."
    python embedding_indexer.py \
        --data-file "$TRAIN_CSV_PATH" \
        --index-file "$RAG_INDEX_FILE" \
        --text-column "$COMMENT_COLUMN" \
        --classification-column "ai_classification"
    
    if [ $? -eq 0 ]; then
        echo "✅ FAISS index created successfully!"
    else
        echo "❌ Failed to create FAISS index. Exiting..."
        exit 1
    fi
else
    echo "✅ FAISS index already exists: $RAG_INDEX_FILE"
fi

echo "🚀 Starting benchmark with all templates..."

# Template 1: Zero-shot
python review_classifier.py \
    --csv-file "$INPUT_CSV_PATH" \
    --output-file "$OUTPUT_FOLDER/zero_shot_results.csv" \
    --backend "$BACKEND" \
    --vllm-host "$VLLM_HOST" \
    --vllm-port "$VLLM_PORT" \
    --vllm-model "$VLLM_MODEL" \
    --store-column "$STORE_COLUMN" \
    --comment-column "$COMMENT_COLUMN" \
    --template zero_shot \
    --workers "$WORKERS" \
    --verbose

python benchmark.py \
    --gt-file "$INPUT_CSV_PATH" \
    --pred-file "$OUTPUT_FOLDER/zero_shot_results.csv" \
    --output-dir "$BENCHMARK_FOLDER" \
    --template-name "zero_shot"

# # Template 2: Few-shot (balanced)
python review_classifier.py \
    --csv-file "$INPUT_CSV_PATH" \
    --output-file "$OUTPUT_FOLDER/few_shot_results.csv" \
    --backend "$BACKEND" \
    --vllm-host "$VLLM_HOST" \
    --vllm-port "$VLLM_PORT" \
    --vllm-model "$VLLM_MODEL" \
    --store-column "$STORE_COLUMN" \
    --comment-column "$COMMENT_COLUMN" \
    --template few_shot \
    --workers "$WORKERS"

# Benchmark few-shot
python benchmark.py \
    --gt-file "$INPUT_CSV_PATH" \
    --pred-file "$OUTPUT_FOLDER/few_shot_results.csv" \
    --output-dir "$BENCHMARK_FOLDER" \
    --template-name "few_shot"

# # Template 3: Few-shot + Chain-of-Thought
python review_classifier.py \
    --csv-file "$INPUT_CSV_PATH" \
    --output-file "$OUTPUT_FOLDER/few_shot_cot_results.csv" \
    --backend "$BACKEND" \
    --vllm-host "$VLLM_HOST" \
    --vllm-port "$VLLM_PORT" \
    --vllm-model "$VLLM_MODEL" \
    --store-column "$STORE_COLUMN" \
    --comment-column "$COMMENT_COLUMN" \
    --template few_shot_cot \
    --workers "$WORKERS"

# Benchmark few-shot + CoT
python benchmark.py \
    --gt-file "$INPUT_CSV_PATH" \
    --pred-file "$OUTPUT_FOLDER/few_shot_cot_results.csv" \
    --output-dir "$BENCHMARK_FOLDER" \
    --template-name "few_shot_cot"

echo "🔍 Running COT RAG template..."
# Template 4: COT RAG (Chain-of-Thought + Retrieval Augmented Generation)
python review_classifier.py \
    --csv-file "$INPUT_CSV_PATH" \
    --output-file "$OUTPUT_FOLDER/cot_rag_results.csv" \
    --backend "$BACKEND" \
    --vllm-host "$VLLM_HOST" \
    --vllm-port "$VLLM_PORT" \
    --vllm-model "$VLLM_MODEL" \
    --store-column "$STORE_COLUMN" \
    --comment-column "$COMMENT_COLUMN" \
    --template cot_rag \
    --rag-index-file "$RAG_INDEX_FILE" \
    --rag-k "$RAG_K" \
    --workers "$WORKERS"

# Benchmark COT RAG
python benchmark.py \
    --gt-file "$INPUT_CSV_PATH" \
    --pred-file "$OUTPUT_FOLDER/cot_rag_results.csv" \
    --output-dir "$BENCHMARK_FOLDER" \
    --template-name "cot_rag"

echo "🎉 All benchmarks completed!"
echo "📊 Results saved in: $BENCHMARK_FOLDER"