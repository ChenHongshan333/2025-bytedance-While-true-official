# Harmonia: Hierarchical Multimodal Hybrid Review Validation with Retrieval-Augmented Generation

## Environment Setup
```bash
# Create and activate environment
conda create -n review_classifier python=3.10 -y
conda activate review_classifier

# Install dependencies
pip install -r requirements.txt

# Process data
bash exp/data_process.sh
```

## Quick Start

### 1. COT RAG (Recommended)
```bash
# Build FAISS index
python embedding_indexer.py --data-file data/google_local-Alabama/train_gt.csv

# Single classification
python review_classifier.py --store-info "Pizza restaurant" --user-comment "Great food!" --template cot_rag --rag-index-file embeddings/train_index.faiss

# Batch processing
python review_classifier.py --csv-file data/collect_data/test_gt.csv --template cot_rag --rag-index-file embeddings/train_index.faiss --store-column store_info --comment-column text_info

# Run full benchmark
bash exp/benchmark.sh
```

### 2. BERT Fine-tuning
```bash
# Run BERT benchmark
chmod +x bert_fine_tuning/benchmark.sh
bash bert_fine_tuning/benchmark.sh
```

## Methods

### COT RAG (Chain-of-Thought + Retrieval Augmented Generation)
- **Files**: `embedding_indexer.py`, `rag_retriever.py`, `prompt_templates.py`, `review_classifier.py`
- **Features**: Semantic similarity search with OpenAI embeddings, dynamic example retrieval, chain-of-thought reasoning
- **Best for**: High accuracy with contextual understanding

### BERT Fine-tuning
- **Files**: `bert_fine_tuning/` directory
- **Features**: Fine-tuned BERT model for classification
- **Model**: `laikexi/bert_review_classifier`
- **Best for**: Fast inference, traditional ML approach

## Templates
- `zero_shot`: Direct classification (fastest)
- `few_shot`: Static examples (balanced)
- `few_shot_cot`: Static examples + reasoning (accurate)
- `cot_rag`: Retrieved examples + reasoning (most accurate)

## BERT Fine-tuning Details

### Example Input Format:
```csv
text,business_name,rating,label
"Great service and food!",Restaurant A,5,valid
"Check out our new deals!",Restaurant B,4,advertisement
"Random comment here",Restaurant C,3,irrelevant
```

### Key Training Parameters:
- **max_length**: 128-256 tokens
- **per_device_train_batch_size**: 8-32
- **learning_rate**: 2e-5 to 5e-5
- **num_train_epochs**: 3-5
- **weight_decay**: 0.01
- **evaluation_strategy**: epoch or steps

### Using the Python API:

```python
from bert_fine_tuning.Function import ReviewClassifier

# Initialize classifier
classifier = ReviewClassifier(model_name="laikexi/bert-review-classifier")

# Classify single review
result = classifier.classify_single_comment(
    business_name="Restaurant A",
    comment="Amazing food and service!",
    rating=5
)
print(result)
# Output: {'result': 'valid', 'category': 'valid', 'confidence': 0.95}

# Process CSV file
classifier.classify_csv(
    input_csv_path="reviews.csv",
    output_csv_path="classified_reviews.csv"
)
```

## Evaluation and Benchmarking
```bash
# Run inference on a CSV file
cd bert_fine_tuning
bash benchmark.sh

# Or run evaluation manually
python benchmark_eval.py --model_name laikexi/bert-review-classifier --output_dir results/
```

## Model
Our model is saved as `laikexi/bert_review_classifier` on Hugging Face Hub.