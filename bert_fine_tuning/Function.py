import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Union, Dict, List

class ReviewClassifier:
    """Review classifier using fine-tuned BERT model from Hugging Face"""
    
    def __init__(self, model_name = "laikexi/bert-review-classifier"):
        """
        Initialize the classifier with a Hugging Face model
        
        Args:
            model_name: Hugging Face model name (e.g., "your-username/bert-review-classifier")
        """
        self.model_name = model_name
        self.categories = ["valid", "advertisement", "irrelevant", "rants without visit"]
        
        print(f"Loading model from Hugging Face: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
    
    def classify_single_comment(self, business_name: str, comment: str, rating: Union[int, float, None] = None) -> Dict[str, Union[str, float]]:
        """
        Classify a single comment and determine if it's valid or invalid
        
        Args:
            business_name: Name of the business being reviewed
            comment: The review text to classify
            rating: Optional rating (1-5)
            
        Returns:
            Dictionary containing:
            - 'result': 'valid' or 'invalid'
            - 'category': specific category name
            - 'confidence': confidence score (0-1)
        """
        # Format input text same as training
        if rating is not None:
            text = f"Business: {business_name} | Rating: {rating}/5 | Review: {comment}"
        else:
            text = f"Business: {business_name} | Review: {comment}"
        
        # Tokenize and predict
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        category = self.categories[predicted_class]
        
        # Determine if valid or invalid
        if category == "valid":
            result = "valid"
        else:
            result = "invalid"
        
        return {
            'result': result,
            'category': category,
            'confidence': confidence
        }
    
    def classify_csv(self, 
                    input_csv_path: str, 
                    output_csv_path: str, 
                    text_column: str = 'text',
                    business_column: str = 'business_name',
                    rating_column: Union[str, None] = 'rating') -> pd.DataFrame:
        """
        Process a CSV file and add classification results
        
        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path to save output CSV file
            text_column: Name of column containing review text
            business_column: Name of column containing business names (optional, will use placeholder if not found)
            rating_column: Name of column containing ratings (optional)
            
        Returns:
            DataFrame with added classification columns
        """
        # Load CSV
        try:
            df = pd.read_csv(input_csv_path)
            print(f"Loaded CSV with {len(df)} rows")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
        
        # Check required columns
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Handle missing business column
        if business_column not in df.columns:
            print(f"Business column '{business_column}' not found. Using placeholder business name.")
            df[business_column] = "Unknown Business"
        
        # Handle missing rating column
        use_ratings = rating_column is not None and rating_column in df.columns
        if rating_column and not use_ratings:
            print(f"Rating column '{rating_column}' not found. Proceeding without ratings.")
        
        # Initialize result columns
        df['classification_result'] = ''
        df['classification_category'] = ''
        df['classification_confidence'] = 0.0
        
        # Process each row
        print("Processing reviews...")
        batch_size = 32  # Process in batches for efficiency
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end]
            
            # Prepare batch texts
            batch_texts = []
            for _, row in batch_df.iterrows():
                business = str(row[business_column])
                text = str(row[text_column])
                
                if use_ratings and pd.notna(row[rating_column]):
                    rating = row[rating_column]
                    formatted_text = f"Business: {business} | Rating: {rating}/5 | Review: {text}"
                else:
                    formatted_text = f"Business: {business} | Review: {text}"
                    
                batch_texts.append(formatted_text)
            
            # Batch prediction
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1).cpu().numpy()
                confidences = torch.max(probabilities, dim=-1).values.cpu().numpy()
            
            # Update dataframe
            for j, (pred_class, confidence) in enumerate(zip(predicted_classes, confidences)):
                idx = i + j
                category = self.categories[pred_class]
                result = "valid" if category == "valid" else "invalid"
                
                df.at[idx, 'classification_result'] = result
                df.at[idx, 'classification_category'] = category
                df.at[idx, 'classification_confidence'] = float(confidence)
            
            if (i + batch_size) % 100 == 0 or batch_end == len(df):
                print(f"Processed {batch_end}/{len(df)} rows")
        
        # Save results
        df.to_csv(output_csv_path, index=False)
        print(f"Results saved to: {output_csv_path}")
        
        # Print summary
        print("\nClassification Summary:")
        print(df['classification_result'].value_counts())
        print(f"\nCategory Breakdown:")
        print(df['classification_category'].value_counts())
        
        return df

# Convenience functions
def classify_single_review(model_name: str, business_name: str, comment: str, rating: Union[int, float, None] = None) -> Dict[str, Union[str, float]]:
    """
    Classify a single review comment
    
    Args:
        model_name: Hugging Face model name
        business_name: Name of the business
        comment: Review text
        rating: Optional rating (1-5)
        
    Returns:
        Classification result dictionary
    """
    classifier = ReviewClassifier(model_name)
    return classifier.classify_single_comment(business_name, comment, rating)

def classify_csv_file(model_name: str, 
                     input_csv: str, 
                     output_csv: str,
                     text_column: str = 'text',
                     business_column: str = 'business_name',
                     rating_column: Union[str, None] = 'rating') -> pd.DataFrame:
    """
    Classify reviews in a CSV file
    
    Args:
        model_name: Hugging Face model name
        input_csv: Path to input CSV
        output_csv: Path to output CSV
        text_column: Name of text column
        business_column: Name of business column
        rating_column: Name of rating column (optional)
        
    Returns:
        DataFrame with classification results
    """
    classifier = ReviewClassifier(model_name)
    return classifier.classify_csv(input_csv, output_csv, text_column, business_column, rating_column)

# Example usage
if __name__ == "__main__":
    # Replace with your actual model name
    MODEL_NAME = "laikexi/bert-review-classifier"
    
    # Example 1: Single comment classification
    print("=== Single Comment Classification ===")
    result = classify_single_review(
        model_name=MODEL_NAME,
        business_name="Joe's Pizza",
        comment="He's a very beautiful guy",
        rating=5
    )
    print(f"Result: {result}")
    
    # Example 2: CSV file classification
    # print("\n=== CSV File Classification ===")
    # # Assumes you have 'input_reviews.csv' with required columns
    # try:
    #     df_results = classify_csv_file(
    #         model_name=MODEL_NAME,
    #         input_csv="input_reviews.csv",
    #         output_csv="classified_reviews.csv",
    #         text_column="text",
    #         rating_column="rating"
    #     )
    # except Exception as e:
    #     print(f"CSV classification failed: {e}")
    #     print("Make sure you have an input CSV file with the correct columns.")