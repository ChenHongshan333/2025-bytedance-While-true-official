#!/usr/bin/env python3
"""
Embedding Indexer for RAG System
"""

import os
import click
import pandas as pd
import numpy as np
import pickle
import faiss
from openai import OpenAI
from dotenv import load_dotenv
import time
from typing import List, Tuple, Dict, Any
import json


def load_environment():
    """Load environment variables from .env file if it exists."""
    try:
        load_dotenv()
    except Exception:
        pass  # .env file doesn't exist, that's okay


class EmbeddingIndexer:
    """Create and manage embeddings index for RAG system."""
    
    def __init__(self, openai_client: OpenAI, embedding_model: str = "text-embedding-3-small"):
        self.client = openai_client
        self.embedding_model = embedding_model
        self.index = None
        self.data = None
        self.embedding_dim = None
        
    def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        Get embedding for a single text using OpenAI API.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retries on API failure
            
        Returns:
            List of embedding values
        """
        for attempt in range(max_retries):
            try:
                # Clean the text
                text = str(text).strip()
                if not text:
                    text = "No content"
                
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
                
            except Exception as e:
                if attempt == max_retries - 1:
                    click.echo(f"❌ Failed to get embedding after {max_retries} attempts: {e}")
                    # Return zero vector as fallback
                    if self.embedding_dim is None:
                        # Default dimension for text-embedding-3-small
                        self.embedding_dim = 1536
                    return [0.0] * self.embedding_dim
                else:
                    click.echo(f"⚠️  Attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            
        Returns:
            NumPy array of embeddings
        """
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with click.progressbar(length=len(texts), label='Getting embeddings') as bar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = []
                
                # Process each text in the batch
                for text in batch_texts:
                    embedding = self.get_embedding(text)
                    batch_embeddings.append(embedding)
                    bar.update(1)
                
                embeddings.extend(batch_embeddings)
                
                # Set embedding dimension from first successful embedding
                if self.embedding_dim is None and batch_embeddings:
                    self.embedding_dim = len(batch_embeddings[0])
        
        return np.array(embeddings, dtype=np.float32)
    
    def build_index(self, data_file: str, text_column: str = 'text_info', 
                   classification_column: str = 'ai_classification') -> None:
        """
        Build FAISS index from the training data.
        
        Args:
            data_file: Path to the CSV file containing training data
            text_column: Name of the column containing text to embed
            classification_column: Name of the column containing classifications
        """
        click.echo(f"📂 Loading data from {data_file}")
        
        # Load the data
        df = pd.read_csv(data_file)
        
        # Validate required columns
        required_columns = [text_column, classification_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Filter out rows with null values in required columns
        initial_rows = len(df)
        df = df.dropna(subset=[text_column, classification_column])
        filtered_rows = len(df)
        
        if initial_rows != filtered_rows:
            click.echo(f"🔍 Filtered out {initial_rows - filtered_rows} rows with null values")
        
        click.echo(f"📊 Processing {filtered_rows} rows")
        
        # Store the data for later retrieval
        self.data = df.reset_index(drop=True)
        
        # Extract texts to embed
        texts = self.data[text_column].astype(str).tolist()
        
        # Get embeddings
        click.echo(f"🔄 Getting embeddings using {self.embedding_model}")
        embeddings = self.get_embeddings_batch(texts)
        
        # Build FAISS index
        click.echo("🏗️  Building FAISS index")
        self.embedding_dim = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        click.echo(f"✅ Index built successfully with {self.index.ntotal} vectors")
    
    def save_index(self, index_file: str, metadata_file: str = None) -> None:
        """
        Save the FAISS index and associated metadata.
        
        Args:
            index_file: Path to save the FAISS index
            metadata_file: Path to save the metadata (optional, auto-generated if not provided)
        """
        if self.index is None or self.data is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        if metadata_file is None:
            metadata_file = index_file.replace('.faiss', '_metadata.pkl')
        
        metadata = {
            'data': self.data,
            'embedding_model': self.embedding_model,
            'embedding_dim': self.embedding_dim,
            'total_vectors': self.index.ntotal
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        click.echo(f"💾 Index saved to: {index_file}")
        click.echo(f"💾 Metadata saved to: {metadata_file}")
    
    def load_index(self, index_file: str, metadata_file: str = None) -> None:
        """
        Load a previously saved FAISS index and metadata.
        
        Args:
            index_file: Path to the FAISS index file
            metadata_file: Path to the metadata file (optional, auto-generated if not provided)
        """
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        if metadata_file is None:
            metadata_file = index_file.replace('.faiss', '_metadata.pkl')
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.data = metadata['data']
        self.embedding_model = metadata['embedding_model']
        self.embedding_dim = metadata['embedding_dim']
        
        click.echo(f"📂 Index loaded from: {index_file}")
        click.echo(f"📊 Loaded {self.index.ntotal} vectors")
        click.echo(f"🤖 Embedding model: {self.embedding_model}")


@click.command()
@click.option(
    '--data-file',
    required=True,
    type=click.Path(exists=True),
    help='Path to the CSV file containing training data'
)
@click.option(
    '--index-file',
    default='embeddings/train_index.faiss',
    help='Path to save the FAISS index (default: embeddings/train_index.faiss)'
)
@click.option(
    '--text-column',
    default='text_info',
    help='Name of the column containing text to embed (default: text_info)'
)
@click.option(
    '--classification-column',
    default='ai_classification',
    help='Name of the column containing classifications (default: ai_classification)'
)
@click.option(
    '--embedding-model',
    default='text-embedding-3-small',
    help='OpenAI embedding model to use (default: text-embedding-3-small)'
)
@click.option(
    '--force-rebuild',
    is_flag=True,
    help='Force rebuild even if index already exists'
)
def main(data_file, index_file, text_column, classification_column, embedding_model, force_rebuild):
    """
    Build FAISS embedding index from training data for RAG system.
    
    This tool creates a FAISS index from the text_info column in the training CSV file,
    using OpenAI embeddings. The index is used for retrieving similar examples in the
    COT RAG (Chain of Thought Retrieval Augmented Generation) prompt template.
    
    Requirements:
    - OpenAI API key set as OPENAI_API_KEY environment variable
    - Training CSV file with text_info and ai_classification columns
    
    The tool will create:
    - FAISS index file (.faiss)
    - Metadata file (.pkl) containing the original data and model info
    """
    try:
        # Load environment variables
        load_environment()
        
        # Check if index already exists
        if os.path.exists(index_file) and not force_rebuild:
            click.echo(f"⚠️  Index file already exists: {index_file}")
            click.echo("Use --force-rebuild to recreate the index")
            return
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or create a .env file with your API key."
            )
        
        client = OpenAI(api_key=api_key)
        
        # Initialize indexer
        indexer = EmbeddingIndexer(client, embedding_model)
        
        click.echo(f"🚀 Starting embedding index creation")
        click.echo(f"📂 Data file: {data_file}")
        click.echo(f"💾 Index file: {index_file}")
        click.echo(f"📝 Text column: {text_column}")
        click.echo(f"🏷️  Classification column: {classification_column}")
        click.echo(f"🤖 Embedding model: {embedding_model}")
        click.echo("")
        
        start_time = time.time()
        
        # Build the index
        indexer.build_index(data_file, text_column, classification_column)
        
        # Save the index
        indexer.save_index(index_file)
        
        elapsed_time = time.time() - start_time
        click.echo(f"\n⚡ Index creation completed in {elapsed_time:.2f} seconds")
        
        # Show summary
        click.echo(f"\n📈 Index Summary:")
        click.echo(f"   Total vectors: {indexer.index.ntotal}")
        click.echo(f"   Embedding dimension: {indexer.embedding_dim}")
        click.echo(f"   Model used: {indexer.embedding_model}")
        
        # Show classification distribution
        if indexer.data is not None:
            classification_counts = indexer.data[classification_column].value_counts()
            click.echo(f"\n📊 Classification Distribution:")
            for category, count in classification_counts.items():
                click.echo(f"   {category}: {count}")
        
    except ValueError as e:
        click.echo(f"❌ Configuration Error: {e}", err=True)
        click.echo("\n💡 Setup Instructions:")
        click.echo("1. Get your API key from https://platform.openai.com/api-keys")
        click.echo("2. Set environment variable: export OPENAI_API_KEY='your_key_here'")
        click.echo("3. Or create a .env file with: OPENAI_API_KEY=your_key_here")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


if __name__ == '__main__':
    main()
