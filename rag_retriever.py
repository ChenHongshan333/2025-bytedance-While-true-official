#!/usr/bin/env python3
"""
RAG Retriever for Review Classification
"""

import os
import numpy as np
import faiss
import pickle
from openai import OpenAI
from typing import List, Tuple, Dict, Any
import pandas as pd
from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env file if it exists."""
    try:
        load_dotenv()
    except Exception:
        pass  # .env file doesn't exist, that's okay


class RAGRetriever:
    """Retrieve similar examples using FAISS index for RAG system."""
    
    def __init__(self, index_file: str, metadata_file: str = None, 
                 openai_client: OpenAI = None, embedding_model: str = None):
        self.index_file = index_file
        self.metadata_file = metadata_file or index_file.replace('.faiss', '_metadata.pkl')
        self.index = None
        self.data = None
        self.embedding_model = embedding_model
        self.embedding_dim = None
        
        # Initialize OpenAI client if not provided
        if openai_client is None:
            load_environment()
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                    "or provide an OpenAI client instance."
                )
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = openai_client
        
        # Load the index and metadata
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the FAISS index and metadata."""
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Index file not found: {self.index_file}")
        
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_file)
        
        # Load metadata
        with open(self.metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.data = metadata['data']
        if self.embedding_model is None:
            self.embedding_model = metadata['embedding_model']
        self.embedding_dim = metadata['embedding_dim']
    
    def get_embedding(self, text: str, max_retries: int = 3) -> np.ndarray:
        """
        Get embedding for a query text using OpenAI API.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retries on API failure
            
        Returns:
            NumPy array of embedding values
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
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embedding.reshape(1, -1))
                
                return embedding
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to get embedding after {max_retries} attempts: {e}")
                else:
                    # Wait before retrying with exponential backoff
                    import time
                    time.sleep(2 ** attempt)
    
    def retrieve_similar(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar examples for a query text.
        
        Args:
            query_text: The text to find similar examples for
            k: Number of similar examples to retrieve
            
        Returns:
            List of dictionaries containing similar examples with their metadata
        """
        if self.index is None or self.data is None:
            raise ValueError("Index not loaded. Please check if index files exist.")
        
        # Get embedding for the query
        query_embedding = self.get_embedding(query_text)
        
        # Search for similar vectors
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Retrieve the corresponding data
        similar_examples = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.data):  # Valid index
                row = self.data.iloc[idx]
                example = {
                    'rank': i + 1,
                    'similarity_score': float(score),
                    'store_info': row.get('store_info', ''),
                    'text_info': row.get('text_info', ''),
                    'ai_classification': row.get('ai_classification', ''),
                    'gmap_id': row.get('gmap_id', ''),
                    'index': int(idx)
                }
                similar_examples.append(example)
        
        return similar_examples
    
    def format_examples_for_prompt(self, examples: List[Dict[str, Any]], 
                                 max_examples: int = 5) -> str:
        """
        Format retrieved examples for use in a prompt template.
        
        Args:
            examples: List of similar examples from retrieve_similar()
            max_examples: Maximum number of examples to include
            
        Returns:
            Formatted string ready to be inserted into a prompt
        """
        if not examples:
            return "No similar examples found."
        
        formatted_examples = []
        for example in examples[:max_examples]:
            formatted_example = f"""Business: "{example['store_info']}"
User comment: "{example['text_info']}"
Classification: {example['ai_classification']}"""
            formatted_examples.append(formatted_example)
        
        return "\n\n".join(formatted_examples)
    
    def get_classification_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get the distribution of classifications in the retrieved examples.
        
        Args:
            examples: List of similar examples from retrieve_similar()
            
        Returns:
            Dictionary with classification counts
        """
        distribution = {}
        for example in examples:
            classification = example.get('ai_classification', 'unknown')
            distribution[classification] = distribution.get(classification, 0) + 1
        
        return distribution
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded index.
        
        Returns:
            Dictionary containing index statistics
        """
        if self.index is None or self.data is None:
            return {"error": "Index not loaded"}
        
        stats = {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "data_shape": self.data.shape,
            "classification_distribution": self.data['ai_classification'].value_counts().to_dict() if 'ai_classification' in self.data.columns else {}
        }
        
        return stats


def create_retriever(index_file: str = "embeddings/train_index.faiss", 
                    openai_client: OpenAI = None) -> RAGRetriever:
    """
    Convenience function to create a RAG retriever.
    
    Args:
        index_file: Path to the FAISS index file
        openai_client: OpenAI client instance (optional)
        
    Returns:
        RAGRetriever instance
    """
    return RAGRetriever(index_file, openai_client=openai_client)


# Example usage and testing
if __name__ == "__main__":
    import click
    
    @click.command()
    @click.option('--index-file', default='embeddings/train_index.faiss', 
                  help='Path to the FAISS index file')
    @click.option('--query', required=True, help='Query text to search for similar examples')
    @click.option('--k', default=5, help='Number of similar examples to retrieve')
    def test_retriever(index_file, query, k):
        """Test the RAG retriever with a sample query."""
        try:
            retriever = RAGRetriever(index_file)
            
            click.echo(f"🔍 Searching for similar examples to: '{query}'")
            click.echo(f"📊 Index stats: {retriever.get_stats()}")
            click.echo("")
            
            # Retrieve similar examples
            examples = retriever.retrieve_similar(query, k)
            
            click.echo(f"📋 Found {len(examples)} similar examples:")
            click.echo("=" * 80)
            
            for example in examples:
                click.echo(f"\nRank {example['rank']} (Similarity: {example['similarity_score']:.4f})")
                click.echo(f"Business: {example['store_info'][:100]}...")
                click.echo(f"Comment: {example['text_info'][:100]}...")
                click.echo(f"Classification: {example['ai_classification']}")
                click.echo("-" * 40)
            
            # Show distribution
            distribution = retriever.get_classification_distribution(examples)
            click.echo(f"\n📈 Classification distribution in results:")
            for classification, count in distribution.items():
                click.echo(f"   {classification}: {count}")
            
            # Show formatted examples
            click.echo(f"\n📝 Formatted examples for prompt:")
            click.echo("=" * 80)
            formatted = retriever.format_examples_for_prompt(examples)
            click.echo(formatted)
            
        except Exception as e:
            click.echo(f"❌ Error: {e}")
    
    test_retriever()
