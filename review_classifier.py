#!/usr/bin/env python3
"""
GPT-4O Review Classification Demo

This script uses OpenAI's GPT-4O API to classify store reviews into categories.

Setup:
1. Install dependencies: pip install -r requirements.txt
2. Set your OpenAI API key as an environment variable:
   export OPENAI_API_KEY="your_api_key_here"
   
   Or create a .env file with:
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o

Usage:

Single review (OpenAI):
    python review_classifier.py --store-info "Store description" --user-comment "User's comment"
    python review_classifier.py --store-info "Store description" --user-comment "User's comment" --template zero_shot

Single review (vLLM):
    python review_classifier.py --store-info "Store description" --user-comment "User's comment" --backend vllm --vllm-host localhost --vllm-port 8000 --vllm-model llama-2-7b-chat

Batch processing (OpenAI):
    python review_classifier.py --csv-file data/demo_data.csv
    python review_classifier.py --csv-file data/demo_data.csv --output-file results.csv --template few_shot_cot

Batch processing (vLLM):
    python review_classifier.py --csv-file data/demo_data.csv --backend vllm --vllm-host localhost --vllm-port 8000 --vllm-model llama-2-7b-chat

Interactive mode:
    python review_classifier.py --template few_shot
"""

import os
import click
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import multiprocessing as mp
from functools import partial
import time
import json
import requests
from prompt_templates import (
    create_classification_prompt, 
    parse_cot_response, 
    parse_classification_response,
    get_available_templates, 
    get_template_description,
    TEMPLATE_INFO
)
from rag_retriever import RAGRetriever


def load_environment():
    """Load environment variables from .env file if it exists."""
    try:
        load_dotenv()
    except Exception:
        pass  # .env file doesn't exist, that's okay


class LLMBackend:
    """Flexible LLM backend that supports both OpenAI and vLLM."""
    
    def __init__(self, backend_type='openai', vllm_host=None, vllm_port=None, vllm_model=None):
        self.backend_type = backend_type.lower()
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.vllm_model = vllm_model
        
        if self.backend_type == 'openai':
            self._init_openai()
        elif self.backend_type == 'vllm':
            self._init_vllm()
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}. Use 'openai' or 'vllm'")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or create a .env file with your API key."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o')
    
    def _init_vllm(self):
        """Initialize vLLM configuration."""
        if not all([self.vllm_host, self.vllm_port, self.vllm_model]):
            raise ValueError(
                "For vLLM backend, you must provide vllm_host, vllm_port, and vllm_model"
            )
        
        # Build the vLLM API endpoint
        self.vllm_url = f"http://{self.vllm_host}:{self.vllm_port}/v1/chat/completions"
        self.model = self.vllm_model
        
        # Test connection
        try:
            test_response = requests.get(f"http://{self.vllm_host}:{self.vllm_port}/health", timeout=5)
            if test_response.status_code != 200:
                click.echo(f"⚠️  Warning: vLLM health check failed with status {test_response.status_code}")
        except requests.exceptions.RequestException as e:
            click.echo(f"⚠️  Warning: Could not connect to vLLM server: {e}")
    
    def chat_completion(self, messages, max_tokens=1024, temperature=0.1):
        """Make a chat completion request using the configured backend."""
        if self.backend_type == 'openai':
            return self._openai_completion(messages, max_tokens, temperature)
        elif self.backend_type == 'vllm':
            return self._vllm_completion(messages, max_tokens, temperature)
    
    def _openai_completion(self, messages, max_tokens, temperature):
        """Make OpenAI API completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1
        )
        return response.choices[0].message.content.strip()
    
    def _vllm_completion(self, messages, max_tokens, temperature):
        """Make vLLM API completion."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": 1
        }
        
        try:
            response = requests.post(self.vllm_url, json=payload, timeout=60)
            response.raise_for_status()
            
            response_data = response.json()
            if 'choices' not in response_data or not response_data['choices']:
                raise Exception("Invalid response format from vLLM server")
            
            return response_data['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"vLLM API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Invalid response format from vLLM: missing {str(e)}")
        except Exception as e:
            raise Exception(f"vLLM completion error: {str(e)}")


def get_llm_backend(backend_type='openai', vllm_host=None, vllm_port=None, vllm_model=None):
    """Initialize and return LLM backend."""
    return LLMBackend(backend_type, vllm_host, vllm_port, vllm_model)


def get_openai_client():
    """Initialize and return OpenAI client. (Deprecated - use get_llm_backend instead)"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or create a .env file with your API key."
        )
    return OpenAI(api_key=api_key)


# Removed - now using prompt_templates.py


def classify_review(llm_backend, store_info, user_comment, template_type='few_shot', 
                   return_raw_response=False, rag_retriever=None, rag_k=5):
    """Classify a review using the configured LLM backend."""
    
    # Handle RAG template
    retrieved_examples = None
    if template_type == 'cot_rag':
        if rag_retriever is None:
            raise ValueError("RAG retriever is required for cot_rag template type")
        
        # Retrieve similar examples
        similar_examples = rag_retriever.retrieve_similar(user_comment, k=rag_k)
        retrieved_examples = rag_retriever.format_examples_for_prompt(similar_examples, max_examples=rag_k)
    
    prompt = create_classification_prompt(store_info, user_comment, template_type, retrieved_examples)
    
    # Adjust max_tokens based on template type
    max_tokens = 1024
    if template_type == 'cot_rag':
        max_tokens = 2048  # More tokens for RAG responses
    
    try:
        raw_response = llm_backend.chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1  # Low temperature for consistent classification
        )
        
        classification = raw_response
        
        # Parse CoT response if needed (both few_shot_cot and cot_rag use CoT format)
        if template_type in ['few_shot_cot', 'cot_rag']:
            classification = parse_cot_response(raw_response)
        else:
            # Parse response to extract classification
            classification = parse_classification_response(raw_response)
        
        if return_raw_response:
            return classification, raw_response
        else:
            return classification
    
    except Exception as e:
        backend_name = llm_backend.backend_type.upper()
        raise Exception(f"Error calling {backend_name} API: {str(e)}")


# Global variables for multiprocessing
_backend_config = {}

def _init_worker_backend():
    """Initialize backend for worker process."""
    global _backend_config
    return get_llm_backend(
        backend_type=_backend_config.get('backend_type', 'openai'),
        vllm_host=_backend_config.get('vllm_host'),
        vllm_port=_backend_config.get('vllm_port'),
        vllm_model=_backend_config.get('vllm_model')
    )

def classify_single_row(row_data, store_column, comment_column, template_type='few_shot', 
                       backend_config=None, rag_config=None):
    """Classify a single row - designed for multiprocessing."""
    try:
        # Create a new backend for this process using passed config
        if backend_config:
            llm_backend = get_llm_backend(
                backend_type=backend_config.get('backend_type', 'openai'),
                vllm_host=backend_config.get('vllm_host'),
                vllm_port=backend_config.get('vllm_port'),
                vllm_model=backend_config.get('vllm_model')
            )
        else:
            # Fallback to global config (for backward compatibility)
            llm_backend = _init_worker_backend()
        
        # Initialize RAG retriever if needed
        rag_retriever = None
        rag_k = 5
        if template_type == 'cot_rag' and rag_config:
            rag_retriever = RAGRetriever(
                rag_config.get('index_file'),
                openai_client=llm_backend.client if hasattr(llm_backend, 'client') else None
            )
            rag_k = rag_config.get('rag_k', 5)
            
        index, row = row_data
        
        classification, raw_response = classify_review(
            llm_backend, 
            row[store_column], 
            row[comment_column],
            template_type,
            return_raw_response=True,
            rag_retriever=rag_retriever,
            rag_k=rag_k
        )
        return index, classification, raw_response
    except Exception as e:
        return index, f"ERROR: {str(e)}", f"ERROR: {str(e)}"


def process_file_batch(llm_backend, input_file, output_file=None, store_column='store_info', 
                      comment_column='user_comment', workers=16, template_type='few_shot',
                      rag_index_file=None, rag_k=5):
    """Process a CSV or JSON file with batch review classification."""
    try:
        # Determine file type and read accordingly
        file_extension = input_file.lower().split('.')[-1]
        
        if file_extension == 'csv':
            # Read CSV file
            df = pd.read_csv(input_file)
        elif file_extension == 'json':
            # Read JSON file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValueError("JSON file should contain a list of objects")
            
            # For JSON files, if store_column doesn't exist, create a placeholder
            if store_column not in df.columns:
                # Try common alternatives or create from available data
                if 'gmap_id' in df.columns:
                    df[store_column] = df['gmap_id'].astype(str)
                elif 'business_name' in df.columns:
                    df[store_column] = df['business_name']
                else:
                    df[store_column] = 'Unknown Business'
                    
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only CSV and JSON files are supported.")
        
        # Validate required columns
        required_columns = [store_column, comment_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Filter out rows where text column is null
        initial_rows = len(df)
        df = df.dropna(subset=[comment_column])
        filtered_rows = len(df)
        
        if initial_rows != filtered_rows:
            click.echo(f"🔍 Filtered out {initial_rows - filtered_rows} rows with null text values")
            click.echo(f"📊 Processing {filtered_rows} rows with valid text")
        
        # Reset index after filtering
        df = df.reset_index(drop=True)
        
        # Initialize results columns
        df['ai_classification'] = ''
        df['ai_raw_response'] = ''
        
        # Initialize RAG retriever if needed
        rag_retriever = None
        if template_type == 'cot_rag':
            if not rag_index_file:
                raise ValueError("RAG index file is required for cot_rag template type")
            
            click.echo(f"🔍 Loading RAG index from: {rag_index_file}")
            rag_retriever = RAGRetriever(
                rag_index_file,
                openai_client=llm_backend.client if hasattr(llm_backend, 'client') else None
            )
            click.echo(f"📊 RAG index stats: {rag_retriever.get_stats()}")

        # Process each row
        total_rows = len(df)
        click.echo(f"📊 Processing {total_rows} reviews with {workers} workers...")
        
        if workers == 1:
            # Single-threaded processing
            with click.progressbar(length=total_rows, label='Classifying reviews') as bar:
                for index, row in df.iterrows():
                    try:
                        # Classify this review
                        classification, raw_response = classify_review(
                            llm_backend, 
                            row[store_column], 
                            row[comment_column],
                            template_type,
                            return_raw_response=True,
                            rag_retriever=rag_retriever,
                            rag_k=rag_k
                        )
                        df.at[index, 'ai_classification'] = classification
                        df.at[index, 'ai_raw_response'] = raw_response
                        
                    except Exception as e:
                        click.echo(f"\n⚠️  Error processing row {index + 1}: {str(e)}")
                        df.at[index, 'ai_classification'] = 'ERROR'
                        df.at[index, 'ai_raw_response'] = f'ERROR: {str(e)}'
                    
                    bar.update(1)
        else:
            # Multi-threaded processing
            start_time = time.time()
            
            # Prepare backend configuration for worker processes
            backend_config = {
                'backend_type': llm_backend.backend_type,
                'vllm_host': llm_backend.vllm_host,
                'vllm_port': llm_backend.vllm_port,
                'vllm_model': llm_backend.vllm_model
            }
            
            # Prepare RAG configuration for worker processes
            rag_config = None
            if template_type == 'cot_rag':
                rag_config = {
                    'index_file': rag_index_file,
                    'rag_k': rag_k
                }
            
            # Set global backend configuration for worker processes (backward compatibility)
            global _backend_config
            _backend_config = backend_config
            
            # Prepare data for multiprocessing
            row_data = [(index, row) for index, row in df.iterrows()]
            
            # Create partial function with fixed arguments, passing backend_config explicitly
            classify_func = partial(classify_single_row, 
                                  store_column=store_column, 
                                  comment_column=comment_column,
                                  template_type=template_type,
                                  backend_config=backend_config,
                                  rag_config=rag_config)
            
            # Initialize results columns
            df['ai_classification'] = ''
            df['ai_raw_response'] = ''
            
            with click.progressbar(length=total_rows, label='Classifying reviews') as bar:
                with mp.Pool(workers) as pool:
                    # Process in chunks to update progress bar
                    chunk_size = max(1, total_rows // (workers * 4))  # 4 chunks per worker
                    results = []
                    
                    for i in range(0, total_rows, chunk_size):
                        chunk = row_data[i:i + chunk_size]
                        chunk_results = pool.map(classify_func, chunk)
                        results.extend(chunk_results)
                        bar.update(len(chunk))
                    
                    # Update dataframe with results
                    for index, classification, raw_response in results:
                        df.at[index, 'ai_classification'] = classification
                        df.at[index, 'ai_raw_response'] = raw_response
            
            elapsed_time = time.time() - start_time
            click.echo(f"\n⚡ Completed in {elapsed_time:.2f} seconds")
        
        # Save results
        if output_file is None:
            # Generate output filename based on input
            base_name = os.path.splitext(input_file)[0]
            # Always save as CSV regardless of input format
            output_file = f"{base_name}_classified.csv"
        
        # Ensure output is CSV format
        if not output_file.lower().endswith('.csv'):
            output_file = output_file.rsplit('.', 1)[0] + '.csv'
            
        df.to_csv(output_file, index=False)
        
        # Show summary
        click.echo(f"\n✅ Classification completed!")
        click.echo(f"📁 Results saved to: {output_file}")
        
        # Show classification summary
        classification_counts = df['ai_classification'].value_counts()
        click.echo(f"\n📈 Classification Summary:")
        for category, count in classification_counts.items():
            click.echo(f"   {category}: {count}")
        
        return output_file
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty: {input_file}")
    except Exception as e:
        raise Exception(f"Error processing CSV file: {str(e)}")


@click.command()
@click.option(
    '--store-info', 
    help='Information about the store being reviewed'
)
@click.option(
    '--user-comment', 
    help='The user comment to classify'
)
@click.option(
    '--csv-file',
    type=click.Path(exists=True),
    help='Path to CSV or JSON file for batch processing'
)
@click.option(
    '--output-file',
    help='Output file path for batch processing results (optional, auto-generated if not provided)'
)
@click.option(
    '--store-column',
    default='gmap_id',
    help='Name of the column containing store/business information (default: gmap_id for JSON, business_name for CSV)'
)
@click.option(
    '--comment-column',
    default='text',
    help='Name of the column containing user comments (default: text)'
)
@click.option(
    '--workers',
    default=16,
    type=int,
    help='Number of parallel workers for batch processing (default: 16, use 1 for single-threaded)'
)
@click.option(
    '--verbose', 
    is_flag=True,
    help='Show detailed output including the prompt sent to GPT'
)
@click.option(
    '--template',
    type=click.Choice(get_available_templates()),
    default='few_shot',
    help='Prompt template to use for classification'
)
@click.option(
    '--backend',
    type=click.Choice(['openai', 'vllm']),
    default='openai',
    help='LLM backend to use (default: openai)'
)
@click.option(
    '--vllm-host',
    help='vLLM server host (required for vllm backend)'
)
@click.option(
    '--vllm-port',
    type=int,
    help='vLLM server port (required for vllm backend)'
)
@click.option(
    '--vllm-model',
    help='vLLM model name (required for vllm backend)'
)
@click.option(
    '--rag-index-file',
    help='Path to FAISS index file for RAG (required for cot_rag template)'
)
@click.option(
    '--rag-k',
    default=5,
    type=int,
    help='Number of similar examples to retrieve for RAG (default: 5)'
)
def main(store_info, user_comment, csv_file, output_file, store_column, comment_column, workers, verbose, template, backend, vllm_host, vllm_port, vllm_model, rag_index_file, rag_k):
    """
    Classify a store review using OpenAI or vLLM backends.
    
    This tool can work in two modes:
    
    1. Single review classification:
       Provide --store-info and --user-comment to classify a single review.
       
    2. Batch file processing:
       Provide --csv-file to process multiple reviews from a CSV or JSON file.
       CSV files must have the specified store and comment columns.
       JSON files should contain a list of objects with a 'text' field.
    
    Backend options:
    - openai: Use OpenAI API (requires OPENAI_API_KEY)
    - vllm: Use local vLLM server (requires --vllm-host, --vllm-port, --vllm-model)
    
    Template options:
    - zero_shot: Direct classification without examples (fastest)
    - few_shot: Classification with examples (balanced, default)
    - few_shot_cot: Classification with examples and reasoning (most accurate)
    - cot_rag: Classification with retrieved similar examples and reasoning (most context-aware)
    
    Categories:
    - valid: Genuine reviews about the business (positive or negative experiences)
    - advertisement: Comments promoting external businesses/products/services
    - irrelevant: Comments unrelated to the business being reviewed
    - rants without visit: Comments from users who haven't actually visited the business
    """
    
    try:
        # Load environment variables
        load_environment()
        
        # Validate vLLM parameters if using vLLM backend
        if backend == 'vllm':
            if not all([vllm_host, vllm_port, vllm_model]):
                raise ValueError(
                    "For vLLM backend, you must provide --vllm-host, --vllm-port, and --vllm-model"
                )
        
        # Validate RAG parameters if using cot_rag template
        if template == 'cot_rag':
            if not rag_index_file:
                raise ValueError(
                    "For cot_rag template, you must provide --rag-index-file with the path to the FAISS index"
                )
            if not os.path.exists(rag_index_file):
                raise ValueError(f"RAG index file not found: {rag_index_file}")
        
        # Initialize LLM backend
        llm_backend = get_llm_backend(backend, vllm_host, vllm_port, vllm_model)
        
        # Determine processing mode
        if csv_file:
            # Batch processing mode
            click.echo(f"🔄 Batch processing mode")
            click.echo(f"📂 Input file: {csv_file}")
            click.echo(f"📋 Using columns: store='{store_column}', comment='{comment_column}'")
            click.echo(f"🚀 Workers: {workers} {'(single-threaded)' if workers == 1 else '(multi-threaded)'}")
            click.echo(f"📝 Template: {template} ({get_template_description(template)})")
            click.echo(f"🔧 Backend: {backend.upper()}")
            if backend == 'vllm':
                click.echo(f"🌐 vLLM Server: {vllm_host}:{vllm_port}")
                click.echo(f"🤖 Model: {vllm_model}")
            else:
                click.echo(f"🤖 Using model: {llm_backend.model}")
            if output_file:
                click.echo(f"📁 Output file: {output_file}")
            click.echo("")
            
            # Process the file (CSV or JSON)
            result_file = process_file_batch(llm_backend, csv_file, output_file, store_column, 
                                           comment_column, workers, template, rag_index_file, rag_k)
            
        else:
            # Single review processing mode
            if not store_info:
                store_info = click.prompt('Store information')
            if not user_comment:
                user_comment = click.prompt('User comment')
            
            if verbose:
                click.echo(f"\n📍 Store Info: {store_info}")
                click.echo(f"💬 User Comment: {user_comment}")
                click.echo(f"📝 Template: {template} ({get_template_description(template)})")
                click.echo(f"🔧 Backend: {backend.upper()}")
                if backend == 'vllm':
                    click.echo(f"🌐 vLLM Server: {vllm_host}:{vllm_port}")
                    click.echo(f"🤖 Model: {vllm_model}")
                else:
                    click.echo(f"🤖 Using model: {llm_backend.model}")
                click.echo("\n" + "="*50)
            
            # Initialize RAG retriever if needed for single review
            rag_retriever = None
            if template == 'cot_rag':
                click.echo(f"🔍 Loading RAG index from: {rag_index_file}")
                rag_retriever = RAGRetriever(
                    rag_index_file,
                    openai_client=llm_backend.client if hasattr(llm_backend, 'client') else None
                )

            # Classify the review
            with click.progressbar(length=1, label='Classifying review') as bar:
                classification = classify_review(llm_backend, store_info, user_comment, template, 
                                               rag_retriever=rag_retriever, rag_k=rag_k)
                bar.update(1)
            
            # Display results
            click.echo(f"\n🎯 Classification Result: {click.style(classification, fg='green', bold=True)}")
            
            if verbose:
                click.echo(f"\n📝 Full prompt sent to {backend.upper()}:")
                click.echo("-" * 50)
                prompt = create_classification_prompt(store_info, user_comment, template)
                click.echo(prompt)
        
    except ValueError as e:
        click.echo(f"❌ Configuration Error: {e}", err=True)
        click.echo("\n💡 Setup Instructions:")
        if backend == 'openai':
            click.echo("For OpenAI backend:")
            click.echo("1. Get your API key from https://platform.openai.com/api-keys")
            click.echo("2. Set environment variable: export OPENAI_API_KEY='your_key_here'")
            click.echo("3. Or create a .env file with: OPENAI_API_KEY=your_key_here")
        elif backend == 'vllm':
            click.echo("For vLLM backend:")
            click.echo("1. Start your vLLM server: python -m vllm.entrypoints.openai.api_server --model your_model --host localhost --port 8000")
            click.echo("2. Provide required parameters: --vllm-host localhost --vllm-port 8000 --vllm-model your_model")
        else:
            click.echo("1. Choose a backend with --backend openai or --backend vllm")
            click.echo("2. For OpenAI: Set OPENAI_API_KEY environment variable")
            click.echo("3. For vLLM: Provide --vllm-host, --vllm-port, and --vllm-model")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


if __name__ == '__main__':
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()
