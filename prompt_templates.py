#!/usr/bin/env python3
"""
Prompt Templates for Review Classification
"""

import re
import json


def get_zero_shot_template():
    return """You are a strict review classification assistant. 
Your task is to read the given business/store information and a user's comment, then decide which category best describes the comment. 

Categories (choose exactly one):
- valid: Genuine reviews about the business (positive or negative experiences)
- advertisement: Comments promoting external businesses/products/services
- irrelevant: Comments unrelated to the business being reviewed
- rants without visit: Comments from users who haven't actually visited the business

Business information: [STORE_INFO_PLACEHOLDER]  
User comment: [USER_COMMENT_PLACEHOLDER]  

Classification: """


def get_few_shot_template():
    return """You are a strict review classification assistant. 
Your task is to read the given business/store information and a user's comment, then decide which category best describes the comment. 
Return ONLY the category name, nothing else.

Categories (choose exactly one):
- valid: Genuine reviews about the business (positive or negative experiences)
- advertisement: Comments promoting external businesses/products/services
- irrelevant: Comments unrelated to the business being reviewed
- rants without visit: Comments from users who haven't actually visited the business

Examples:

Business: "Joe's Pizza, a local Italian restaurant in New York."
User comment: "Best pizza ever! Highly recommend the pepperoni slice."
Output: valid

Business: "Sunshine Cafe, located in Los Angeles."
User comment: "Check out my website www.bestcoffeemachines.com for amazing coffee deals!"
Output: advertisement

Business: "Tokyo Sushi Bar, serving fresh sashimi in San Francisco."
User comment: "I love my new iPhone. The camera quality is fantastic."
Output: irrelevant

Business: "Paris Boulangerie, a French bakery in Boston."
User comment: "Never been here, but I heard the staff is very rude."
Output: rants without visit

Business: "Mountain View Diner, serving American comfort food since 1985."
User comment: "Terrible service last night. The waitress was incredibly rude and the food was cold."
Output: valid

Business: "Tech Solutions Inc, providing IT support services."
User comment: "Visit my online store at buybestgadgets.net for the latest electronics!"
Output: advertisement

Business: "Green Garden Restaurant, specializing in vegetarian cuisine."
User comment: "Just watched the latest Marvel movie. The special effects were amazing!"
Output: irrelevant

Business: "Downtown Auto Repair, fixing cars for over 20 years."
User comment: "I've never been to this place, but my friend said they overcharge customers."
Output: rants without visit

Now classify the following:

Business information: [STORE_INFO_PLACEHOLDER]  
User comment: [USER_COMMENT_PLACEHOLDER]  
Output:"""


def get_few_shot_cot_template():
    return """You are a strict review classification assistant. 
Your task is to read the given business/store information and a user's comment, then decide which category best describes the comment.

Categories (choose exactly one):
- valid: Genuine reviews about the business (positive or negative experiences)
- advertisement: Comments promoting external businesses/products/services
- irrelevant: Comments unrelated to the business being reviewed
- rants without visit: Comments from users who haven't actually visited the business

For each classification, think through your reasoning step by step, then provide your final answer in the format:
Reasoning: [your step-by-step analysis]
Classification: [category name]

Examples:

Business: "Joe's Pizza, a local Italian restaurant in New York."
User comment: "Best pizza ever! Highly recommend the pepperoni slice."
Reasoning: The comment is directly about the business (Joe's Pizza), mentions specific food items (pepperoni slice), and expresses a positive experience. This is clearly a genuine review of the restaurant.
Classification: valid

Business: "Sunshine Cafe, located in Los Angeles."
User comment: "Check out my website www.bestcoffeemachines.com for amazing coffee deals!"
Reasoning: The comment is not about Sunshine Cafe at all. Instead, it's promoting an external website selling coffee machines. This is clearly promotional content trying to advertise another business.
Classification: advertisement

Business: "Tokyo Sushi Bar, serving fresh sashimi in San Francisco."
User comment: "I love my new iPhone. The camera quality is fantastic."
Reasoning: The comment is about an iPhone and its camera, which has nothing to do with Tokyo Sushi Bar or sushi/food in general. This comment is completely unrelated to the business being reviewed.
Classification: irrelevant

Business: "Paris Boulangerie, a French bakery in Boston."
User comment: "Never been here, but I heard the staff is very rude."
Reasoning: The user explicitly states they have "never been here," meaning they haven't actually visited the business. They're sharing second-hand information without personal experience. This is a rant without actual visitation.
Classification: rants without visit

Business: "Mountain View Diner, serving American comfort food since 1985."
User comment: "Terrible service last night. The waitress was incredibly rude and the food was cold."
Reasoning: The comment describes a specific experience ("last night") with details about service and food quality. Even though it's negative, it's a genuine review based on an actual visit to the diner.
Classification: valid

Now classify the following:

Business information: [STORE_INFO_PLACEHOLDER]  
User comment: [USER_COMMENT_PLACEHOLDER]  

Please provide your reasoning and classification:"""


def get_cot_rag_template():
    return """You are a strict review classification assistant. 
Your task is to read the given business/store information and a user's comment, then decide which category best describes the comment.

Categories (choose exactly one):
- valid: Genuine reviews about the business (positive or negative experiences)
- advertisement: Comments promoting external businesses/products/services
- irrelevant: Comments unrelated to the business being reviewed
- rants without visit: Comments from users who haven't actually visited the business

I will provide you with similar examples from our training data to help guide your classification. These examples were retrieved based on semantic similarity to the current comment.

Similar Examples from Training Data:
[RETRIEVED_EXAMPLES_PLACEHOLDER]

Based on these similar examples and your understanding of the categories, analyze the following comment step by step, then provide your final answer in the format:
Reasoning: [your step-by-step analysis, referencing similar patterns from the examples above]
Classification: [category name]

Now classify the following:

Business information: [STORE_INFO_PLACEHOLDER]  
User comment: [USER_COMMENT_PLACEHOLDER]  

Please provide your reasoning and classification:"""


def parse_classification_response(response_text):
    """
    Parse the classification response to extract the category.
    
    Args:
        response_text (str): The full response from the model
        
    Returns:
        str: The extracted classification category
    """
    # Clean up the response
    response_text = response_text.strip()
    
    # Valid categories
    valid_categories = ['valid', 'advertisement', 'irrelevant', 'rants without visit']
    
    # Look for "Classification: [category]" pattern
    classification_match = re.search(r'Classification:\s*([a-zA-Z\s]+)', response_text, re.IGNORECASE)
    if classification_match:
        classification = classification_match.group(1).strip().lower()
        if classification in valid_categories:
            return classification
    
    # Look for any of the valid categories at the end of the response
    response_lower = response_text.lower()
    
    # Check for category names appearing at the end of the response
    for category in valid_categories:
        if response_lower.strip().endswith(category):
            return category
    
    # Check for category names appearing in the last line
    last_line = response_text.strip().split('\n')[-1].lower()
    for category in valid_categories:
        if category in last_line:
            return category
    
    # Check for category names appearing anywhere in the response
    for category in valid_categories:
        if category in response_lower:
            return category
    
    # If we can't parse it, return the raw response (will be handled as an error)
    return response_text.strip()


def parse_cot_response(response_text):
    """
    Parse the Chain-of-Thought response to extract the classification.
    
    Args:
        response_text (str): The full response from the model
        
    Returns:
        str: The extracted classification category
    """
    # Clean up the response
    response_text = response_text.strip()
    
    # Look for "Classification: [category]" pattern
    classification_match = re.search(r'Classification:\s*([a-zA-Z\s]+)', response_text, re.IGNORECASE)
    if classification_match:
        classification = classification_match.group(1).strip().lower()
        # Validate that it's one of our expected categories
        valid_categories = ['valid', 'advertisement', 'irrelevant', 'rants without visit']
        if classification in valid_categories:
            return classification
    
    # Fallback: Look for any of the valid categories in the response
    response_lower = response_text.lower()
    valid_categories = ['valid', 'advertisement', 'irrelevant', 'rants without visit']
    
    # Check for exact matches first
    for category in valid_categories:
        if f"classification: {category}" in response_lower:
            return category
    
    # Check for category names appearing at the end of the response
    for category in valid_categories:
        if response_lower.strip().endswith(category):
            return category
    
    # Check for category names appearing anywhere in the last line
    last_line = response_text.strip().split('\n')[-1].lower()
    for category in valid_categories:
        if category in last_line:
            return category
    
    # If we can't parse it, return the raw response (will be handled as an error)
    return response_text.strip()


def create_classification_prompt(store_info, user_comment, template_type='few_shot', retrieved_examples=None):
    """
    Create the classification prompt with placeholders replaced.
    
    Args:
        store_info (str): Information about the store/business
        user_comment (str): The user's comment to classify
        template_type (str): Type of template to use ('zero_shot', 'few_shot', 'few_shot_cot', 'cot_rag')
        retrieved_examples (str, optional): Formatted retrieved examples for RAG template
        
    Returns:
        str: The complete prompt ready to send to the model
    """
    # Get the appropriate template
    if template_type == 'zero_shot':
        template = get_zero_shot_template()
    elif template_type == 'few_shot':
        template = get_few_shot_template()
    elif template_type == 'few_shot_cot':
        template = get_few_shot_cot_template()
    elif template_type == 'cot_rag':
        template = get_cot_rag_template()
        if retrieved_examples is None:
            raise ValueError("retrieved_examples is required for cot_rag template type")
    else:
        raise ValueError(f"Unknown template type: {template_type}. Valid options: zero_shot, few_shot, few_shot_cot, cot_rag")
    
    # Replace placeholders with actual content
    final_prompt = template.replace('[STORE_INFO_PLACEHOLDER]', store_info)
    final_prompt = final_prompt.replace('[USER_COMMENT_PLACEHOLDER]', user_comment)
    
    # Replace retrieved examples placeholder for RAG template
    if template_type == 'cot_rag':
        final_prompt = final_prompt.replace('[RETRIEVED_EXAMPLES_PLACEHOLDER]', retrieved_examples)
    
    return final_prompt


# Template metadata for CLI help
TEMPLATE_INFO = {
    'zero_shot': {
        'name': 'Zero-shot',
        'description': 'Direct classification without examples - fastest and most cost-effective'
    },
    'few_shot': {
        'name': 'Few-shot', 
        'description': 'Classification with examples - balanced accuracy and cost'
    },
    'few_shot_cot': {
        'name': 'Few-shot + Chain-of-Thought',
        'description': 'Classification with examples and reasoning - highest accuracy but slower'
    },
    'cot_rag': {
        'name': 'COT RAG (Chain-of-Thought + Retrieval Augmented Generation)',
        'description': 'Classification with retrieved similar examples and reasoning - most context-aware'
    }
}


def get_available_templates():
    """
    Get list of available template types.
    
    Returns:
        list: List of available template type strings
    """
    return list(TEMPLATE_INFO.keys())


def get_template_description(template_type):
    """
    Get description for a template type.
    
    Args:
        template_type (str): The template type
        
    Returns:
        str: Description of the template
    """
    return TEMPLATE_INFO.get(template_type, {}).get('description', 'Unknown template')
