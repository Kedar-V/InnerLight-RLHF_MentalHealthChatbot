"""
RAG (Retrieval-Augmented Generation) utilities for crisis resource retrieval
"""

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


# Crisis resources knowledge base
CRISIS_RESOURCES_DATA = [
    {
        "country": "United States",
        "hotline": "988 Suicide & Crisis Lifeline",
        "number": "988",
        "text_service": "Text HOME to 741741 (Crisis Text Line)",
        "website": "https://988lifeline.org",
        "description": "24/7 free and confidential support for people in distress"
    },
    {
        "country": "United Kingdom",
        "hotline": "Samaritans",
        "number": "116 123",
        "email": "jo@samaritans.org",
        "website": "https://www.samaritans.org",
        "description": "24/7 emotional support for anyone struggling to cope"
    },
    {
        "country": "Canada",
        "hotline": "Crisis Services Canada",
        "number": "1-833-456-4566",
        "text_service": "Text 45645",
        "website": "https://www.crisisservicescanada.ca",
        "description": "24/7 support in English and French"
    },
    {
        "country": "Australia",
        "hotline": "Lifeline",
        "number": "13 11 14",
        "website": "https://www.lifeline.org.au",
        "description": "24/7 crisis support and suicide prevention"
    },
    {
        "country": "International",
        "resource": "Find a Helpline",
        "website": "https://findahelpline.com",
        "description": "Directory of crisis centers worldwide"
    }
]


def format_resources_to_text(resources_data):
    """
    Convert resource data to text format for embedding.
    
    Args:
        resources_data (list): List of resource dictionaries
        
    Returns:
        list: Text representations of resources
    """
    resource_texts = []
    for resource in resources_data:
        text = f"{resource['country']}: {resource.get('hotline', resource.get('resource', ''))}. "
        if 'number' in resource:
            text += f"Call {resource['number']}. "
        if 'text_service' in resource:
            text += f"{resource['text_service']}. "
        text += f"{resource['description']}. Visit {resource['website']}"
        resource_texts.append(text)
    
    return resource_texts


def create_rag_index(resource_texts, model_name='all-MiniLM-L6-v2'):
    """
    Create FAISS index for crisis resources.
    
    Args:
        resource_texts (list): List of resource text documents
        model_name (str): SentenceTransformer model to use
        
    Returns:
        tuple: (FAISS index, embedding model)
    """
    embedding_model = SentenceTransformer(model_name)
    
    # Create embeddings
    embeddings = embedding_model.encode(resource_texts, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, embedding_model


def retrieve_crisis_resources(query, index, resource_texts, embedding_model, top_k=2, resources_data=None):
    """
    Retrieve relevant crisis resources based on query.
    
    Args:
        query (str): Search query
        index: FAISS index
        resource_texts (list): List of resource texts
        embedding_model: SentenceTransformer model
        top_k (int): Number of results to return
        resources_data (list): Original resource data dictionaries
        
    Returns:
        list: Retrieved resources with metadata
    """
    # Encode query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Get results
    results = []
    for idx in indices[0]:
        result = {
            "text": resource_texts[idx],
            "data": resources_data[idx] if resources_data else None
        }
        results.append(result)
    
    return results


def save_rag_artifacts(index, resource_texts, resources_data, output_dir):
    """
    Save FAISS index and resource data to disk.
    
    Args:
        index: FAISS index
        resource_texts (list): List of resource texts
        resources_data (list): Resource data dictionaries
        output_dir (str): Output directory path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, "crisis_resources.index"))
    
    # Save resource data
    with open(os.path.join(output_dir, "crisis_resources_data.pkl"), 'wb') as f:
        pickle.dump({
            'texts': resource_texts,
            'data': resources_data
        }, f)
    
    print(f"RAG artifacts saved to {output_dir}")


def load_rag_artifacts(artifact_dir, model_name='all-MiniLM-L6-v2'):
    """
    Load FAISS index and resource data from disk.
    
    Args:
        artifact_dir (str): Directory containing saved artifacts
        model_name (str): SentenceTransformer model to use
        
    Returns:
        tuple: (index, embedding_model, resource_texts, resources_data)
    """
    import os
    
    # Load FAISS index
    index = faiss.read_index(os.path.join(artifact_dir, "crisis_resources.index"))
    
    # Load resource data
    with open(os.path.join(artifact_dir, "crisis_resources_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    resource_texts = data['texts']
    resources_data = data['data']
    
    # Load embedding model
    embedding_model = SentenceTransformer(model_name)
    
    return index, embedding_model, resource_texts, resources_data


def process_with_rag(user_input, classifier_pipe, generator_pipe, index, 
                     embedding_model, resource_texts, resources_data, user_location="United States"):
    """
    Process user input with RAG-enhanced resource retrieval.
    
    Args:
        user_input (str): User message
        classifier_pipe: Classification pipeline
        generator_pipe: Generation pipeline
        index: FAISS index
        embedding_model: SentenceTransformer model
        resource_texts (list): List of resource texts
        resources_data (list): Resource data dictionaries
        user_location (str): User's location for relevant resources
        
    Returns:
        dict: Complete response with resources
    """
    # Classify risk
    classification = classifier_pipe(user_input)[0]
    risk_level = classification['label']
    confidence = classification['score']
    
    label_to_risk = {
        'LABEL_0': 'safe',
        'LABEL_1': 'concerning',
        'LABEL_2': 'high-risk'
    }
    risk_label = label_to_risk.get(risk_level, 'safe')
    
    # Set instruction based on risk level
    if risk_label == "safe":
        instruction = "You are a supportive assistant. Provide helpful guidance."
    elif risk_label == "concerning":
        instruction = (
            "You are a compassionate counselor. The user seems to be struggling. "
            "Provide empathetic support."
        )
    else:
        instruction = (
            "You are a crisis counselor. Respond with urgent care and empathy. "
            "Strongly encourage immediate help."
        )
    
    # Generate response
    prompt = f"{instruction}\n\nUser: {user_input}\n\nCounselor:"
    generated = generator_pipe(prompt)[0]['generated_text']
    response = generated.split("Counselor:")[-1].strip()
    
    # Retrieve resources if needed
    retrieved_resources = None
    if risk_label in ["concerning", "high-risk"]:
        query = f"crisis support {user_location}"
        retrieved_resources = retrieve_crisis_resources(
            query, index, resource_texts, embedding_model, 
            top_k=2, resources_data=resources_data
        )
    
    return {
        "risk_level": risk_label,
        "confidence": confidence,
        "response": response,
        "retrieved_resources": retrieved_resources
    }


def format_rag_output(result):
    """
    Format RAG result for display.
    
    Args:
        result (dict): Result from process_with_rag
        
    Returns:
        str: Formatted output
    """
    output = f"\n🔍 Risk Assessment: {result['risk_level'].upper()} (confidence: {result['confidence']:.2%})\n"
    output += f"💬 Response:\n{result['response']}"
    
    if result['retrieved_resources']:
        output += "\n\n📞 Recommended Resources:"
        for i, resource in enumerate(result['retrieved_resources'], 1):
            data = resource['data']
            output += f"\n\n{i}. {data['country']}"
            if 'hotline' in data:
                output += f" - {data['hotline']}"
            if 'number' in data:
                output += f"\n   Phone: {data['number']}"
            if 'text_service' in data:
                output += f"\n   Text: {data['text_service']}"
            if 'website' in data:
                output += f"\n   Website: {data['website']}"
    
    return output
