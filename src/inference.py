"""
Inference utilities for the InnerLight system
"""

import torch
from transformers import pipeline
from peft import PeftModel


CRISIS_RESOURCES = {
    "US": {
        "hotline": "988 Suicide & Crisis Lifeline",
        "number": "988",
        "text": "Text HOME to 741741",
        "website": "https://988lifeline.org"
    },
    "United Kingdom": {
        "hotline": "Samaritans",
        "number": "116 123",
        "email": "jo@samaritans.org",
        "website": "https://www.samaritans.org",
    },
    "Canada": {
        "hotline": "Crisis Services Canada",
        "number": "1-833-456-4566",
        "text": "Text 45645",
        "website": "https://www.crisisservicescanada.ca",
    },
    "Australia": {
        "hotline": "Lifeline",
        "number": "13 11 14",
        "website": "https://www.lifeline.org.au",
    },
    "International": {
        "resource": "Find a Helpline",
        "website": "https://findahelpline.com",
    }
}


def load_classifier_for_inference(base_model_name, lora_path, tokenizer):
    """
    Load trained classifier with LoRA weights for inference.
    
    Args:
        base_model_name (str): Base model identifier
        lora_path (str): Path to saved LoRA weights
        tokenizer: Tokenizer
        
    Returns:
        tuple: (model, pipeline)
    """
    from transformers import AutoModelForSequenceClassification
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3,
        id2label={0: 'safe', 1: 'concerning', 2: 'high-risk'},
        label2id={'safe': 0, 'concerning': 1, 'high-risk': 2}
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    return model, pipe


def load_generator_for_inference(base_model_name, lora_path, tokenizer):
    """
    Load trained generator with LoRA weights for inference.
    
    Args:
        base_model_name (str): Base model identifier
        lora_path (str): Path to saved LoRA weights
        tokenizer: Tokenizer
        
    Returns:
        tuple: (model, pipeline)
    """
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        device=0 if torch.cuda.is_available() else -1
    )
    
    return model, pipe


def get_crisis_resources(location="US"):
    """
    Retrieve crisis resources for a specific location.
    
    Args:
        location (str): Geographic location
        
    Returns:
        str: Formatted crisis resources message
    """
    resources = CRISIS_RESOURCES.get(location, CRISIS_RESOURCES["International"])
    
    message = "\n\n📞 Crisis Resources:\n"
    if "hotline" in resources:
        message += f"- {resources['hotline']}: {resources['number']}\n"
    if "text" in resources:
        message += f"- Crisis Text Line: {resources['text']}\n"
    if "website" in resources:
        message += f"- Website: {resources['website']}\n"
    if "email" in resources:
        message += f"- Email: {resources['email']}\n"
    if "resource" in resources:
        message += f"- {resources['resource']}\n"
        
    return message


def process_user_input(user_input, classifier_pipe, generator_pipe, location="US"):
    """
    Process user input through complete InnerLight system.
    
    Args:
        user_input (str): User message
        classifier_pipe: Classification pipeline
        generator_pipe: Generation pipeline
        location (str): User's geographic location
        
    Returns:
        dict: Risk level, confidence, and response
    """
    # Classify risk
    classification = classifier_pipe(user_input)[0]
    risk_level = classification['label']
    confidence = classification['score']
    
    # Get numeric label from string
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
            "Provide empathetic support and gently suggest seeking help if needed."
        )
    else:  # high-risk
        instruction = (
            "You are a crisis counselor. The user may be in immediate danger. "
            "Respond with urgent care and empathy. Strongly encourage contacting "
            "crisis services immediately."
        )
    
    # Format prompt for generator
    prompt = f"{instruction}\n\nUser: {user_input}\n\nCounselor:"
    
    # Generate response
    generated = generator_pipe(prompt)[0]['generated_text']
    response = generated.split("Counselor:")[-1].strip()
    
    # Include crisis resources if needed
    crisis_resources = None
    if risk_label in ["concerning", "high-risk"]:
        crisis_resources = get_crisis_resources(location)
    
    return {
        "risk_level": risk_label,
        "confidence": confidence,
        "response": response,
        "crisis_resources": crisis_resources
    }


def format_inference_output(result):
    """
    Format inference result for display.
    
    Args:
        result (dict): Result from process_user_input
        
    Returns:
        str: Formatted output
    """
    output = f"\n🔍 Risk Assessment: {result['risk_level'].upper()} (confidence: {result['confidence']:.2%})\n"
    output += f"💬 Response:\n{result['response']}"
    
    if result['crisis_resources']:
        output += result['crisis_resources']
    
    return output
