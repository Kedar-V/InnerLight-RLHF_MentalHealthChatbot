"""
Model configuration and setup utilities
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def get_quantization_config():
    """
    Get BitsAndBytes quantization configuration for memory efficiency.
    
    Returns:
        BitsAndBytesConfig: Quantization config for 4-bit training
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )


def setup_classifier_model(model_name="bert-base-uncased", num_labels=3, id2label=None, label2id=None):
    """
    Load and configure BERT classifier with LoRA.
    
    Args:
        model_name (str): Model identifier from HuggingFace
        num_labels (int): Number of classification labels
        id2label (dict): Mapping from label IDs to strings
        label2id (dict): Mapping from label strings to IDs
        
    Returns:
        tuple: (model, tokenizer)
    """
    bnb_config = get_quantization_config()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def setup_generator_model(model_name="Qwen/Qwen1.5-1.8B"):
    """
    Load and configure causal LM with LoRA for generation.
    
    Args:
        model_name (str): Model identifier from HuggingFace
        
    Returns:
        tuple: (model, tokenizer)
    """
    bnb_config = get_quantization_config()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def print_model_info(model, name="Model"):
    """
    Print model parameter statistics.
    
    Args:
        model: PyTorch model
        name (str): Model name for display
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{name} Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Trainable %: {trainable_params/total_params*100:.2f}%")
