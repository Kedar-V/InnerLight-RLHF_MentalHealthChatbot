"""
Training utilities for classifier and generator models
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


def compute_classification_metrics(eval_pred):
    """
    Compute classification metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
        
    Returns:
        dict: Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def get_classifier_training_args(output_dir, num_epochs=3, batch_size=8):
    """
    Create training arguments for classifier.
    
    Args:
        output_dir (str): Output directory for model checkpoints
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        
    Returns:
        TrainingArguments: Configuration for trainer
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to=[],
        disable_tqdm=False,
        logging_first_step=True
    )


def get_generator_training_args(output_dir, num_epochs=2, max_steps=1500):
    """
    Create training arguments for generator.
    
    Args:
        output_dir (str): Output directory for model checkpoints
        num_epochs (int): Number of training epochs
        max_steps (int): Maximum training steps
        
    Returns:
        TrainingArguments: Configuration for trainer
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=300,
        max_steps=max_steps,
        fp16=True,
        report_to="none"
    )


def create_classifier_trainer(model, args, train_dataset, eval_dataset):
    """
    Create Trainer for classification.
    
    Args:
        model: Classification model
        args: TrainingArguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        
    Returns:
        Trainer: Configured trainer
    """
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_classification_metrics
    )


def create_generator_trainer(model, args, train_dataset, tokenizer):
    """
    Create Trainer for generation.
    
    Args:
        model: Generation model
        args: TrainingArguments
        train_dataset: Training dataset
        tokenizer: Tokenizer for data collation
        
    Returns:
        Trainer: Configured trainer
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )


def tokenize_classifier_data(examples, tokenizer, max_length=512):
    """
    Tokenize data for classification.
    
    Args:
        examples (dict): Batch of examples with 'text' key
        tokenizer: Tokenizer to use
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Tokenized batch
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )


def tokenize_generation_data(batch, tokenizer, max_length=1024):
    """
    Tokenize data for generation.
    
    Args:
        batch (dict): Batch of examples with 'text' key
        tokenizer: Tokenizer to use
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Tokenized batch
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )


def build_generation_prompt(example, instruction_text=None):
    """
    Format example as prompt for generation training.
    
    Args:
        example (dict): Example with 'instruction', 'input', 'output' keys
        instruction_text (str): Override instruction if provided
        
    Returns:
        dict: Formatted prompt
    """
    instruction = instruction_text or example.get('instruction', '')
    prompt = (
        f"{instruction}\n\n"
        f"User: {example['input']}\n\n"
        f"Counselor: {example['output']}"
    )
    return {"text": prompt}
