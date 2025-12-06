"""
Main training script for InnerLight complete pipeline
"""

import os
import torch
import random
import numpy as np
import kagglehub

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from src import data, models, training, evaluation, rag


def setup_paths(base_dir):
    """Create necessary directory structure."""
    paths = {
        'classifier_output': os.path.join(base_dir, 'suicide_risk_classifier'),
        'generator_output': os.path.join(base_dir, 'safety_response_generator'),
        'rag_output': os.path.join(base_dir, 'rag_artifacts'),
        'evaluation_output': os.path.join(base_dir, 'evaluation'),
        'data_cache': os.path.join(base_dir, 'data_cache')
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def download_and_prepare_data(kaggle_dataset):
    """Download and prepare classification data."""
    print("\n" + "="*60)
    print("STEP 1: DATA DOWNLOAD AND PREPARATION")
    print("="*60)
    
    path = kagglehub.dataset_download(kaggle_dataset)
    dataset_path = os.path.join(path, 'dataset.csv')
    
    print(f"Dataset downloaded to: {path}")
    
    # Load and label
    df_conversations = data.load_classification_data(dataset_path)
    print(f"Loaded {len(df_conversations)} conversation samples")
    print(f"\nRisk Label Distribution:")
    print(df_conversations['risk_label'].value_counts())
    
    # Balance dataset
    df_balanced = data.balance_classification_dataset(df_conversations)
    print(f"\nBalanced dataset size: {len(df_balanced)}")
    
    # Prepare classification dataset
    dataset_classifier, label2id, id2label = data.prepare_classifier_dataset(df_balanced)
    print(f"\nClassification dataset ready!")
    print(f"Train: {len(dataset_classifier['train'])}, "
          f"Val: {len(dataset_classifier['validation'])}, "
          f"Test: {len(dataset_classifier['test'])}")
    
    # Prepare generation dataset
    df_generation = data.prepare_generation_dataset(df_conversations)
    print(f"\nGeneration dataset size: {len(df_generation)}")
    
    return dataset_classifier, label2id, id2label, df_generation, df_conversations


def train_classifier(dataset_classifier, label2id, id2label, output_dir):
    """Train risk classifier."""
    print("\n" + "="*60)
    print("STEP 2: TRAINING SUICIDE RISK CLASSIFIER")
    print("="*60)
    
    # Setup model
    classifier_model, classifier_tokenizer = models.setup_classifier_model(
        model_name="bert-base-uncased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )
    models.print_model_info(classifier_model, "Classifier")
    
    # Tokenize dataset
    def tokenize_fn(examples):
        return training.tokenize_classifier_data(examples, classifier_tokenizer)
    
    tokenized_classifier = dataset_classifier.map(
        tokenize_fn,
        batched=True,
        remove_columns=['text']
    )
    tokenized_classifier.set_format("torch")
    
    # Training arguments and trainer
    training_args = training.get_classifier_training_args(
        output_dir=output_dir,
        num_epochs=3,
        batch_size=8
    )
    
    trainer = training.create_classifier_trainer(
        classifier_model,
        training_args,
        tokenized_classifier["train"],
        tokenized_classifier["validation"]
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_classifier["test"])
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
    
    # Save
    classifier_model.save_pretrained(os.path.join(output_dir, "lora_weights"))
    classifier_tokenizer.save_pretrained(os.path.join(output_dir, "lora_weights"))
    print(f"\nClassifier saved to {output_dir}/lora_weights")
    
    return trainer, classifier_model, classifier_tokenizer, tokenized_classifier


def train_generator(df_generation, output_dir):
    """Train response generator."""
    print("\n" + "="*60)
    print("STEP 3: TRAINING SAFETY RESPONSE GENERATOR")
    print("="*60)
    
    # Setup model
    generator_model, generator_tokenizer = models.setup_generator_model(
        model_name="Qwen/Qwen1.5-1.8B"
    )
    models.print_model_info(generator_model, "Generator")
    
    # Load dataset
    from datasets import Dataset
    ds_generation = Dataset.from_pandas(df_generation)
    
    # Format prompts
    def format_fn(example):
        return training.build_generation_prompt(example)
    
    ds_prompted = ds_generation.map(format_fn)
    ds_prompted = ds_prompted.remove_columns(
        [c for c in ds_prompted.column_names if c != "text"]
    )
    
    # Tokenize
    def tokenize_fn(batch):
        return training.tokenize_generation_data(batch, generator_tokenizer)
    
    ds_tokenized = ds_prompted.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds_prompted.column_names
    )
    ds_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Training arguments and trainer
    training_args = training.get_generator_training_args(
        output_dir=output_dir,
        num_epochs=2,
        max_steps=1500
    )
    
    trainer = training.create_generator_trainer(
        generator_model,
        training_args,
        ds_tokenized,
        generator_tokenizer
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Save
    generator_model.save_pretrained(os.path.join(output_dir, "lora_weights"))
    generator_tokenizer.save_pretrained(os.path.join(output_dir, "lora_weights"))
    print(f"\nGenerator saved to {output_dir}/lora_weights")
    
    return trainer, generator_model, generator_tokenizer


def setup_rag_system(output_dir):
    """Setup RAG system for crisis resources."""
    print("\n" + "="*60)
    print("STEP 4: SETTING UP RAG SYSTEM")
    print("="*60)
    
    # Format resources to text
    resource_texts = rag.format_resources_to_text(rag.CRISIS_RESOURCES_DATA)
    print(f"Created {len(resource_texts)} crisis resource documents")
    
    # Create FAISS index
    index, embedding_model = rag.create_rag_index(resource_texts)
    print(f"Created FAISS index with {index.ntotal} resources")
    
    # Save artifacts
    rag.save_rag_artifacts(index, resource_texts, rag.CRISIS_RESOURCES_DATA, output_dir)
    
    return index, embedding_model, resource_texts


def evaluate_system(classifier_trainer, generator_model, classifier_model, 
                   tokenized_classifier, id2label, output_dir):
    """Evaluate complete system."""
    print("\n" + "="*60)
    print("STEP 5: SYSTEM EVALUATION")
    print("="*60)
    
    eval_report = evaluation.generate_system_report(
        classifier_trainer,
        generator_model,
        classifier_model,
        tokenized_classifier,
        id2label
    )
    
    # Plot confusion matrix
    labels = [id2label[i] for i in range(len(id2label))]
    evaluation.plot_confusion_matrix(
        eval_report["confusion_matrix"],
        labels,
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    
    # Save report
    evaluation.save_evaluation_report(eval_report, output_dir)
    
    return eval_report


def main(data_dir="/content/drive/MyDrive/NLP"):
    """Main training pipeline."""
    print("\n" + "="*70)
    print("INNERLIGHT: SUICIDE RISK DETECTION & SAFETY RESPONSE SYSTEM")
    print("="*70)
    
    # Setup paths
    paths = setup_paths(data_dir)
    
    # Step 1: Data preparation
    dataset_classifier, label2id, id2label, df_generation, df_conversations = \
        download_and_prepare_data("birdy654/human-and-llm-mental-health-conversations")
    
    # Step 2: Train classifier
    clf_trainer, clf_model, clf_tokenizer, tokenized_clf = \
        train_classifier(dataset_classifier, label2id, id2label, paths['classifier_output'])
    
    # Step 3: Train generator
    gen_trainer, gen_model, gen_tokenizer = \
        train_generator(df_generation, paths['generator_output'])
    
    # Step 4: Setup RAG
    rag_index, rag_embedding_model, rag_texts = setup_rag_system(paths['rag_output'])
    
    # Step 5: Evaluate
    eval_report = evaluate_system(
        clf_trainer, gen_model, clf_model, 
        tokenized_clf, id2label, paths['evaluation_output']
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel artifacts saved to: {data_dir}")
    print(f"- Classifier: {paths['classifier_output']}")
    print(f"- Generator: {paths['generator_output']}")
    print(f"- RAG: {paths['rag_output']}")
    print(f"- Evaluation: {paths['evaluation_output']}")
    
    return {
        'paths': paths,
        'models': {
            'classifier': clf_model,
            'generator': gen_model
        },
        'tokenizers': {
            'classifier': clf_tokenizer,
            'generator': gen_tokenizer
        },
        'rag': {
            'index': rag_index,
            'embedding_model': rag_embedding_model,
            'texts': rag_texts
        },
        'evaluation': eval_report
    }


if __name__ == "__main__":
    main()
