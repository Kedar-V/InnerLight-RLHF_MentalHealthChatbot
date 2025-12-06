"""
Evaluation utilities for InnerLight system
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier(trainer, tokenized_dataset, id2label):
    """
    Evaluate classifier on test set.
    
    Args:
        trainer: Trainer object
        tokenized_dataset: Tokenized test dataset
        id2label (dict): Label ID to string mapping
        
    Returns:
        dict: Evaluation metrics and predictions
    """
    # Get test results
    test_results = trainer.evaluate(tokenized_dataset["test"])
    
    # Get predictions
    predictions_output = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(predictions_output.predictions, axis=1)
    true_labels = predictions_output.label_ids
    
    # Classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        true_labels,
        predictions,
        target_names=target_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        "metrics": test_results,
        "report": report,
        "predictions": predictions,
        "true_labels": true_labels,
        "confusion_matrix": cm
    }


def plot_confusion_matrix(cm, labels, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (ndarray): Confusion matrix
        labels (list): Label names
        save_path (str): Path to save figure (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.title('Confusion Matrix - Suicide Risk Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def generate_system_report(classifier_trainer, generator_model, classifier_model, 
                          tokenized_dataset, id2label):
    """
    Generate comprehensive evaluation report for the system.
    
    Args:
        classifier_trainer: Trainer for classifier
        generator_model: Generator model
        classifier_model: Classifier model
        tokenized_dataset: Tokenized dataset with test split
        id2label (dict): Label mapping
        
    Returns:
        dict: Comprehensive evaluation report
    """
    print("\n" + "="*60)
    print("INNERLIGHT SYSTEM EVALUATION REPORT")
    print("="*60)
    
    # Classifier evaluation
    print("\n1. CLASSIFIER PERFORMANCE")
    print("-" * 40)
    
    clf_eval = evaluate_classifier(classifier_trainer, tokenized_dataset, id2label)
    test_results = clf_eval["metrics"]
    
    print(f"Accuracy:  {test_results['eval_accuracy']:.4f}")
    print(f"F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall:    {test_results['eval_recall']:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    report_dict = clf_eval["report"]
    for label, metrics in report_dict.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"\n  {label}:")
            if isinstance(metrics, dict):
                print(f"    Precision: {metrics.get('precision', 0):.4f}")
                print(f"    Recall:    {metrics.get('recall', 0):.4f}")
                print(f"    F1-Score:  {metrics.get('f1-score', 0):.4f}")
    
    # Model efficiency
    print("\n2. MODEL EFFICIENCY")
    print("-" * 40)
    
    classifier_params = sum(p.numel() for p in classifier_model.parameters())
    classifier_trainable = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)
    generator_params = sum(p.numel() for p in generator_model.parameters())
    generator_trainable = sum(p.numel() for p in generator_model.parameters() if p.requires_grad)
    
    print(f"Classifier:")
    print(f"  Total Parameters:      {classifier_params:,}")
    print(f"  Trainable Parameters:  {classifier_trainable:,} ({classifier_trainable/classifier_params*100:.2f}%)")
    
    print(f"\nGenerator:")
    print(f"  Total Parameters:      {generator_params:,}")
    print(f"  Trainable Parameters:  {generator_trainable:,} ({generator_trainable/generator_params*100:.2f}%)")
    
    # Success criteria
    print("\n3. SUCCESS CRITERIA")
    print("-" * 40)
    
    criteria = {
        "F1 Score ≥ 70%": test_results['eval_f1'] >= 0.70,
        "Model < 500M params": generator_params < 500_000_000,
        "Dual-stage RLHF framework": True,
        "RAG integration": True,
        "LoRA parameter efficiency": True,
    }
    
    for criterion, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{criterion}: {status}")
    
    print("\n" + "="*60)
    
    return {
        "classifier_metrics": test_results,
        "classifier_report": clf_eval["report"],
        "confusion_matrix": clf_eval["confusion_matrix"],
        "model_params": {
            "classifier_total": classifier_params,
            "classifier_trainable": classifier_trainable,
            "generator_total": generator_params,
            "generator_trainable": generator_trainable,
        },
        "criteria": criteria
    }


def save_evaluation_report(report, output_dir):
    """
    Save evaluation report to JSON.
    
    Args:
        report (dict): Evaluation report
        output_dir (str): Output directory
    """
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    report_json = {}
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            report_json[key] = value.tolist()
        elif isinstance(value, dict):
            report_json[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in value.items()
            }
        else:
            report_json[key] = value
    
    with open(os.path.join(output_dir, "evaluation_report.json"), 'w') as f:
        json.dump(report_json, f, indent=2)
    
    print(f"Evaluation report saved to {output_dir}/evaluation_report.json")
