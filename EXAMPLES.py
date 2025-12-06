"""
Example usage patterns for InnerLight components
"""

# ==============================================================================
# EXAMPLE 1: Complete Training Pipeline
# ==============================================================================

from src import data, models, training, evaluation, rag
import torch
import random
import numpy as np

# Set seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Setup paths
output_dirs = {
    'classifier': './models/classifier',
    'generator': './models/generator',
    'rag': './models/rag',
    'evaluation': './results/evaluation'
}

# Step 1: Load and prepare data
print("Loading data...")
df_conversations = data.load_classification_data("path/to/dataset.csv")
df_balanced = data.balance_classification_dataset(df_conversations)
dataset_classifier, label2id, id2label = data.prepare_classifier_dataset(df_balanced)
df_generation = data.prepare_generation_dataset(df_conversations)

# Step 2: Setup and train classifier
print("Training classifier...")
clf_model, clf_tokenizer = models.setup_classifier_model(
    model_name="bert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

def tokenize_clf(examples):
    return training.tokenize_classifier_data(examples, clf_tokenizer)

tokenized_clf = dataset_classifier.map(tokenize_clf, batched=True, remove_columns=['text'])
tokenized_clf.set_format("torch")

clf_args = training.get_classifier_training_args(output_dirs['classifier'])
clf_trainer = training.create_classifier_trainer(
    clf_model, clf_args, tokenized_clf["train"], tokenized_clf["validation"]
)

clf_trainer.train()
test_results = clf_trainer.evaluate(tokenized_clf["test"])
print(f"Classifier F1: {test_results['eval_f1']:.4f}")

# Step 3: Setup and train generator
print("Training generator...")
gen_model, gen_tokenizer = models.setup_generator_model(model_name="Qwen/Qwen1.5-1.8B")

from datasets import Dataset
ds_gen = Dataset.from_pandas(df_generation)
ds_prompted = ds_gen.map(training.build_generation_prompt)
ds_prompted = ds_prompted.remove_columns([c for c in ds_prompted.column_names if c != "text"])

def tokenize_gen(batch):
    return training.tokenize_generation_data(batch, gen_tokenizer)

ds_tokenized = ds_prompted.map(tokenize_gen, batched=True, remove_columns=ds_prompted.column_names)
ds_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

gen_args = training.get_generator_training_args(output_dirs['generator'])
gen_trainer = training.create_generator_trainer(gen_model, gen_args, ds_tokenized, gen_tokenizer)

gen_trainer.train()
print("Generator training complete!")

# Step 4: Setup RAG
print("Setting up RAG...")
resource_texts = rag.format_resources_to_text(rag.CRISIS_RESOURCES_DATA)
rag_index, rag_model = rag.create_rag_index(resource_texts)
rag.save_rag_artifacts(rag_index, resource_texts, rag.CRISIS_RESOURCES_DATA, output_dirs['rag'])

# Step 5: Evaluate
print("Evaluating system...")
eval_report = evaluation.generate_system_report(
    clf_trainer, gen_model, clf_model, tokenized_clf, id2label
)
evaluation.save_evaluation_report(eval_report, output_dirs['evaluation'])

print("Training complete!")


# ==============================================================================
# EXAMPLE 2: Using Trained Models for Inference
# ==============================================================================

from src import inference, rag
from transformers import AutoTokenizer

# Paths to trained models
clf_path = "./models/classifier/lora_weights"
gen_path = "./models/generator/lora_weights"
rag_path = "./models/rag"

# Load models
print("Loading models...")
clf_tokenizer = AutoTokenizer.from_pretrained(clf_path)
clf_model, clf_pipe = inference.load_classifier_for_inference(
    "bert-base-uncased", clf_path, clf_tokenizer
)

gen_tokenizer = AutoTokenizer.from_pretrained(gen_path)
gen_model, gen_pipe = inference.load_generator_for_inference(
    "Qwen/Qwen1.5-1.8B", gen_path, gen_tokenizer
)

# Load RAG
rag_index, rag_model, rag_texts, rag_data = rag.load_rag_artifacts(rag_path)

# Process user input
user_input = "I've been having suicidal thoughts for weeks"
result = rag.process_with_rag(
    user_input,
    clf_pipe,
    gen_pipe,
    rag_index,
    rag_model,
    rag_texts,
    rag_data,
    user_location="United States"
)

# Display results
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nResponse:\n{result['response']}")

if result['retrieved_resources']:
    print("\nCrisis Resources:")
    for resource in result['retrieved_resources']:
        data = resource['data']
        print(f"  {data['country']}: {data.get('hotline', data.get('resource', ''))}")


# ==============================================================================
# EXAMPLE 3: Batch Processing Multiple Inputs
# ==============================================================================

# Process multiple user inputs
test_cases = [
    "I'm stressed about work",
    "I'm having suicidal thoughts",
    "Can you help me find mental health support?",
    "I feel hopeless and alone"
]

results = []
for user_input in test_cases:
    result = rag.process_with_rag(
        user_input, clf_pipe, gen_pipe,
        rag_index, rag_model, rag_texts, rag_data
    )
    results.append(result)
    
    # Print summary
    print(f"Input: {user_input}")
    print(f"Risk: {result['risk_level']}")
    print(f"Confidence: {result['confidence']:.2%}\n")

# Analyze results
safe_count = sum(1 for r in results if r['risk_level'] == 'safe')
concerning_count = sum(1 for r in results if r['risk_level'] == 'concerning')
high_risk_count = sum(1 for r in results if r['risk_level'] == 'high-risk')

print(f"Summary: Safe={safe_count}, Concerning={concerning_count}, High-Risk={high_risk_count}")


# ==============================================================================
# EXAMPLE 4: Custom Data Processing
# ==============================================================================

# Load custom data
import pandas as pd

# Your custom dataset
df_custom = pd.read_csv("my_conversations.csv")

# Apply risk labeling
df_custom['risk_label'] = df_custom['text'].apply(data.assign_risk_label)

# Check distribution
print(df_custom['risk_label'].value_counts())

# Balance and prepare
df_balanced = data.balance_classification_dataset(df_custom)
dataset, label2id, id2label = data.prepare_classifier_dataset(df_balanced)

print(f"Train: {len(dataset['train'])}")
print(f"Val: {len(dataset['validation'])}")
print(f"Test: {len(dataset['test'])}")


# ==============================================================================
# EXAMPLE 5: Fine-tuning with Custom Hyperparameters
# ==============================================================================

from src import training

# Custom training arguments
custom_args = training.TrainingArguments(
    output_dir="./custom_classifier",
    per_device_train_batch_size=16,  # Larger batch
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=5,  # More epochs
    learning_rate=5e-4,  # Higher learning rate
    weight_decay=0.01,
    warmup_steps=200,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True
)

# Create custom trainer
custom_trainer = training.create_classifier_trainer(
    clf_model, custom_args, 
    tokenized_clf["train"], 
    tokenized_clf["validation"]
)

# Train with custom settings
custom_trainer.train()


# ==============================================================================
# EXAMPLE 6: Evaluating Specific Aspects
# ==============================================================================

from src import evaluation
import numpy as np

# Get predictions
predictions_output = clf_trainer.predict(tokenized_clf["test"])
predictions = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids

# Detailed evaluation
eval_results = evaluation.evaluate_classifier(clf_trainer, tokenized_clf, id2label)

print("Evaluation Results:")
print(f"  Accuracy: {eval_results['metrics']['eval_accuracy']:.4f}")
print(f"  F1 Score: {eval_results['metrics']['eval_f1']:.4f}")
print(f"  Precision: {eval_results['metrics']['eval_precision']:.4f}")
print(f"  Recall: {eval_results['metrics']['eval_recall']:.4f}")

# Plot confusion matrix
labels = [id2label[i] for i in range(len(id2label))]
evaluation.plot_confusion_matrix(eval_results['confusion_matrix'], labels)

# Detailed per-class report
print("\nPer-class Report:")
for label, metrics in eval_results['report'].items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"  {label}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1-score']:.4f}")


# ==============================================================================
# EXAMPLE 7: Interactive Chat Interface
# ==============================================================================

def interactive_chat():
    """Run interactive chat with the system"""
    print("InnerLight Chat Interface")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Get location for resources
        location = input("Your location (US/UK/Canada/Australia or press Enter): ").strip() or "US"
        
        # Process input
        result = rag.process_with_rag(
            user_input, clf_pipe, gen_pipe,
            rag_index, rag_model, rag_texts, rag_data,
            location
        )
        
        # Display response
        print(f"\nRisk Assessment: {result['risk_level'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nCounselor: {result['response']}\n")
        
        if result['retrieved_resources']:
            print("Nearby Crisis Resources:")
            for r in result['retrieved_resources']:
                d = r['data']
                print(f"  • {d['country']}: {d.get('number', 'N/A')}")
        
        print()

# Uncomment to run interactive chat
# interactive_chat()


# ==============================================================================
# EXAMPLE 8: Model Information and Inspection
# ==============================================================================

# Print model statistics
print("Model Information:")
print("=" * 50)

models.print_model_info(clf_model, "Classifier")
models.print_model_info(gen_model, "Generator")

# Calculate model size
def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 * 1024)

print(f"\nClassifier Size: {get_model_size_mb(clf_model):.2f} MB")
print(f"Generator Size: {get_model_size_mb(gen_model):.2f} MB")

# Inspect LoRA configuration
from peft import LoraConfig

print("\nLoRA Configuration:")
print(f"  Rank (r): 16")
print(f"  Alpha: 32")
print(f"  Dropout: 0.05")
print(f"  Target Modules: query, value (classifier) / q_proj, k_proj, v_proj, o_proj (generator)")


# ==============================================================================
# EXAMPLE 9: Performance Benchmarking
# ==============================================================================

import time

def benchmark_inference(user_input, num_runs=5):
    """Benchmark inference speed"""
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        result = rag.process_with_rag(
            user_input, clf_pipe, gen_pipe,
            rag_index, rag_model, rag_texts, rag_data
        )
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Inference Benchmark ({num_runs} runs)")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Std Dev: {std_time:.2f}s")
    print(f"  Min: {min(times):.2f}s")
    print(f"  Max: {max(times):.2f}s")
    
    return avg_time

test_input = "I'm feeling really bad and don't know what to do"
benchmark_inference(test_input, num_runs=3)


# ==============================================================================
# EXAMPLE 10: Saving and Loading Artifacts
# ==============================================================================

import os
import pickle
import json

# Save trained models
clf_model.save_pretrained("./checkpoints/classifier")
clf_tokenizer.save_pretrained("./checkpoints/classifier")

gen_model.save_pretrained("./checkpoints/generator")
gen_tokenizer.save_pretrained("./checkpoints/generator")

# Save evaluation results
eval_report = evaluation.generate_system_report(
    clf_trainer, gen_model, clf_model, tokenized_clf, id2label
)

with open("./results/evaluation.json", 'w') as f:
    json.dump(
        {k: str(v) if isinstance(v, np.ndarray) else v 
         for k, v in eval_report.items()},
        f, indent=2
    )

# Load saved models
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM

loaded_clf = AutoModelForSequenceClassification.from_pretrained("./checkpoints/classifier")
loaded_gen = AutoModelForCausalLM.from_pretrained("./checkpoints/generator")

print("Models saved and loaded successfully!")
