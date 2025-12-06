"""
RLHF Training Examples and Usage Patterns
"""

# ==============================================================================
# EXAMPLE 1: Complete RLHF Pipeline
# ==============================================================================

from rlhf_train import RLHFPipeline

# Configure training
config = {
    'teacher_model_name': 'meta-llama/Llama-2-7b-hf',
    'student_model_key': 'qwen-0.5b',
    'reward_model_name': 'distilbert-base-uncased',
    'learning_rate': 5e-5,
    'num_rlhf_epochs': 3,
    'num_training_prompts': 1000,
}

# Create pipeline and run
pipeline = RLHFPipeline(output_dir="./rlhf_models", config=config)
pipeline.run_complete_pipeline()


# ==============================================================================
# EXAMPLE 2: Teacher-Student Model Setup
# ==============================================================================

from rlhf.teacher_student import create_teacher_student_pair, calculate_model_efficiency

# Initialize models
teacher, student = create_teacher_student_pair(
    teacher_model_name="meta-llama/Llama-2-7b-hf",
    student_model_key='qwen-0.5b',
    student_use_lora=True
)

# Calculate efficiency
efficiency = calculate_model_efficiency(student, teacher)
print(f"Compression Ratio: {efficiency['compression_ratio']:.1f}x")
print(f"Student Size: {efficiency['student_size_mb']:.2f} MB")
print(f"Trainable %: {efficiency['student_trainable_percent']:.2f}%")


# ==============================================================================
# EXAMPLE 3: Generate Training Data
# ==============================================================================

from rlhf.data_generation import TeacherDataGenerator, SyntheticDataGenerator

# Generate synthetic prompts
prompts = SyntheticDataGenerator.generate_mental_health_prompts(num_prompts=100)

# Generate preference pairs
data_gen = TeacherDataGenerator(teacher, student)
training_data = data_gen.generate_preference_pairs(prompts, num_samples=4)

print(f"Generated {len(training_data['preference_pairs'])} preference pairs")

# Access preference pairs
for pair in training_data['preference_pairs'][:3]:
    print(f"\nPrompt: {pair['prompt']}")
    print(f"Chosen (score: {pair['chosen_score']:.4f}): {pair['chosen'][:100]}...")
    print(f"Rejected (score: {pair['rejected_score']:.4f}): {pair['rejected'][:100]}...")


# ==============================================================================
# EXAMPLE 4: RLHF Training
# ==============================================================================

from rlhf.rlhf_trainer import RLHFTrainer

# Initialize trainer
trainer = RLHFTrainer(
    student_model=student,
    teacher_model=teacher,
    learning_rate=5e-5
)

# Generate and score responses
generation_output = trainer.generate_and_score(
    prompts=['Tell me about mental health support'],
    num_return_sequences=4
)

# Access scores
print(f"Generated {len(generation_output['responses'][0])} responses")
print(f"Scores: {generation_output['scores'][0]}")

# Train
train_data = [{'prompt': p} for p in prompts[:50]]
metrics = trainer.train_epoch(train_data, num_epochs=1)

print(f"Training Loss: {metrics['avg_loss']:.4f}")


# ==============================================================================
# EXAMPLE 5: Reward Model Training
# ==============================================================================

from rlhf.reward_model import RewardModel, PreferenceDataset

# Initialize reward model
reward_model = RewardModel(base_model_name="distilbert-base-uncased", use_lora=True)

# Score responses
score = reward_model.score_response(
    "I'm feeling anxious",
    "That's completely understandable. Have you considered talking to someone?"
)
print(f"Response Score: {score:.4f}")

# Create preference dataset
pref_dataset = PreferenceDataset()
pref_dataset.add_preference(
    prompt="I'm feeling depressed",
    response_a="I understand. Professional support can really help.",
    response_b="Just stop being sad.",
    preferred_response="I understand. Professional support can really help."
)

print(f"Preference dataset size: {len(pref_dataset)}")


# ==============================================================================
# EXAMPLE 6: Inference with Trained Model
# ==============================================================================

from rlhf.inference import RLHFInference

# Load trained student model
inference = RLHFInference(
    student_model_path="./rlhf_models/run_20240101_120000/checkpoint_rlhf/student_model",
    base_model_name="Qwen/Qwen1.5-0.5B"
)

# Generate response
prompt = "I've been struggling with anxiety"
response = inference.generate_response(prompt, max_length=256, temperature=0.7)
print(f"Student Response: {response}")

# Batch generation
prompts_batch = [
    "I feel lonely",
    "I'm having trouble sleeping",
    "I'm worried about my future"
]
responses = inference.batch_generate(prompts_batch)
for p, r in zip(prompts_batch, responses):
    print(f"\nPrompt: {p}")
    print(f"Response: {r[:100]}...")


# ==============================================================================
# EXAMPLE 7: Evaluate Student Model
# ==============================================================================

# Test prompts
test_prompts = SyntheticDataGenerator.generate_mental_health_prompts(num_prompts=10)

# Evaluate
eval_metrics = trainer.evaluate(test_prompts, num_samples=2)

print(f"Mean Score: {eval_metrics['mean_score']:.4f}")
print(f"Std Dev: {eval_metrics['std_score']:.4f}")

# Access generated responses
for i, responses in enumerate(eval_metrics['generation_output']['responses'][:3]):
    print(f"\nGeneration {i+1}:")
    for j, r in enumerate(responses):
        print(f"  Sample {j+1}: {r[:80]}...")


# ==============================================================================
# EXAMPLE 8: Compare Teacher vs Student
# ==============================================================================

# Generate comparison
comparison_results = pipeline.compare_models(test_prompts=test_prompts[:5])

# Analyze results
for result in comparison_results:
    print(f"\nPrompt: {result['prompt']}")
    print(f"Teacher Score: {result['teacher_score']:.4f}")
    print(f"Student Score: {result['student_score']:.4f}")
    print(f"Score Difference: {result['score_diff']:.4f}")


# ==============================================================================
# EXAMPLE 9: DPO Training (Direct Preference Optimization)
# ==============================================================================

from rlhf.rlhf_trainer import DPOTrainer

# Initialize DPO trainer
dpo_trainer = DPOTrainer(
    student_model=student,
    reference_model=None,
    learning_rate=5e-5,
    beta=0.1
)

# Compute DPO loss for preference pairs
for pair in training_data['preference_pairs'][:5]:
    loss = dpo_trainer.compute_dpo_loss(
        prompt=pair['prompt'],
        chosen_response=pair['chosen'],
        rejected_response=pair['rejected']
    )
    print(f"DPO Loss: {loss.item():.4f}")


# ==============================================================================
# EXAMPLE 10: Knowledge Distillation
# ==============================================================================

from rlhf.rlhf_trainer import KnowledgeDistillationTrainer

# Initialize distillation trainer
kd_trainer = KnowledgeDistillationTrainer(
    student_model=student,
    teacher_model=teacher,
    temperature=3.0,
    alpha=0.7,
    learning_rate=5e-5
)

# Use on training data (requires proper batch format)
print("Knowledge distillation trainer ready")
print("Can be applied after RLHF training for additional refinement")


# ==============================================================================
# EXAMPLE 11: Custom Configuration
# ==============================================================================

# Create custom config for specific use case
custom_config = {
    'teacher_model_name': 'mistralai/Mistral-7B-v0.1',
    'student_model_key': 'qwen-0.5b',
    'reward_model_name': 'roberta-base',
    'learning_rate': 1e-4,
    'num_rlhf_epochs': 5,
    'num_training_prompts': 2000,
    'eval_prompts_num': 100,
    'batch_size': 16,
    'accumulation_steps': 8,
    'max_generation_length': 512,
}

# Create pipeline with custom config
custom_pipeline = RLHFPipeline(output_dir="./custom_rlhf", config=custom_config)


# ==============================================================================
# EXAMPLE 12: Manual Step-by-Step Training
# ==============================================================================

# Step 1: Initialize
pipeline = RLHFPipeline()
pipeline.initialize_models()
pipeline.initialize_reward_model()

# Step 2: Generate data
training_data = pipeline.generate_training_data()

# Step 3: Train
rlhf_metrics = pipeline.train_with_rlhf(training_data)

# Step 4: Evaluate
eval_metrics = pipeline.evaluate()

# Step 5: Compare
comparison = pipeline.compare_models()

# Step 6: Save
pipeline._save_checkpoint("final")


# ==============================================================================
# EXAMPLE 13: Synthetic Data Generation
# ==============================================================================

from rlhf.data_generation import SyntheticDataGenerator

# Generate mental health prompts
prompts = SyntheticDataGenerator.generate_mental_health_prompts(num_prompts=50)
for i, p in enumerate(prompts[:3]):
    print(f"{i+1}. {p}")

# Generate synthetic good/bad response pairs
response_pairs = SyntheticDataGenerator.generate_response_pairs(num_pairs=20)
for prompt, good, bad in response_pairs[:2]:
    print(f"\nPrompt: {prompt}")
    print(f"  Good: {good[:60]}...")
    print(f"  Bad:  {bad[:60]}...")


# ==============================================================================
# EXAMPLE 14: Monitor Training Progress
# ==============================================================================

import json

# Load training metrics
metrics_path = "./rlhf_models/run_20240101_120000/final_metrics.json"
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# Analyze metrics
training_loss = metrics['training_metrics'][0]['avg_loss']
eval_score = metrics['evaluation_metrics'][0]['mean_score']

print(f"Final Training Loss: {training_loss:.4f}")
print(f"Evaluation Score: {eval_score:.4f}")


# ==============================================================================
# EXAMPLE 15: Load and Use Trained Model
# ==============================================================================

from rlhf.inference import RLHFInference

# Load model
model = RLHFInference(
    student_model_path="./rlhf_models/run_20240101_120000/checkpoint_rlhf/student_model",
    base_model_name="Qwen/Qwen1.5-0.5B"
)

# Interactive mode
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit']:
        break
    
    response = model.generate_response(user_input)
    print(f"\nStudent: {response}")
