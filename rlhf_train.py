"""
Main RLHF training pipeline
Train student model using teacher model feedback
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from rlhf.teacher_student import (
    TeacherModel, StudentModel, create_teacher_student_pair, calculate_model_efficiency
)
from rlhf.reward_model import RewardModel, TeacherBasedReward
from rlhf.data_generation import (
    TeacherDataGenerator, SyntheticDataGenerator, PreferenceDataBuilder
)
from rlhf.rlhf_trainer import RLHFTrainer, DPOTrainer, KnowledgeDistillationTrainer


class RLHFPipeline:
    """Complete RLHF training pipeline"""
    
    def __init__(self, output_dir="./rlhf_models", config=None):
        """
        Initialize RLHF pipeline.
        
        Args:
            output_dir (str): Directory to save models and logs
            config (dict): Configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or self._get_default_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.teacher_model = None
        self.student_model = None
        self.reward_model = None
        
        # Training metrics
        self.metrics = {
            'training': [],
            'evaluation': [],
            'generation_samples': []
        }
        
        print(f"✓ RLHF Pipeline initialized")
        print(f"  Output directory: {self.run_dir}")
        self._save_config()
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            'teacher_model_name': 'meta-llama/Llama-2-7b-hf',
            'student_model_key': 'qwen-0.5b',
            'reward_model_name': 'distilbert-base-uncased',
            'learning_rate': 5e-5,
            'num_rlhf_epochs': 3,
            'num_dpo_epochs': 2,
            'batch_size': 8,
            'accumulation_steps': 4,
            'max_generation_length': 256,
            'num_samples_per_prompt': 4,
            'num_training_prompts': 1000,
            'eval_prompts_num': 50,
        }
    
    def _save_config(self):
        """Save configuration to file."""
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def initialize_models(self):
        """Initialize teacher and student models."""
        print("\n" + "="*70)
        print("STAGE 1: INITIALIZING MODELS")
        print("="*70)
        
        self.teacher_model, self.student_model = create_teacher_student_pair(
            teacher_model_name=self.config['teacher_model_name'],
            student_model_key=self.config['student_model_key'],
            student_use_lora=True
        )
        
        # Print efficiency metrics
        efficiency = calculate_model_efficiency(self.student_model, self.teacher_model)
        print("\n✓ Model Efficiency:")
        print(f"  Student Total Parameters: {efficiency['student_total_params']:,}")
        print(f"  Student Trainable Parameters: {efficiency['student_trainable_params']:,}")
        print(f"  Trainable Percentage: {efficiency['student_trainable_percent']:.2f}%")
        print(f"  Compression Ratio (Teacher/Student): {efficiency['compression_ratio']:.2f}x")
        print(f"  Student Model Size: {efficiency['student_size_mb']:.2f} MB")
        
        # Save efficiency metrics
        with open(self.run_dir / "efficiency_metrics.json", 'w') as f:
            json.dump(efficiency, f, indent=2, default=float)
    
    def initialize_reward_model(self):
        """Initialize reward model for preference learning."""
        print("\n" + "="*70)
        print("STAGE 2: INITIALIZING REWARD MODEL")
        print("="*70)
        
        self.reward_model = RewardModel(
            base_model_name=self.config['reward_model_name'],
            use_lora=True
        )
        
        print("✓ Reward model initialized for preference learning")
    
    def generate_training_data(self):
        """Generate training data using teacher model."""
        print("\n" + "="*70)
        print("STAGE 3: GENERATING TRAINING DATA")
        print("="*70)
        
        # Generate synthetic prompts
        print("\nGenerating synthetic mental health prompts...")
        prompts = SyntheticDataGenerator.generate_mental_health_prompts(
            num_prompts=self.config['num_training_prompts']
        )
        
        print(f"✓ Generated {len(prompts)} training prompts")
        
        # Generate preference pairs using teacher
        print("\nGenerating preference pairs from teacher model...")
        data_gen = TeacherDataGenerator(self.teacher_model, self.student_model)
        
        training_data = data_gen.generate_preference_pairs(
            prompts,
            num_samples=self.config['num_samples_per_prompt']
        )
        
        # Save generated data
        with open(self.run_dir / "training_data.json", 'w') as f:
            json.dump(training_data['preference_pairs'][:100], f, indent=2, default=float)  # Save sample
        
        print(f"✓ Generated {len(training_data['preference_pairs'])} preference pairs")
        print(f"✓ Generated {len(training_data['all_responses'])} total responses")
        
        return training_data
    
    def train_with_rlhf(self, training_data):
        """Train student model with RLHF."""
        print("\n" + "="*70)
        print("STAGE 4: RLHF TRAINING")
        print("="*70)
        
        trainer = RLHFTrainer(
            self.student_model,
            self.teacher_model,
            reward_model=self.reward_model,
            learning_rate=self.config['learning_rate']
        )
        
        # Prepare training dataset
        train_prompts = [pair['prompt'] for pair in training_data['preference_pairs']]
        train_data = [{'prompt': p} for p in train_prompts]
        
        # Train
        print(f"\nTraining for {self.config['num_rlhf_epochs']} epochs...")
        metrics = trainer.train_epoch(
            train_data,
            num_epochs=self.config['num_rlhf_epochs'],
            accumulation_steps=self.config['accumulation_steps']
        )
        
        print(f"✓ RLHF Training Complete")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print(f"  Average Loss: {metrics['avg_loss']:.4f}")
        print(f"  Number of Updates: {metrics['num_updates']}")
        
        self.metrics['training'].append(metrics)
        
        # Save checkpoint
        self._save_checkpoint("rlhf")
        
        return metrics
    
    def train_with_dpo(self, training_data):
        """Train with Direct Preference Optimization."""
        print("\n" + "="*70)
        print("STAGE 5: DPO TRAINING (Optional)")
        print("="*70)
        
        print("DPO training can be added for improved preference alignment")
        print("Skipping in main pipeline for now")
    
    def train_with_distillation(self, training_data):
        """Train with knowledge distillation."""
        print("\n" + "="*70)
        print("STAGE 6: KNOWLEDGE DISTILLATION (Optional)")
        print("="*70)
        
        print("Knowledge distillation can be applied after RLHF")
        print("Skipping in main pipeline for now")
    
    def evaluate(self):
        """Evaluate trained model."""
        print("\n" + "="*70)
        print("STAGE 7: EVALUATION")
        print("="*70)
        
        # Generate evaluation prompts
        eval_prompts = SyntheticDataGenerator.generate_mental_health_prompts(
            num_prompts=self.config['eval_prompts_num']
        )
        
        trainer = RLHFTrainer(
            self.student_model,
            self.teacher_model,
            learning_rate=self.config['learning_rate']
        )
        
        # Evaluate
        print(f"\nEvaluating on {len(eval_prompts)} prompts...")
        eval_metrics = trainer.evaluate(eval_prompts, num_samples=2)
        
        print(f"✓ Evaluation Complete")
        print(f"  Mean Score: {eval_metrics['mean_score']:.4f}")
        print(f"  Std Score: {eval_metrics['std_score']:.4f}")
        print(f"  Min Score: {eval_metrics['min_score']:.4f}")
        print(f"  Max Score: {eval_metrics['max_score']:.4f}")
        
        self.metrics['evaluation'].append(eval_metrics)
        
        # Save sample generations
        sample_generations = []
        for i, (prompt, responses) in enumerate(zip(eval_prompts[:5], eval_metrics['generation_output']['responses'])):
            sample_generations.append({
                'prompt': prompt,
                'responses': responses[:2]  # Save top 2
            })
        
        with open(self.run_dir / "sample_generations.json", 'w') as f:
            json.dump(sample_generations, f, indent=2)
        
        return eval_metrics
    
    def compare_models(self, test_prompts=None):
        """Compare teacher and student models."""
        print("\n" + "="*70)
        print("STAGE 8: TEACHER vs STUDENT COMPARISON")
        print("="*70)
        
        if test_prompts is None:
            test_prompts = SyntheticDataGenerator.generate_mental_health_prompts(num_prompts=10)
        
        comparison_results = []
        
        for prompt in test_prompts:
            teacher_response = self.teacher_model.generate_response(prompt)[0]
            
            inputs = self.student_model.tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.student_model.model.generate(
                    **inputs,
                    max_length=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            student_response = self.student_model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            teacher_score = self.teacher_model.score_response(prompt, teacher_response)
            student_score = self.teacher_model.score_response(prompt, student_response)
            
            comparison_results.append({
                'prompt': prompt,
                'teacher_response': teacher_response,
                'student_response': student_response,
                'teacher_score': float(teacher_score),
                'student_score': float(student_score),
                'score_diff': float(teacher_score - student_score)
            })
        
        # Calculate statistics
        student_scores = [r['student_score'] for r in comparison_results]
        
        print(f"\n✓ Comparison Complete")
        print(f"  Average Student Score: {np.mean(student_scores):.4f}")
        print(f"  Std Dev: {np.std(student_scores):.4f}")
        
        # Save results
        with open(self.run_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        return comparison_results
    
    def _save_checkpoint(self, stage_name):
        """Save model checkpoint."""
        checkpoint_dir = self.run_dir / f"checkpoint_{stage_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save student model
        student_path = checkpoint_dir / "student_model"
        self.student_model.model.save_pretrained(str(student_path))
        self.student_model.tokenizer.save_pretrained(str(student_path))
        
        print(f"  Checkpoint saved: {checkpoint_dir}")
    
    def run_complete_pipeline(self):
        """Run complete RLHF training pipeline."""
        print("\n" + "="*70)
        print("INNERLIGHT RLHF TRAINING PIPELINE")
        print("="*70)
        
        try:
            # Stage 1: Initialize models
            self.initialize_models()
            
            # Stage 2: Initialize reward model
            self.initialize_reward_model()
            
            # Stage 3: Generate data
            training_data = self.generate_training_data()
            
            # Stage 4: RLHF training
            self.train_with_rlhf(training_data)
            
            # Stage 5: Evaluation
            self.evaluate()
            
            # Stage 6: Comparison
            self.compare_models()
            
            # Save final metrics
            self._save_final_metrics()
            
            print("\n" + "="*70)
            print("✓ TRAINING COMPLETE!")
            print("="*70)
            print(f"\nAll artifacts saved to: {self.run_dir}")
            
        except Exception as e:
            print(f"\n✗ Pipeline failed with error:")
            print(f"  {str(e)}")
            raise
    
    def _save_final_metrics(self):
        """Save all final metrics."""
        metrics_file = self.run_dir / "final_metrics.json"
        
        # Prepare metrics for serialization
        metrics_to_save = {
            'training_metrics': self.metrics['training'],
            'evaluation_metrics': [
                {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                 for k, v in m.items() if k != 'generation_output'}
                for m in self.metrics['evaluation']
            ],
            'timestamp': self.timestamp
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_file}")


def main():
    """Main entry point for RLHF training."""
    # Configure pipeline
    config = {
        'teacher_model_name': 'meta-llama/Llama-2-7b-hf',  # Or use smaller model for testing
        'student_model_key': 'qwen-0.5b',
        'reward_model_name': 'distilbert-base-uncased',
        'learning_rate': 5e-5,
        'num_rlhf_epochs': 2,
        'batch_size': 8,
        'num_training_prompts': 500,  # Reduced for faster testing
        'eval_prompts_num': 20,
    }
    
    # Create and run pipeline
    pipeline = RLHFPipeline(output_dir="./rlhf_models", config=config)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
