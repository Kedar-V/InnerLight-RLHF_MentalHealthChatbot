"""
Reward model for RLHF training
Learns to rate response quality based on teacher model feedback and human preferences
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class RewardModel:
    """
    Reward model that learns to score responses based on quality.
    Uses a classification head to rate response pairs.
    """
    
    def __init__(self, base_model_name="distilbert-base-uncased", use_lora=True):
        """
        Initialize reward model.
        
        Args:
            base_model_name (str): Base model for sequence classification
            use_lora (bool): Whether to use LoRA fine-tuning
        """
        self.base_model_name = base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_lora = use_lora
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load model for sequence classification (binary: good/bad response)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,  # Regression task: score between 0-1
            problem_type="regression"
        )
        
        self.model.to(self.device)
        
        # Apply LoRA if enabled
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["query", "value"],
                bias="none",
                task_type="SEQ_CLS"
            )
            self.model = get_peft_model(self.model, lora_config)
        
        print(f"✓ Reward model initialized: {base_model_name}")
    
    def score_response(self, prompt, response):
        """
        Score a single response.
        
        Args:
            prompt (str): Input prompt
            response (str): Response to score
            
        Returns:
            float: Score between 0-1
        """
        # Combine prompt and response
        combined = f"{prompt} [SEP] {response}"
        
        inputs = self.tokenizer(
            combined,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs.logits[0, 0]).item()
        
        return score
    
    def score_batch(self, prompts, responses):
        """
        Score multiple responses.
        
        Args:
            prompts (list): List of prompts
            responses (list): List of responses
            
        Returns:
            list: Scores for each response
        """
        scores = []
        for prompt, response in zip(prompts, responses):
            score = self.score_response(prompt, response)
            scores.append(score)
        return scores
    
    def score_response_pair(self, prompt, response_a, response_b):
        """
        Compare two responses and return preferences.
        
        Args:
            prompt (str): Input prompt
            response_a (str): First response
            response_b (str): Second response
            
        Returns:
            dict: Scores and preference
        """
        score_a = self.score_response(prompt, response_a)
        score_b = self.score_response(prompt, response_b)
        
        return {
            "score_a": score_a,
            "score_b": score_b,
            "preferred": "response_a" if score_a > score_b else "response_b",
            "margin": abs(score_a - score_b)
        }


class TeacherBasedReward:
    """
    Reward function based on teacher model likelihood and preference learning.
    Combines direct teacher feedback with learned preferences.
    """
    
    def __init__(self, teacher_model, reward_model=None, use_teacher_likelihood=True):
        """
        Initialize teacher-based reward.
        
        Args:
            teacher_model: Teacher model for scoring
            reward_model: Optional learned reward model
            use_teacher_likelihood (bool): Use teacher model likelihood as reward
        """
        self.teacher_model = teacher_model
        self.reward_model = reward_model
        self.use_teacher_likelihood = use_teacher_likelihood
    
    def compute_reward(self, prompt, response, use_learned=False):
        """
        Compute reward score for a response.
        
        Args:
            prompt (str): Input prompt
            response (str): Response to evaluate
            use_learned (bool): Use learned reward model if available
            
        Returns:
            float: Reward score
        """
        reward = 0.0
        
        # Use teacher model likelihood if enabled
        if self.use_teacher_likelihood:
            teacher_score = self.teacher_model.score_response(prompt, response)
            reward = teacher_score
        
        # Combine with learned reward model if available
        if use_learned and self.reward_model:
            learned_score = self.reward_model.score_response(prompt, response)
            # Weight teacher more heavily initially
            reward = 0.7 * teacher_score + 0.3 * learned_score if self.use_teacher_likelihood else learned_score
        
        return reward
    
    def compute_batch_rewards(self, prompts, responses, use_learned=False):
        """
        Compute rewards for batches.
        
        Args:
            prompts (list): List of prompts
            responses (list): List of responses
            use_learned (bool): Use learned reward model
            
        Returns:
            list: Reward scores
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.compute_reward(prompt, response, use_learned)
            rewards.append(reward)
        return rewards
    
    def compare_responses(self, prompt, response_a, response_b, use_learned=False):
        """
        Compare two responses based on rewards.
        
        Args:
            prompt (str): Input prompt
            response_a (str): First response
            response_b (str): Second response
            use_learned (bool): Use learned reward model
            
        Returns:
            dict: Comparison results
        """
        reward_a = self.compute_reward(prompt, response_a, use_learned)
        reward_b = self.compute_reward(prompt, response_b, use_learned)
        
        return {
            "reward_a": reward_a,
            "reward_b": reward_b,
            "preferred": "response_a" if reward_a > reward_b else "response_b",
            "margin": abs(reward_a - reward_b)
        }


class PreferenceDataset:
    """
    Dataset for learning response preferences from teacher feedback.
    """
    
    def __init__(self):
        """Initialize preference dataset."""
        self.prompts = []
        self.responses = []
        self.rewards = []
        self.preferences = []  # List of (better_idx, worse_idx) tuples
    
    def add_preference(self, prompt, response_a, response_b, preferred_response):
        """
        Add preference pair.
        
        Args:
            prompt (str): Input prompt
            response_a (str): First response
            response_b (str): Second response
            preferred_response (str): Which response is preferred
        """
        self.prompts.append(prompt)
        self.responses.append([response_a, response_b])
        
        if preferred_response == response_a:
            self.preferences.append((0, 1))
        else:
            self.preferences.append((1, 0))
    
    def add_reward(self, prompt, response, reward_score):
        """
        Add response with reward score.
        
        Args:
            prompt (str): Input prompt
            response (str): Response
            reward_score (float): Reward/quality score
        """
        self.prompts.append(prompt)
        self.responses.append(response)
        self.rewards.append(reward_score)
    
    def get_preference_pairs(self):
        """Get preference training pairs."""
        return list(zip(self.prompts, self.responses, self.preferences))
    
    def get_reward_pairs(self):
        """Get reward training pairs."""
        return list(zip(self.prompts, self.responses, self.rewards))
    
    def __len__(self):
        """Get dataset size."""
        return len(self.prompts)


def calculate_reward_statistics(teacher_model, prompts, responses):
    """
    Calculate reward statistics for responses.
    
    Args:
        teacher_model: Teacher model
        prompts (list): List of prompts
        responses (list): List of responses
        
    Returns:
        dict: Statistics (mean, std, min, max rewards)
    """
    import numpy as np
    
    rewards = [teacher_model.score_response(p, r) for p, r in zip(prompts, responses)]
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "rewards": rewards
    }
