"""
RLHF training pipeline
Train student model using reinforcement learning from teacher feedback
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm


class RLHFTrainer:
    """
    Trainer for RLHF using teacher model feedback.
    Implements advantage actor-critic style training.
    """
    
    def __init__(self, student_model, teacher_model, reward_model=None, 
                 learning_rate=5e-5, device="cuda"):
        """
        Initialize RLHF trainer.
        
        Args:
            student_model: StudentModel object
            teacher_model: TeacherModel object
            reward_model: RewardModel object (optional)
            learning_rate (float): Learning rate for optimization
            device (str): Device to train on
        """
        self.student_model = student_model.model
        self.student_tokenizer = student_model.tokenizer
        self.teacher_model = teacher_model
        self.reward_model = reward_model
        
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer
        self.optimizer = AdamW(self.student_model.parameters(), lr=learning_rate)
    
    def generate_and_score(self, prompts, max_length=256, num_return_sequences=4):
        """
        Generate multiple responses and score them with teacher.
        
        Args:
            prompts (list): Input prompts
            max_length (int): Maximum generation length
            num_return_sequences (int): Number of responses per prompt
            
        Returns:
            dict: Generated responses and their scores
        """
        all_responses = []
        all_scores = []
        
        for prompt in prompts:
            # Generate multiple responses
            inputs = self.student_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.student_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.student_tokenizer.eos_token_id
                )
            
            responses = self.student_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            # Score with teacher model
            scores = [self.teacher_model.score_response(prompt, r) for r in responses]
            
            all_responses.append(responses)
            all_scores.append(scores)
        
        return {
            "prompts": prompts,
            "responses": all_responses,
            "scores": all_scores
        }
    
    def compute_policy_loss(self, prompts, responses, scores, baseline=0.0):
        """
        Compute policy gradient loss with advantage weighting.
        
        Args:
            prompts (list): Input prompts
            responses (list): Generated responses
            scores (list): Reward scores from teacher
            baseline (float): Baseline for advantage estimation
            
        Returns:
            torch.Tensor: Loss to minimize
        """
        loss = 0.0
        total_samples = 0
        
        for prompt, response_list, score_list in zip(prompts, responses, scores):
            # Compute advantages (normalized scores)
            scores_array = np.array(score_list)
            advantages = scores_array - baseline
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute log probabilities for each response
            for response, advantage in zip(response_list, advantages):
                full_text = f"{prompt}{response}"
                inputs = self.student_tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                # Get logits and compute loss
                outputs = self.student_model(**inputs)
                logits = outputs.logits
                
                # Compute log probability of response tokens
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                loss_per_token = F.nll_loss(
                    log_probs.view(-1, log_probs.size(-1)),
                    shift_labels.view(-1),
                    reduction='none'
                )
                
                # Weight by advantage (higher reward = stronger gradient)
                response_loss = (loss_per_token.mean() * -advantage)
                loss += response_loss
                total_samples += 1
        
        return loss / (total_samples + 1e-8)
    
    def train_epoch(self, train_data, num_epochs=1, accumulation_steps=4):
        """
        Train for one epoch with RLHF.
        
        Args:
            train_data (list): List of dicts with 'prompt' and 'metadata'
            num_epochs (int): Number of training epochs
            accumulation_steps (int): Gradient accumulation steps
            
        Returns:
            dict: Training metrics
        """
        self.student_model.train()
        
        total_loss = 0.0
        num_updates = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, sample in enumerate(tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}")):
                prompt = sample['prompt']
                
                # Generate and score responses
                generation_output = self.generate_and_score([prompt], num_return_sequences=4)
                
                responses = generation_output['responses'][0]
                scores = generation_output['scores'][0]
                
                # Compute loss
                loss = self.compute_policy_loss([prompt], [responses], [scores])
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                epoch_loss += loss.item()
                
                # Optimizer step with accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    num_updates += 1
            
            total_loss += epoch_loss
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_data):.4f}")
        
        return {
            "total_loss": total_loss,
            "num_updates": num_updates,
            "avg_loss": total_loss / len(train_data)
        }
    
    def evaluate(self, eval_prompts, num_samples=3):
        """
        Evaluate student model on prompts.
        
        Args:
            eval_prompts (list): Evaluation prompts
            num_samples (int): Number of samples per prompt
            
        Returns:
            dict: Evaluation metrics
        """
        self.student_model.eval()
        
        generation_output = self.generate_and_score(eval_prompts, num_return_sequences=num_samples)
        
        # Compute metrics
        all_scores = np.array([scores for scores in generation_output['scores']])
        
        return {
            "mean_score": all_scores.mean(),
            "std_score": all_scores.std(),
            "min_score": all_scores.min(),
            "max_score": all_scores.max(),
            "generation_output": generation_output
        }


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    Trains student model to prefer better responses without explicit reward model.
    """
    
    def __init__(self, student_model, reference_model=None, learning_rate=5e-5, 
                 beta=0.1, device="cuda"):
        """
        Initialize DPO trainer.
        
        Args:
            student_model: StudentModel object
            reference_model: Reference model (copy of initial student for KL penalty)
            learning_rate (float): Learning rate
            beta (float): KL divergence penalty weight
            device (str): Device to train on
        """
        self.student_model = student_model.model
        self.student_tokenizer = student_model.tokenizer
        self.reference_model = reference_model
        self.beta = beta
        self.device = device
        
        self.optimizer = AdamW(self.student_model.parameters(), lr=learning_rate)
    
    def compute_dpo_loss(self, prompt, chosen_response, rejected_response):
        """
        Compute DPO loss for preference pair.
        
        Args:
            prompt (str): Input prompt
            chosen_response (str): Preferred response
            rejected_response (str): Non-preferred response
            
        Returns:
            torch.Tensor: DPO loss
        """
        # Compute log probabilities
        chosen_loss = self._compute_log_prob(prompt, chosen_response)
        rejected_loss = self._compute_log_prob(prompt, rejected_response)
        
        # DPO loss: log sigmoid of difference
        dpo_loss = -torch.nn.functional.logsigmoid(self.beta * (chosen_loss - rejected_loss))
        
        return dpo_loss
    
    def _compute_log_prob(self, prompt, response):
        """Compute average log probability of response."""
        full_text = f"{prompt}{response}"
        inputs = self.student_tokenizer(
            full_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.student_model(**inputs)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                shift_labels.view(-1)
            )
        
        return loss


class KnowledgeDistillationTrainer:
    """
    Knowledge distillation trainer.
    Train student to mimic teacher model outputs.
    """
    
    def __init__(self, student_model, teacher_model, temperature=3.0, 
                 alpha=0.7, learning_rate=5e-5, device="cuda"):
        """
        Initialize knowledge distillation trainer.
        
        Args:
            student_model: StudentModel object
            teacher_model: TeacherModel object (for logits)
            temperature (float): Softening temperature
            alpha (float): Weight for distillation loss vs task loss
            learning_rate (float): Learning rate
            device (str): Device to train on
        """
        self.student_model = student_model.model
        self.student_tokenizer = student_model.tokenizer
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        self.optimizer = AdamW(self.student_model.parameters(), lr=learning_rate)
    
    def compute_distillation_loss(self, inputs, labels):
        """
        Compute distillation loss.
        
        Args:
            inputs: Tokenized inputs
            labels: Target labels
            
        Returns:
            torch.Tensor: Distillation loss
        """
        # Student forward pass
        student_outputs = self.student_model(**inputs)
        student_logits = student_outputs.logits
        
        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model.model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Distillation loss: KL divergence at higher temperature
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Task loss: standard cross entropy
        task_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        
        return total_loss
    
    def train_epoch(self, train_dataloader, num_epochs=1):
        """
        Train for one epoch with knowledge distillation.
        
        Args:
            train_dataloader: DataLoader with input/label pairs
            num_epochs (int): Number of epochs
            
        Returns:
            dict: Training metrics
        """
        self.student_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                # Compute loss
                loss = self.compute_distillation_loss(inputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            "total_loss": total_loss,
            "avg_loss": total_loss / num_batches
        }


def create_training_dataset(prompts, responses=None, scores=None, format='prompt'):
    """
    Create training dataset for RLHF.
    
    Args:
        prompts (list): List of prompts
        responses (list): Optional list of responses
        scores (list): Optional list of scores
        format (str): Dataset format ('prompt', 'prompt_response', 'prompt_pair')
        
    Returns:
        Dataset: HuggingFace Dataset
    """
    if format == 'prompt':
        data = {'text': prompts}
    elif format == 'prompt_response' and responses:
        data = {
            'prompt': prompts,
            'response': responses,
            'score': scores if scores else [0.0] * len(prompts)
        }
    elif format == 'prompt_pair':
        # Expects list of (chosen, rejected) pairs in responses
        data = {
            'prompt': prompts,
            'chosen': [r[0] for r in responses],
            'rejected': [r[1] for r in responses]
        }
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return Dataset.from_dict(data)
