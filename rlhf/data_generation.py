"""
Data generation and preparation for RLHF training
Generate preference pairs using teacher model
"""

import torch
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


class TeacherDataGenerator:
    """Generate training data using teacher model for RLHF"""
    
    def __init__(self, teacher_model, student_model):
        """
        Initialize data generator.
        
        Args:
            teacher_model: TeacherModel instance
            student_model: StudentModel instance
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
    
    def generate_preference_pairs(self, prompts: List[str], num_samples: int = 4) -> Dict:
        """
        Generate preference pairs by having student generate multiple responses
        and teacher ranking them.
        
        Args:
            prompts (list): Input prompts
            num_samples (int): Number of responses to generate per prompt
            
        Returns:
            dict: Preference pairs with better/worse responses
        """
        preference_pairs = []
        all_data = []
        
        print(f"Generating {len(prompts)} preference pairs with {num_samples} samples each...")
        
        for prompt in tqdm(prompts):
            # Generate multiple responses from student
            inputs = self.student_model.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.student_model.model.generate(
                    **inputs,
                    max_length=256,
                    num_return_sequences=num_samples,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.student_model.tokenizer.eos_token_id
                )
            
            responses = self.student_model.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            # Score all responses with teacher
            scores = []
            for response in responses:
                score = self.teacher_model.score_response(prompt, response)
                scores.append(score)
                all_data.append({
                    'prompt': prompt,
                    'response': response,
                    'score': score
                })
            
            # Create preference pairs (better vs worse)
            scores_array = np.array(scores)
            best_idx = np.argmax(scores_array)
            worst_idx = np.argmin(scores_array)
            
            preference_pairs.append({
                'prompt': prompt,
                'chosen': responses[best_idx],
                'rejected': responses[worst_idx],
                'chosen_score': scores[best_idx],
                'rejected_score': scores[worst_idx]
            })
        
        return {
            'preference_pairs': preference_pairs,
            'all_responses': all_data
        }
    
    def generate_teacher_responses(self, prompts: List[str]) -> List[Dict]:
        """
        Generate golden responses from teacher model.
        
        Args:
            prompts (list): Input prompts
            
        Returns:
            list: Prompts with teacher responses
        """
        teacher_data = []
        
        print(f"Generating teacher responses for {len(prompts)} prompts...")
        
        for prompt in tqdm(prompts):
            response = self.teacher_model.generate_response(
                prompt, max_length=256, num_samples=1, temperature=0.7
            )[0]
            
            teacher_data.append({
                'prompt': prompt,
                'response': response,
                'source': 'teacher'
            })
        
        return teacher_data
    
    def generate_diverse_responses(self, prompts: List[str], num_per_prompt: int = 3) -> List[Dict]:
        """
        Generate diverse responses using different temperatures.
        
        Args:
            prompts (list): Input prompts
            num_per_prompt (int): Number of responses per prompt
            
        Returns:
            list: Prompts with diverse responses
        """
        diverse_data = []
        temperatures = [0.5, 0.7, 0.9][:num_per_prompt]
        
        print(f"Generating diverse responses with {len(temperatures)} temperatures...")
        
        for prompt in tqdm(prompts):
            for temp in temperatures:
                response = self.teacher_model.generate_response(
                    prompt, max_length=256, num_samples=1, temperature=temp
                )[0]
                
                diverse_data.append({
                    'prompt': prompt,
                    'response': response,
                    'temperature': temp
                })
        
        return diverse_data


class PreferenceDataBuilder:
    """Build structured preference datasets"""
    
    def __init__(self):
        """Initialize preference data builder."""
        self.data = {
            'preference_pairs': [],
            'scored_responses': [],
            'contrastive_pairs': []
        }
    
    def add_preference_pair(self, prompt: str, chosen: str, rejected: str, margin: float = 0.0):
        """
        Add a preference pair.
        
        Args:
            prompt (str): Input prompt
            chosen (str): Preferred response
            rejected (str): Non-preferred response
            margin (float): Preference margin
        """
        self.data['preference_pairs'].append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'margin': margin
        })
    
    def add_scored_response(self, prompt: str, response: str, score: float):
        """
        Add scored response.
        
        Args:
            prompt (str): Input prompt
            response (str): Response
            score (float): Quality score
        """
        self.data['scored_responses'].append({
            'prompt': prompt,
            'response': response,
            'score': score
        })
    
    def add_contrastive_pair(self, prompt: str, positive: str, negatives: List[str]):
        """
        Add contrastive learning pair.
        
        Args:
            prompt (str): Input prompt
            positive (str): Positive example
            negatives (list): Negative examples
        """
        self.data['contrastive_pairs'].append({
            'prompt': prompt,
            'positive': positive,
            'negatives': negatives
        })
    
    def build_hf_dataset(self, dataset_type: str = 'preference_pairs'):
        """
        Convert to HuggingFace Dataset.
        
        Args:
            dataset_type (str): Type of dataset to build
            
        Returns:
            Dataset: HuggingFace Dataset
        """
        from datasets import Dataset
        
        if dataset_type == 'preference_pairs':
            data = self.data['preference_pairs']
        elif dataset_type == 'scored_responses':
            data = self.data['scored_responses']
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return Dataset.from_list(data)
    
    def __len__(self):
        """Get total number of samples."""
        return sum(len(v) for v in self.data.values())


class SyntheticDataGenerator:
    """Generate synthetic training data for initial training"""
    
    @staticmethod
    def generate_mental_health_prompts(num_prompts: int = 100) -> List[str]:
        """
        Generate synthetic mental health conversation prompts.
        
        Args:
            num_prompts (int): Number of prompts to generate
            
        Returns:
            list: Generated prompts
        """
        prompt_templates = [
            "I've been feeling {emotion} lately. Can you help?",
            "I'm struggling with {issue}. What should I do?",
            "I feel {emotion} about {situation}. Any advice?",
            "How do I deal with {issue}?",
            "I'm worried about {concern}. Is this normal?",
            "I can't stop thinking about {issue}. Help!",
            "Everything feels {emotion}. What's wrong with me?",
            "I have {symptom} and it's affecting my {area}.",
            "{situation} happened and I feel {emotion}.",
            "Is it normal to feel {emotion} about {issue}?",
        ]
        
        emotions = ["sad", "anxious", "stressed", "overwhelmed", "hopeless", "alone", "angry", "scared"]
        issues = ["work stress", "relationships", "health", "finances", "family", "identity", "self-worth"]
        situations = ["losing a job", "breakup", "failure", "rejection", "criticism", "loss"]
        symptoms = ["insomnia", "panic attacks", "headaches", "fatigue", "intrusive thoughts"]
        areas = ["work", "relationships", "studies", "daily life"]
        concerns = ["depression", "anxiety", "stress", "loneliness", "worthlessness"]
        
        prompts = []
        import random
        
        for _ in range(num_prompts):
            template = random.choice(prompt_templates)
            prompt = template.format(
                emotion=random.choice(emotions),
                issue=random.choice(issues),
                situation=random.choice(situations),
                symptom=random.choice(symptoms),
                area=random.choice(areas),
                concern=random.choice(concerns)
            )
            prompts.append(prompt)
        
        return prompts
    
    @staticmethod
    def generate_response_pairs(num_pairs: int = 50) -> List[Tuple[str, str, str]]:
        """
        Generate synthetic prompt-response pairs (good, bad).
        
        Args:
            num_pairs (int): Number of pairs to generate
            
        Returns:
            list: Tuples of (prompt, good_response, bad_response)
        """
        good_response_templates = [
            "I'm sorry you're going through this. It's completely understandable to feel this way. Consider reaching out to a professional.",
            "That sounds really challenging. Have you thought about talking to someone about this? Your feelings are valid.",
            "It's brave of you to share this. Many people experience similar feelings. Professional support could really help.",
            "I hear you. This must be difficult. Remember you're not alone, and there are people who want to help.",
            "Your feelings matter. This is a common struggle. Let's think about some concrete steps you could take.",
        ]
        
        bad_response_templates = [
            "Just get over it. Everyone has problems.",
            "That's not a real problem. Stop complaining.",
            "You're being dramatic. Just think positive.",
            "That's your own fault. You should've known better.",
            "Nobody cares about your problems. Deal with it yourself.",
        ]
        
        prompts = SyntheticDataGenerator.generate_mental_health_prompts(num_pairs)
        pairs = []
        
        import random
        for prompt in prompts[:num_pairs]:
            good_resp = random.choice(good_response_templates)
            bad_resp = random.choice(bad_response_templates)
            pairs.append((prompt, good_resp, bad_resp))
        
        return pairs


def batch_generate_data(teacher_model, student_model, prompts: List[str], 
                       batch_size: int = 32) -> Dict:
    """
    Generate training data in batches.
    
    Args:
        teacher_model: Teacher model instance
        student_model: Student model instance
        prompts (list): Input prompts
        batch_size (int): Batch size for generation
        
    Returns:
        dict: Generated data
    """
    generator = TeacherDataGenerator(teacher_model, student_model)
    all_data = {'preference_pairs': [], 'all_responses': []}
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_result = generator.generate_preference_pairs(batch_prompts)
        
        all_data['preference_pairs'].extend(batch_result['preference_pairs'])
        all_data['all_responses'].extend(batch_result['all_responses'])
    
    return all_data
