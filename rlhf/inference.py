"""
RLHF Inference - Use trained student model for inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class RLHFInference:
    """Inference with RLHF-trained student model"""
    
    def __init__(self, student_model_path, base_model_name="Qwen/Qwen1.5-0.5B"):
        """
        Initialize inference.
        
        Args:
            student_model_path (str): Path to trained student model
            base_model_name (str): Base model identifier
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.model_path = student_model_path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with LoRA
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(base_model, student_model_path)
        self.model.eval()
        
        print(f"✓ RLHF Student model loaded from {student_model_path}")
    
    def generate_response(self, prompt, max_length=256, temperature=0.7, top_p=0.9):
        """
        Generate response using trained model.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            
        Returns:
            str: Generated response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response
    
    def batch_generate(self, prompts, max_length=256, temperature=0.7):
        """
        Generate responses for batch of prompts.
        
        Args:
            prompts (list): List of prompts
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            
        Returns:
            list: Generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, max_length, temperature)
            responses.append(response)
        return responses


def compare_student_teacher(student_model_path, teacher_model, test_prompts):
    """
    Compare student and teacher model outputs.
    
    Args:
        student_model_path (str): Path to trained student model
        teacher_model: Loaded teacher model
        test_prompts (list): Test prompts
        
    Returns:
        list: Comparison results
    """
    inference = RLHFInference(student_model_path)
    
    results = []
    for prompt in test_prompts:
        student_response = inference.generate_response(prompt)
        teacher_response = teacher_model.generate_response(prompt)[0]
        
        student_score = teacher_model.score_response(prompt, student_response)
        teacher_score = teacher_model.score_response(prompt, teacher_response)
        
        results.append({
            'prompt': prompt,
            'student_response': student_response,
            'teacher_response': teacher_response,
            'student_score': float(student_score),
            'teacher_score': float(teacher_score),
        })
    
    return results
