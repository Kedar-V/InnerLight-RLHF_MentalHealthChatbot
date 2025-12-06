"""
Teacher and Student model setup for RLHF training
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class TeacherModel:
    """Large language model to generate training data and preferences"""
    
    def __init__(self, model_name="meta-llama/Llama-2-13b-hf", use_quantization=True):
        """
        Initialize teacher model.
        
        Args:
            model_name (str): Model identifier (large model)
            use_quantization (bool): Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if available
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"✓ Teacher model loaded: {model_name}")
    
    def generate_response(self, prompt, max_length=256, num_samples=1, temperature=0.7):
        """
        Generate responses using teacher model.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum generation length
            num_samples (int): Number of samples to generate
            temperature (float): Sampling temperature
            
        Returns:
            list: Generated responses
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_samples,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return responses
    
    def score_response(self, prompt, response):
        """
        Score response quality using teacher model likelihood.
        
        Args:
            prompt (str): Input prompt
            response (str): Response to score
            
        Returns:
            float: Log probability score
        """
        full_text = f"{prompt}{response}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        # Convert loss to probability score (lower loss = higher quality)
        score = -loss.item()
        return score


class StudentModel:
    """Small language model <500M parameters to be trained via RLHF"""
    
    AVAILABLE_MODELS = {
        'gpt2': ('gpt2', 124_000_000),
        'distilgpt2': ('distilgpt2', 82_000_000),
        'phi-2': ('microsoft/phi-2', 2_700_000_000),  # Too large, for reference
        'qwen-0.5b': ('Qwen/Qwen1.5-0.5B', 500_000_000),
        'qwen-1.8b': ('Qwen/Qwen1.5-1.8B', 1_800_000_000),
        'gemma-2b': ('google/gemma-2b', 2_000_000_000),
        'open-llama-3b': ('openlm-research/open_llama_3b', 3_000_000_000),
    }
    
    def __init__(self, model_name='qwen-0.5b', use_lora=True, use_quantization=False):
        """
        Initialize student model.
        
        Args:
            model_name (str): Model identifier or key from AVAILABLE_MODELS
            use_lora (bool): Whether to use LoRA fine-tuning
            use_quantization (bool): Whether to use 4-bit quantization
        """
        # Resolve model name
        if model_name in self.AVAILABLE_MODELS:
            self.model_name, self.param_count = self.AVAILABLE_MODELS[model_name]
        else:
            self.model_name = model_name
            self.param_count = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_lora = use_lora
        self.use_quantization = use_quantization
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional quantization
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Apply LoRA if enabled
        if use_lora:
            self.model = prepare_model_for_kbit_training(self.model) if use_quantization else self.model
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
        
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n✓ Student model loaded: {self.model_name}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        if total_params > 0:
            print(f"  Trainable %: {trainable_params/total_params*100:.2f}%")
        if self.param_count:
            print(f"  Model Size: {self.param_count/1e6:.1f}M parameters")
    
    def get_model_size_mb(self):
        """Get model size in MB"""
        total_size = 0
        for param in self.model.parameters():
            total_size += param.nelement() * param.element_size()
        return total_size / (1024 * 1024)


def create_teacher_student_pair(
    teacher_model_name="meta-llama/Llama-2-13b-hf",
    student_model_key='qwen-0.5b',
    student_use_lora=True
):
    """
    Create paired teacher and student models.
    
    Args:
        teacher_model_name (str): Large teacher model identifier
        student_model_key (str): Student model key from AVAILABLE_MODELS
        student_use_lora (bool): Whether to use LoRA for student
        
    Returns:
        tuple: (teacher_model, student_model)
    """
    print("\n" + "="*60)
    print("Initializing Teacher-Student Model Pair")
    print("="*60)
    
    teacher = TeacherModel(teacher_model_name, use_quantization=True)
    student = StudentModel(student_model_key, use_lora=student_use_lora, use_quantization=False)
    
    return teacher, student


def calculate_model_efficiency(student_model, teacher_model):
    """
    Calculate training efficiency metrics.
    
    Args:
        student_model: Student model object
        teacher_model: Teacher model object
        
    Returns:
        dict: Efficiency metrics
    """
    student_params = sum(p.numel() for p in student_model.model.parameters())
    teacher_params = sum(p.numel() for p in teacher_model.model.parameters())
    
    student_trainable = sum(p.numel() for p in student_model.model.parameters() if p.requires_grad)
    
    return {
        "student_total_params": student_params,
        "student_trainable_params": student_trainable,
        "student_trainable_percent": student_trainable / student_params * 100,
        "teacher_total_params": teacher_params,
        "compression_ratio": teacher_params / student_params,
        "student_size_mb": student_model.get_model_size_mb()
    }
