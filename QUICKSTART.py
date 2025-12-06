"""
Quick start guide for InnerLight
"""

# ==============================================================================
# INSTALLATION
# ==============================================================================

# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up Kaggle credentials (if downloading datasets)
# mkdir -p ~/.kaggle
# Copy your kaggle.json file to ~/.kaggle/
# chmod 600 ~/.kaggle/kaggle.json


# ==============================================================================
# TRAINING
# ==============================================================================

# Run the complete training pipeline (with all 5 stages)
python train.py

# This will:
# 1. Download datasets from Kaggle and HuggingFace
# 2. Prepare and balance data
# 3. Train suicide risk classifier
# 4. Train response generator
# 5. Create RAG system
# 6. Evaluate and save all artifacts
#
# Output structure:
# /content/drive/MyDrive/NLP/  (or configured path)
# ├── suicide_risk_classifier/
# │   └── lora_weights/
# ├── safety_response_generator/
# │   └── lora_weights/
# ├── rag_artifacts/
# │   ├── crisis_resources.index
# │   └── crisis_resources_data.pkl
# └── evaluation/
#     ├── evaluation_report.json
#     └── confusion_matrix.png


# ==============================================================================
# INFERENCE
# ==============================================================================

# Demo mode (run test cases)
python inference.py --mode demo \
    --classifier_path /path/to/classifier \
    --generator_path /path/to/generator \
    --rag_path /path/to/rag_artifacts

# Interactive mode (chat with system)
python inference.py --mode interactive \
    --classifier_path /path/to/classifier \
    --generator_path /path/to/generator \
    --rag_path /path/to/rag_artifacts


# ==============================================================================
# PROGRAMMATIC USAGE
# ==============================================================================

# Example: Using the system in your code

from src import inference, rag
from transformers import AutoTokenizer

# Paths to trained models
classifier_path = "/path/to/suicide_risk_classifier_lora"
generator_path = "/path/to/safety_generator_lora"
rag_path = "/path/to/rag_artifacts"

# 1. Load models
print("Loading models...")
clf_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
clf_model, clf_pipe = inference.load_classifier_for_inference(
    "bert-base-uncased", classifier_path, clf_tokenizer
)

gen_tokenizer = AutoTokenizer.from_pretrained(generator_path)
gen_model, gen_pipe = inference.load_generator_for_inference(
    "Qwen/Qwen1.5-1.8B", generator_path, gen_tokenizer
)

# 2. Load RAG system
rag_index, rag_model, rag_texts, rag_data = rag.load_rag_artifacts(rag_path)

# 3. Process user input
user_message = "I've been having thoughts of harming myself"
result = rag.process_with_rag(
    user_message,
    clf_pipe,
    gen_pipe,
    rag_index,
    rag_model,
    rag_texts,
    rag_data,
    user_location="United States"
)

# 4. Display results
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Response:\n{result['response']}")
if result['retrieved_resources']:
    print("\nRecommended Crisis Resources:")
    for resource in result['retrieved_resources']:
        print(f"  - {resource['data']['country']}")


# ==============================================================================
# MODULE OVERVIEW
# ==============================================================================

# src/data.py
#   - Data loading and preparation
#   - Risk labeling
#   - Class balancing
#   - Dataset creation

# src/models.py
#   - Model configuration
#   - LoRA setup
#   - Quantization configuration
#   - Parameter efficiency utilities

# src/training.py
#   - Training configurations
#   - Tokenization utilities
#   - Metric computation
#   - Trainer creation

# src/inference.py
#   - Model loading for inference
#   - Single-stage inference
#   - Crisis resource retrieval
#   - Output formatting

# src/rag.py
#   - RAG system setup
#   - FAISS indexing
#   - Resource retrieval
#   - RAG-enhanced inference

# src/evaluation.py
#   - Model evaluation
#   - Metrics computation
#   - Report generation
#   - Visualization


# ==============================================================================
# COMMON TASKS
# ==============================================================================

# Task 1: Train on custom dataset
# - Modify src/data.py load functions
# - Update dataset paths in train.py
# - Run: python train.py

# Task 2: Adjust model hyperparameters
# - Edit src/training.py (learning_rate, batch_size, etc.)
# - Edit src/models.py (LoRA r and alpha values)
# - Re-run training

# Task 3: Evaluate trained models
# - Models are auto-evaluated during training
# - Results saved to evaluation/ directory
# - View evaluation_report.json for details

# Task 4: Deploy to production
# - Load models using inference.py functions
# - Create wrapper API (Flask, FastAPI, etc.)
# - Set up logging and monitoring
# - Configure fallback to human agents


# ==============================================================================
# TROUBLESHOOTING
# ==============================================================================

# Issue: Out of memory (OOM) error
# Solution:
#   - Reduce batch_size in src/training.py
#   - Use smaller base models
#   - Enable gradient checkpointing
#   - Reduce max sequence length

# Issue: Slow inference
# Solution:
#   - Use smaller generator model
#   - Reduce max_new_tokens in pipeline
#   - Run on GPU with CUDA
#   - Batch inference requests

# Issue: Poor classification accuracy
# Solution:
#   - Increase training epochs
#   - Adjust learning rate
#   - Collect more labeled data
#   - Improve keyword-based labeling function

# Issue: Model not loading
# Solution:
#   - Verify path exists: os.path.exists(path)
#   - Check file permissions
#   - Ensure correct model names
#   - Reinstall transformers: pip install -U transformers


# ==============================================================================
# PERFORMANCE EXPECTATIONS
# ==============================================================================

# Training Time
#   - Classifier: 1-2 hours (A100 GPU)
#   - Generator: 2-3 hours (A100 GPU)
#   - Total: 4-6 hours

# Inference Speed
#   - Classifier: 0.1-0.3 seconds
#   - Generator: 2-5 seconds
#   - RAG retrieval: 0.01 seconds
#   - Total latency: 2-6 seconds

# Model Sizes
#   - Classifier: ~110M base + 1M LoRA = 111M total
#   - Generator: ~1.8B base + 20M LoRA = 1.82B total
#   - Memory: ~8GB GPU RAM with 4-bit quantization

# Accuracy
#   - Classifier F1: 80-85%
#   - Risk detection: High precision on extreme cases
#   - Response quality: Subjective, requires human evaluation


# ==============================================================================
# USEFUL LINKS
# ==============================================================================

# Documentation
# - Transformers: https://huggingface.co/docs/transformers/
# - PEFT: https://huggingface.co/docs/peft/
# - Datasets: https://huggingface.co/docs/datasets/
# - FAISS: https://github.com/facebookresearch/faiss

# Models
# - BERT: https://huggingface.co/bert-base-uncased
# - Qwen 1.5: https://huggingface.co/Qwen/Qwen1.5-1.8B

# Datasets
# - Mental Health Conversations: https://www.kaggle.com/datasets/birdy654/
# - MentalChat16K: https://huggingface.co/datasets/ShenLab/MentalChat16K

# Crisis Resources
# - 988 Suicide & Crisis Lifeline: https://988lifeline.org
# - Crisis Text Line: Text HOME to 741741
# - International: https://findahelpline.com
