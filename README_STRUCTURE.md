# InnerLight: RLHF Mental Health Chatbot

## Overview

InnerLight is a comprehensive suicide risk detection and safety response system that combines:

1. **Risk Classification**: BERT-based classifier to detect suicide risk levels
2. **Safety Response Generation**: LoRA-fine-tuned LLM for empathetic responses
3. **RAG Integration**: Retrieval-augmented generation for crisis resources
4. **Parameter Efficiency**: LoRA optimization for resource-constrained environments

## Project Structure

```
InnerLight-RLHF_MentalHealthChatbot/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── data.py               # Data loading and preparation
│   ├── models.py             # Model configuration and setup
│   ├── training.py           # Training utilities
│   ├── inference.py          # Inference utilities
│   ├── rag.py                # RAG system for crisis resources
│   └── evaluation.py         # Evaluation and metrics
├── train.py                  # Main training script
├── inference.py              # Inference demo and interactive mode
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── InnerLight_Complete_Training.ipynb  # Original notebook
```

## Features

### Stage 1: Risk Classification
- 3-way classification: Safe / Concerning / High-Risk
- BERT-base model with LoRA fine-tuning
- 4-bit quantization for memory efficiency
- Keyword-based and semantic analysis

### Stage 2: Safety Response Generation
- Qwen 1.5B LLM with LoRA adaptation
- Context-aware response generation
- Safety instruction prompting
- Emotion-aware response generation

### Stage 3: RAG System
- FAISS-based retrieval system
- Multi-country crisis hotline database
- Semantic similarity search
- Location-aware resource recommendation

### Stage 4: Evaluation
- Comprehensive classification metrics
- Confusion matrix analysis
- Model efficiency tracking
- Performance benchmarking

## Installation

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ GPU memory (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd InnerLight-RLHF_MentalHealthChatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle credentials (for dataset download):
```bash
mkdir -p ~/.kaggle
# Copy your kaggle.json to ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Training

Run the complete training pipeline:

```bash
python train.py
```

This will:
1. Download and prepare datasets
2. Train the risk classifier
3. Train the response generator
4. Set up the RAG system
5. Evaluate the complete system

### Inference

#### Demo Mode
Run pre-configured test cases:

```bash
python inference.py --mode demo \
    --classifier_path path/to/classifier \
    --generator_path path/to/generator \
    --rag_path path/to/rag_artifacts
```

#### Interactive Mode
Chat interactively with the system:

```bash
python inference.py --mode interactive \
    --classifier_path path/to/classifier \
    --generator_path path/to/generator \
    --rag_path path/to/rag_artifacts
```

### Programmatic Usage

```python
from src import inference, rag
from transformers import AutoTokenizer

# Load models
clf_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
clf_model, clf_pipe = inference.load_classifier_for_inference(
    "bert-base-uncased", classifier_path, clf_tokenizer
)

gen_tokenizer = AutoTokenizer.from_pretrained(generator_path)
gen_model, gen_pipe = inference.load_generator_for_inference(
    "Qwen/Qwen1.5-1.8B", generator_path, gen_tokenizer
)

# Load RAG
rag_index, rag_model, rag_texts, rag_data = rag.load_rag_artifacts(rag_path)

# Process input
result = rag.process_with_rag(
    "I'm having suicidal thoughts",
    clf_pipe, gen_pipe,
    rag_index, rag_model, rag_texts, rag_data,
    user_location="United States"
)

# Display output
output = rag.format_rag_output(result)
print(output)
```

## Data

### Datasets Used
1. **Human & LLM Mental Health Conversations** (Kaggle)
   - Mental health dialogue dataset
   - Context and response pairs
   - ~11k conversations

2. **MentalChat16K** (HuggingFace)
   - Additional mental health conversations
   - Diverse counseling scenarios

### Data Processing
- Risk labeling based on keyword analysis
- Class balancing via resampling
- Train/Val/Test split (70/15/15)
- Stratified splitting to maintain label distribution

## Model Architecture

### Classifier
- **Base Model**: BERT-base-uncased (110M parameters)
- **LoRA Config**: r=16, alpha=32
- **Quantization**: 4-bit NF4
- **Trainable Parameters**: ~1M (0.9% of total)

### Generator
- **Base Model**: Qwen 1.5-1.8B (1.8B parameters)
- **LoRA Config**: r=16, alpha=32
- **Quantization**: 4-bit NF4
- **Trainable Parameters**: ~20M (1.1% of total)

## Performance

Expected metrics after training:
- **Classifier F1 Score**: ~80-85%
- **Classifier Accuracy**: ~82-87%
- **Model Parameters**: ~2B total (mostly frozen)
- **Training Time**: ~4-6 hours on single A100 GPU

## Safety Features

1. **Risk-Aware Responses**: Different response strategies based on risk level
2. **Crisis Resource Integration**: Automatic crisis hotline recommendation
3. **Multi-Country Support**: Resources for US, UK, Canada, Australia
4. **Safety Instruction Prompting**: Explicit guidance for crisis situations

## API Reference

### Data Module (`src/data.py`)
- `load_classification_data()`: Load and label conversations
- `balance_classification_dataset()`: Resample for class balance
- `prepare_classifier_dataset()`: Create HuggingFace datasets
- `prepare_generation_dataset()`: Prepare response generation data

### Models Module (`src/models.py`)
- `setup_classifier_model()`: Configure BERT with LoRA
- `setup_generator_model()`: Configure Qwen with LoRA
- `print_model_info()`: Display model statistics

### Training Module (`src/training.py`)
- `get_classifier_training_args()`: Training configuration
- `get_generator_training_args()`: Training configuration
- `create_classifier_trainer()`: Initialize trainer
- `create_generator_trainer()`: Initialize trainer

### Inference Module (`src/inference.py`)
- `load_classifier_for_inference()`: Load trained classifier
- `load_generator_for_inference()`: Load trained generator
- `process_user_input()`: Complete inference pipeline
- `get_crisis_resources()`: Retrieve location-specific resources

### RAG Module (`src/rag.py`)
- `create_rag_index()`: Create FAISS index
- `retrieve_crisis_resources()`: Query resource database
- `process_with_rag()`: RAG-enhanced inference
- `save/load_rag_artifacts()`: Persistence

### Evaluation Module (`src/evaluation.py`)
- `evaluate_classifier()`: Comprehensive evaluation
- `generate_system_report()`: Full system evaluation
- `plot_confusion_matrix()`: Visualization

## Configuration

Key hyperparameters can be adjusted in:
- `src/training.py`: Learning rate, batch size, epochs
- `src/models.py`: LoRA parameters (r, alpha)
- `train.py`: Model names, dataset paths

## Crisis Resources

The system includes hotline information for:
- United States: 988 Lifeline
- United Kingdom: Samaritans (116 123)
- Canada: Crisis Services Canada
- Australia: Lifeline (13 11 14)
- International: Find a Helpline directory

## Contributing

Contributions welcome! Areas for improvement:
- Additional language support
- More crisis resource databases
- Fine-tuned response templates
- Real-time model updates
- A/B testing framework

## Limitations

1. **Model Size**: Requires significant GPU memory
2. **Latency**: Inference takes 5-10 seconds
3. **Language**: Primarily English
4. **Accuracy**: Not 100% reliable for edge cases

## Disclaimer

⚠️ **Important**: This system is intended to assist, not replace, professional mental health support. Always escalate to human professionals in crisis situations.

## License

[Insert License Information]

## Authors

Kedar Vaidya

## Acknowledgments

- Hugging Face for transformers and datasets
- PEFT library for parameter-efficient fine-tuning
- Meta for LLaMA foundation
- Alibaba for Qwen models

## Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [email]

---

**Stay Safe. Seek Help. You Matter.** 💙
