# Python Project Structure - InnerLight

## 📁 File Organization

```
InnerLight-RLHF_MentalHealthChatbot/
│
├── 📄 train.py                    # Main training entry point
├── 📄 inference.py                # Inference and demo scripts
├── 📄 requirements.txt            # Python dependencies
├── 📄 QUICKSTART.py               # Quick reference guide
├── 📄 README.md                   # Original documentation
├── 📄 README_STRUCTURE.md         # Detailed documentation
│
└── 📂 src/                        # Source code package
    ├── 📄 __init__.py             # Package initialization
    ├── 📄 data.py                 # Data loading and processing
    ├── 📄 models.py               # Model configuration
    ├── 📄 training.py             # Training utilities
    ├── 📄 inference.py            # Inference utilities
    ├── 📄 rag.py                  # RAG system
    └── 📄 evaluation.py           # Evaluation tools
```

## 📝 File Descriptions

### Core Scripts

#### `train.py` (210 lines)
Main training pipeline orchestrating all 5 stages:
- Data download and preparation
- Risk classifier training
- Response generator training
- RAG system setup
- System evaluation
- Model saving and reporting

**Usage**: `python train.py`

#### `inference.py` (135 lines)
Inference and demonstration utilities:
- Demo mode with predefined test cases
- Interactive chat mode
- Model loading
- Output formatting
- Crisis resource integration

**Usage**: `python inference.py --mode [demo|interactive] --classifier_path ... --generator_path ... --rag_path ...`

### Source Modules

#### `src/data.py` (120 lines)
Data handling and preprocessing:
- Risk label assignment based on keywords
- Dataset loading from multiple sources
- Class balancing via resampling
- Train/validation/test splitting
- Dataset preparation for HuggingFace

**Key Functions**:
- `assign_risk_label()` - Semantic risk classification
- `load_classification_data()` - Load and label conversations
- `balance_classification_dataset()` - Resample classes
- `prepare_classifier_dataset()` - Create HuggingFace datasets
- `prepare_generation_dataset()` - Combine multiple data sources

#### `src/models.py` (90 lines)
Model setup and configuration:
- BERT classifier with LoRA
- Qwen generator with LoRA
- 4-bit quantization configuration
- Model information utilities

**Key Functions**:
- `get_quantization_config()` - BitsAndBytes config
- `setup_classifier_model()` - BERT + LoRA setup
- `setup_generator_model()` - Qwen + LoRA setup
- `print_model_info()` - Parameter statistics

#### `src/training.py` (145 lines)
Training configuration and utilities:
- Training argument builders
- Tokenization functions
- Metrics computation
- Trainer creation
- Prompt formatting

**Key Functions**:
- `compute_classification_metrics()` - Evaluate classifier
- `get_classifier_training_args()` - Classifier config
- `get_generator_training_args()` - Generator config
- `create_classifier_trainer()` - Build trainer
- `create_generator_trainer()` - Build trainer
- `tokenize_classifier_data()` - Tokenize for classification
- `tokenize_generation_data()` - Tokenize for generation
- `build_generation_prompt()` - Format prompts

#### `src/inference.py` (160 lines)
Inference utilities and pipelines:
- Model loading for inference
- Classifier inference
- Generator inference
- Single-stage and RAG-enhanced inference
- Crisis resource database
- Output formatting

**Key Functions**:
- `load_classifier_for_inference()` - Load trained classifier
- `load_generator_for_inference()` - Load trained generator
- `get_crisis_resources()` - Retrieve location-specific resources
- `process_user_input()` - Complete inference pipeline
- `format_inference_output()` - Format results

#### `src/rag.py` (210 lines)
Retrieval-Augmented Generation system:
- Crisis resource database
- FAISS indexing
- Semantic resource retrieval
- RAG-enhanced inference
- Artifact persistence

**Key Functions**:
- `format_resources_to_text()` - Convert data to embeddings
- `create_rag_index()` - Build FAISS index
- `retrieve_crisis_resources()` - Query resource database
- `save_rag_artifacts()` - Persist to disk
- `load_rag_artifacts()` - Load from disk
- `process_with_rag()` - RAG-enhanced inference
- `format_rag_output()` - Format results

#### `src/evaluation.py` (155 lines)
Model evaluation and reporting:
- Classifier evaluation
- Confusion matrix visualization
- System-wide evaluation
- Metric computation
- Report generation and saving

**Key Functions**:
- `evaluate_classifier()` - Comprehensive evaluation
- `plot_confusion_matrix()` - Visualize results
- `generate_system_report()` - Full system evaluation
- `save_evaluation_report()` - Save to JSON

### Configuration Files

#### `requirements.txt` (20 lines)
Python package dependencies:
- PyTorch 2.0+
- Transformers 4.36+
- PEFT (LoRA implementation)
- BitsAndBytes (quantization)
- Datasets and data processing
- RAG/embedding libraries

#### `src/__init__.py` (3 lines)
Package initialization with metadata

### Documentation

#### `README.md`
Original notebook documentation

#### `README_STRUCTURE.md` (390 lines)
Comprehensive project documentation covering:
- Project structure
- Features and capabilities
- Installation instructions
- Usage examples
- API reference
- Model architecture
- Performance metrics
- Safety features
- Configuration guide
- Contributing guidelines

#### `QUICKSTART.py` (280 lines)
Quick reference guide with:
- Installation steps
- Training and inference commands
- Code examples
- Module overview
- Common tasks
- Troubleshooting
- Performance expectations
- Useful links

## 🏗️ Architecture

### Training Pipeline
```
Raw Data
  ↓
[src/data.py] Data Loading & Labeling
  ↓
Data Splitting & Balancing
  ↓
├→ [src/models.py] Classifier Setup
  │   ↓
  │ [src/training.py] Tokenization
  │   ↓
  │ [src/training.py] Trainer Creation
  │   ↓
  │ Train Classifier
  │   ↓
  │ [src/evaluation.py] Evaluation
  │
├→ [src/models.py] Generator Setup
  │   ↓
  │ [src/training.py] Tokenization
  │   ↓
  │ [src/training.py] Trainer Creation
  │   ↓
  │ Train Generator
  │
└→ [src/rag.py] RAG Setup
    ↓
  Create FAISS Index
    ↓
  Save Artifacts
```

### Inference Pipeline
```
User Input
  ↓
[src/inference.py] Load Models
  ↓
Classifier Pipeline
  ├→ Risk Classification
  │   ↓
  │ [src/inference.py] Risk Level
  │
Generator Pipeline
  ├→ Response Generation
  │   ↓
  │ [src/inference.py] Safe Response
  │
[src/rag.py] RAG System
  ├→ Resource Retrieval
      ↓
    [src/rag.py] Location-aware Resources
      ↓
    Formatted Output
```

## 📊 Module Dependencies

```
train.py
  └── src/data.py
  └── src/models.py
  └── src/training.py
  └── src/evaluation.py
  └── src/rag.py

inference.py
  └── src/inference.py
  └── src/rag.py

src/training.py
  └── src/models.py

src/inference.py
  └── transformers
  └── peft

src/rag.py
  └── faiss
  └── sentence-transformers

src/evaluation.py
  └── sklearn
  └── matplotlib
  └── seaborn
```

## 🔑 Key Design Patterns

### 1. **Modular Architecture**
Each concern separated into its own module:
- Data handling
- Model configuration
- Training logic
- Inference and serving
- Retrieval systems
- Evaluation

### 2. **Configuration-Driven**
All settings configurable:
- Model names and sizes
- Training parameters
- LoRA configurations
- Batch sizes and learning rates

### 3. **Reusable Functions**
Functions designed for:
- Training, validation, and testing
- Inference and serving
- Experimentation and prototyping
- Production deployment

### 4. **Object-Oriented Integration**
- HuggingFace Trainer pattern
- PEFT PeftModel integration
- Transformers pipeline API
- FAISS IndexFlatL2

## 💾 Typical Workflow

### Training
```python
from src import data, models, training

# 1. Load data
dataset, labels = data.prepare_classifier_dataset(...)

# 2. Setup model
model, tokenizer = models.setup_classifier_model()

# 3. Create trainer
trainer = training.create_classifier_trainer(...)

# 4. Train
trainer.train()

# 5. Evaluate
trainer.evaluate(...)
```

### Inference
```python
from src import inference, rag

# 1. Load models
clf_model, clf_pipe = inference.load_classifier_for_inference(...)
gen_model, gen_pipe = inference.load_generator_for_inference(...)
rag_index, rag_model, rag_texts, rag_data = rag.load_rag_artifacts(...)

# 2. Process input
result = rag.process_with_rag(
    user_input, clf_pipe, gen_pipe,
    rag_index, rag_model, rag_texts, rag_data
)

# 3. Format output
output = rag.format_rag_output(result)
```

## 📈 Code Statistics

| File | Lines | Purpose |
|------|-------|---------|
| train.py | 210 | Training orchestration |
| inference.py | 135 | Inference scripts |
| src/data.py | 120 | Data handling |
| src/models.py | 90 | Model setup |
| src/training.py | 145 | Training utilities |
| src/inference.py | 160 | Inference utilities |
| src/rag.py | 210 | RAG system |
| src/evaluation.py | 155 | Evaluation tools |
| **Total** | **1,225** | **~1.2K lines** |

## 🚀 Getting Started

1. **Install**: `pip install -r requirements.txt`
2. **Train**: `python train.py`
3. **Infer**: `python inference.py --mode interactive ...`

See `QUICKSTART.py` for detailed examples and `README_STRUCTURE.md` for comprehensive documentation.
