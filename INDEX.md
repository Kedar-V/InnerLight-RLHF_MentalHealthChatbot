# InnerLight Python Project - Complete Index

## 📚 Documentation Files

| File | Purpose | Size |
|------|---------|------|
| **README.md** | Original project documentation | Original |
| **README_STRUCTURE.md** | Comprehensive guide (390 lines) | Detailed docs |
| **PROJECT_STRUCTURE.md** | Architecture and design patterns | Architecture |
| **QUICKSTART.py** | Quick reference guide (280 lines) | Quick ref |
| **EXAMPLES.py** | 10 usage examples with code (450 lines) | Examples |
| **requirements.txt** | Python dependencies | Dependencies |

## 🐍 Core Python Scripts

### Entry Points

| File | Purpose | Lines |
|------|---------|-------|
| **train.py** | Main training pipeline | 210 |
| **inference.py** | Inference and demo scripts | 135 |

### Source Package (`src/`)

| Module | Purpose | Lines |
|--------|---------|-------|
| `__init__.py` | Package init | 3 |
| `data.py` | Data loading & processing | 120 |
| `models.py` | Model configuration | 90 |
| `training.py` | Training utilities | 145 |
| `inference.py` | Inference utilities | 160 |
| `rag.py` | RAG system | 210 |
| `evaluation.py` | Evaluation tools | 155 |
| **Total** | | **1,225** |

---

## 🎯 Quick Navigation

### I want to...

#### **Get Started Quickly**
1. Read: `QUICKSTART.py` - Installation and basic usage
2. Run: `python train.py` - Start training
3. Explore: `EXAMPLES.py` - See usage patterns

#### **Understand the Architecture**
1. Read: `PROJECT_STRUCTURE.md` - Overview and design
2. Read: `README_STRUCTURE.md` - Detailed documentation
3. Browse: `src/` folder - Review each module

#### **Train Models**
1. Configure: Edit hyperparameters in `src/training.py`
2. Run: `python train.py`
3. Monitor: Check `evaluation/` output directory
4. Reference: See EXAMPLE 1 in `EXAMPLES.py`

#### **Use Pre-trained Models**
1. Load: See EXAMPLE 2 in `EXAMPLES.py`
2. Inference: Run `python inference.py --mode demo`
3. Interactive: Run `python inference.py --mode interactive`

#### **Custom Data**
1. Reference: See EXAMPLE 4 in `EXAMPLES.py`
2. Process: Use `src/data.py` functions
3. Train: Follow `train.py` workflow

#### **Debug Issues**
1. Check: `QUICKSTART.py` - Troubleshooting section
2. Read: `README_STRUCTURE.md` - Limitations section
3. Review: `EXAMPLES.py` - Error handling patterns

#### **Deploy to Production**
1. Load: Use functions from `src/inference.py`
2. Serve: Create Flask/FastAPI wrapper
3. Monitor: Track performance metrics
4. Update: Retrain or fine-tune as needed

---

## 📊 File Statistics

```
Language    Files  Lines
──────────────────────────
Python        16   1,900+
Markdown       4   1,200+
Text           1      20
──────────────────────────
Total         21   3,100+
```

---

## 🏗️ Module Dependency Graph

```
Entry Points
├── train.py
│   ├── src/data.py
│   ├── src/models.py
│   ├── src/training.py
│   ├── src/evaluation.py
│   └── src/rag.py
│
└── inference.py
    ├── src/inference.py
    │   └── transformers
    └── src/rag.py
        └── faiss

src/training.py
└── src/models.py

src/evaluation.py
├── sklearn
├── matplotlib
└── seaborn
```

---

## 🔍 Key Components by Function

### Data Processing
- `src/data.py` - Load, label, balance, split data

### Model Setup
- `src/models.py` - Configure BERT and Qwen with LoRA
- `src/training.py` - Tokenization and trainer setup

### Training
- `train.py` - Orchestrate full pipeline
- `src/training.py` - Training configuration and utilities

### Inference
- `inference.py` - Demo and interactive modes
- `src/inference.py` - Single-stage inference
- `src/rag.py` - RAG-enhanced inference

### Retrieval
- `src/rag.py` - Crisis resource database and retrieval

### Evaluation
- `src/evaluation.py` - Metrics and reporting

---

## 📝 Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Inference Demo
```bash
python inference.py --mode demo \
    --classifier_path path/to/classifier \
    --generator_path path/to/generator \
    --rag_path path/to/rag
```

### Interactive Chat
```bash
python inference.py --mode interactive \
    --classifier_path path/to/classifier \
    --generator_path path/to/generator \
    --rag_path path/to/rag
```

---

## 📖 Documentation Reading Order

1. **Start**: `QUICKSTART.py` - Get oriented (5 min read)
2. **Architecture**: `PROJECT_STRUCTURE.md` - Understand design (10 min read)
3. **Details**: `README_STRUCTURE.md` - Learn features (20 min read)
4. **Examples**: `EXAMPLES.py` - See code patterns (15 min read)
5. **Deep Dive**: Individual `src/*.py` files - Study implementation (varies)

---

## 🎓 Learning Path

### Beginner
- Read `QUICKSTART.py`
- Run `python train.py` on sample data
- Try `python inference.py --mode demo`

### Intermediate
- Study `EXAMPLES.py` (all 10 examples)
- Modify hyperparameters in `src/training.py`
- Experiment with custom data

### Advanced
- Review `src/` module implementations
- Understand LoRA fine-tuning details
- Implement custom metrics or loss functions
- Deploy with Flask/FastAPI

---

## 🚀 Quick Start Template

```python
# 1. Install
pip install -r requirements.txt

# 2. Train
python train.py

# 3. Infer
python inference.py --mode interactive \
    --classifier_path ./models/classifier \
    --generator_path ./models/generator \
    --rag_path ./models/rag
```

---

## 📞 Support Resources

### In Project
- `QUICKSTART.py` - Troubleshooting section
- `README_STRUCTURE.md` - Limitations section
- `EXAMPLES.py` - Usage patterns

### External
- Transformers: https://huggingface.co/docs/transformers/
- PEFT: https://huggingface.co/docs/peft/
- FAISS: https://github.com/facebookresearch/faiss

---

## ✅ Checklist for New Users

- [ ] Read `QUICKSTART.py`
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Set up Kaggle credentials (if downloading data)
- [ ] Run `python train.py` (or use pre-trained models)
- [ ] Try inference: `python inference.py --mode demo`
- [ ] Explore `EXAMPLES.py` for your use case
- [ ] Review `PROJECT_STRUCTURE.md` for deeper understanding

---

## 📂 File Organization Summary

```
Root (Entry points & Config)
├── train.py                 # Training entry point
├── inference.py             # Inference entry point
├── requirements.txt         # Dependencies
└── QUICKSTART.py           # Quick reference

Documentation
├── README.md               # Original docs
├── README_STRUCTURE.md     # Comprehensive guide
├── PROJECT_STRUCTURE.md    # Architecture
├── EXAMPLES.py             # 10 code examples
└── (This file)             # Index

Source Code (src/)
├── __init__.py             # Package init
├── data.py                 # Data utilities
├── models.py               # Model setup
├── training.py             # Training utilities
├── inference.py            # Inference utilities
├── rag.py                  # RAG system
└── evaluation.py           # Evaluation tools
```

---

**Last Updated**: 2025-12-06  
**Total Project Files**: 21  
**Total Lines of Code**: 1,900+  
**Total Documentation**: 1,200+  

For questions or issues, refer to the troubleshooting section in `QUICKSTART.py` or `README_STRUCTURE.md`.
