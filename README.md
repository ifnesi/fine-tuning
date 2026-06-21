# LLM Fine-Tuning for NIST AI Risk Management Framework

A comprehensive project for fine-tuning Large Language Models (LLMs) on the NIST AI Risk Management Framework (AI RMF) dataset. This project demonstrates how to create domain-specific AI assistants using supervised fine-tuning and deploy them locally using Ollama.

## 📋 Table of Contents

- [Overview](#overview)
- [Fine-Tuning vs RAG](#fine-tuning-vs-rag)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Deployment](#model-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project fine-tunes the **Gemma 3 270M** model on [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) data to create a specialized assistant that can:
- Answer questions about the NIST AI Risk Management Framework
- Classify AI RMF functions (MAP, GOVERN, MEASURE, MANAGE)
- Explain trustworthiness characteristics
- Provide guidance on risk management practices

## 🤔 Fine-Tuning vs RAG

### What is Fine-Tuning?

**Fine-tuning** involves training a pre-trained model on domain-specific data to adapt its weights and behavior. The model learns to internalize knowledge and patterns from your dataset.

**Pros:**
- ✅ Knowledge is embedded in model weights
- ✅ Faster inference (no retrieval overhead)
- ✅ Better at learning style, tone, and format
- ✅ Works offline without external databases
- ✅ Consistent responses for similar queries

**Cons:**
- ❌ Requires computational resources for training
- ❌ Knowledge becomes stale (requires retraining to update)
- ❌ Can be expensive for large models
- ❌ Risk of overfitting on small datasets
- ❌ Harder to trace source of information

### What is RAG (Retrieval-Augmented Generation)?

**RAG** retrieves relevant documents from a knowledge base and provides them as context to the model during inference.

**Pros:**
- ✅ Easy to update knowledge (just update the database)
- ✅ Can cite sources and provide references
- ✅ Works with any base model (no training needed)
- ✅ Scales to large knowledge bases
- ✅ More transparent and explainable

**Cons:**
- ❌ Slower inference (retrieval + generation)
- ❌ Requires vector database infrastructure
- ❌ Quality depends on retrieval accuracy
- ❌ Context window limitations
- ❌ May struggle with reasoning across documents

### When to Use Each Approach

| Scenario | Recommendation |
|----------|---------------|
| **Frequently changing information** | RAG |
| **Need source citations** | RAG |
| **Large knowledge base (>1M tokens)** | RAG |
| **Learning specific style/format** | Fine-Tuning |
| **Offline deployment** | Fine-Tuning |
| **Fast inference required** | Fine-Tuning |
| **Small, stable dataset** | Fine-Tuning |
| **Teaching reasoning patterns** | Fine-Tuning |

### Hybrid Approach: Fine-Tuning + RAG

For best results, combine both:
1. **Fine-tune** the model to understand domain terminology, style, and reasoning patterns
2. **Use RAG** to provide up-to-date facts and citations

This project focuses on **fine-tuning**, but the resulting model can be integrated with RAG systems.

## 📁 Project Structure

```
.
├── fine-tune.py           # Main fine-tuning script with argparse
├── requirements.txt       # Python dependencies
├── .env_example          # Environment variables template
├── .gitignore            # Git ignore patterns
├── Modelfile             # Ollama model configuration
├── ollama_create.sh      # Script to create Ollama model
├── dataset/
│   └── nist-rmf.jsonl    # Training dataset (NIST AI RMF Q&A)
└── models/               # Output directory for fine-tuned models
```

## 🔧 Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended) or CPU
- **Hugging Face account** with API token
- **Ollama** (for local model deployment)
- **8GB+ RAM** (16GB+ recommended)

## 🚀 Setup

### 1. Clone the Repository

```bash
git clone git@github.com:ifnesi/fine-tuning.git
cd fine-tuning
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `accelerate` - Distributed training support
- `bitsandbytes` - Quantization and optimization
- `datasets` - Dataset loading and processing
- `dotenv` - Environment variable management
- `peft` - Parameter-Efficient Fine-Tuning (LoRA)
- `transformers` - Hugging Face model library
- `trl` - Transformer Reinforcement Learning (SFT)

### 4. Configure Environment Variables

```bash
cp .env_example .env
```

Edit `.env` and add your Hugging Face token:

```bash
HF_TOKEN="your_huggingface_token_here"
```

**Get your token:** https://huggingface.co/settings/tokens

### 5. Install Ollama (Optional - for local deployment)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download from https://ollama.com/download

## 💻 Usage

### Fine-Tuning the Model

#### Basic Usage (with defaults)

```bash
python fine-tune.py
```

This uses default settings:
- Model: `google/gemma-3-270m-it`
- Dataset: `dataset/nist-rmf.jsonl`
- Epochs: 3
- Batch size: 4
- Learning rate: 5e-5

#### Custom Configuration

```bash
python fine-tune.py \
  --model google/gemma-3-270m-it \
  --dataset dataset/nist-rmf.jsonl \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --output-dir models/gemma-3-270m-nist
```

#### View All Options

```bash
python fine-tune.py --help
```

**Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `google/gemma-3-270m-it` | Hugging Face model name |
| `--dataset` | `dataset/nist-rmf.jsonl` | Path to training dataset |
| `--output-dir` | `models/<model-name>` | Output directory |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Training batch size per device |
| `--learning-rate` | `5e-5` | Learning rate |
| `--max-length` | `2048` | Maximum sequence length |
| `--logging-steps` | `10` | Steps between logging |
| `--save-strategy` | `epoch` | Checkpoint save strategy |
| `--attn-implementation` | `eager` | Attention implementation |
| `--hf-token` | (from .env) | Hugging Face API token |
| `--device` | `auto` | Device for training |

### Training Output

The script will:
1. Load the model and tokenizer from Hugging Face
2. Process the dataset with chat templates
3. Train the model with progress logging
4. Save the fine-tuned model to `models/` directory
5. Generate a training log file: `fine-tune.log`

**Example output:**
```
================================================================================
Starting Fine-Tuning Process
================================================================================
Model: google/gemma-3-270m-it
Dataset: dataset/nist-rmf.jsonl
Epochs: 3
Batch Size: 4
Learning Rate: 5e-05
================================================================================
Loading tokenizer: google/gemma-3-270m-it
Loading model: google/gemma-3-270m-it
Loading dataset from: dataset/nist-rmf.jsonl
Dataset loaded with 150 examples
Formatting dataset examples
Training output directory: ./models/gemma-3-270m-nist
Initializing SFT Trainer
Starting training...
...
Fine-tuning completed successfully!
Model saved to: ./models/gemma-3-270m-nist
================================================================================
```

## 📊 Dataset

The project uses a JSONL dataset with NIST AI RMF questions and answers.

**Format:**
```jsonl
{"prompt": "Question text", "response": "Answer text"}
```

**Example:**
```jsonl
{"prompt": "Which function includes mitigation, transfer, avoidance, or acceptance of risk?", "response": "MANAGE"}
{"prompt": "List a trustworthiness characteristic from the AI RMF.", "response": "Valid and reliable."}
```

**Dataset location:** `dataset/nist-rmf.jsonl`

### Creating Your Own Dataset

1. Create a JSONL file with `prompt` and `response` fields
2. Place it in the `dataset/` directory
3. Run fine-tuning with `--dataset path/to/your/dataset.jsonl`

## 🚢 Model Deployment with Ollama

After fine-tuning, deploy your model locally using Ollama.

### 1. Convert Model to GGUF Format

First, convert your fine-tuned model to GGUF format (required for Ollama):

```bash
# Convert model (adjust paths as needed)
python -m llama_cpp.convert \
  --model-dir ./models/google/gemma-3-270m-nist \
  --output-file ./models/gemma-3-270m-nist/gemma3-270m-nist.gguf
```

### 2. Configure Modelfile

The `Modelfile` in the project root contains the Ollama model configuration. Update the path to your converted GGUF model if needed. The file includes:
- Model path reference
- System prompt for NIST AI RMF specialization
- Chat template configuration
- Model parameters (temperature, context length, stop tokens)

### 3. Create Ollama Model

Run the provided script to create the Ollama model:

```bash
./ollama_create.sh
```

This script executes: `ollama create gemma3-270m-nist -f Modelfile`

### 4. Run the Model

```bash
ollama run gemma3-270m-nist
```

**Example interaction:**
```
>>> What are the four functions of the NIST AI RMF?
The four functions of the NIST AI Risk Management Framework are:
1. MAP - Understanding the AI system context
2. MEASURE - Assessing AI risks and impacts
3. MANAGE - Prioritizing and responding to risks
4. GOVERN - Cultivating organizational culture and oversight

>>> Which function includes risk mitigation?
MANAGE - This function includes mitigation, transfer, avoidance, or acceptance of risk.
```

### 5. Use via API

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3-270m-nist",
  "prompt": "What is the GOVERN function?",
  "stream": false
}'
```

## ⚙️ Configuration

### Environment Variables

See `.env_example` for all available environment variable:
- `HF_TOKEN` - Hugging Face API token (required)

### Training Configuration

Modify training parameters via command-line arguments or edit `fine-tune.py` defaults.

**Key parameters to tune:**
- **Learning rate:** Start with 5e-5, adjust based on loss curves
- **Epochs:** 3-5 for small datasets, 1-2 for large datasets
- **Batch size:** Increase if you have more GPU memory
- **Max length:** Match your dataset's typical sequence length

## 🔍 Troubleshooting

### Out of Memory (OOM) Errors

```bash
# Reduce batch size
python fine-tune.py --batch-size 2

# Reduce max length
python fine-tune.py --max-length 256
```

### Slow Training

```bash
# Use mixed precision (automatic on CUDA)
# Increase batch size if memory allows
python fine-tune.py --batch-size 8

# Use gradient accumulation (edit script)
```

### Model Not Loading

- Verify your HF_TOKEN is correct
- Check internet connection
- Ensure you have access to the model (some models are gated)

### Ollama Model Not Working

- Verify GGUF conversion was successful
- Check Modelfile paths are correct
- Ensure Ollama service is running: `ollama serve`

## 📚 Additional Resources

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Fine-Tuning Best Practices](https://huggingface.co/docs/transformers/training)

## 📝 License

This project is for educational and research purposes. Please ensure compliance with:
- Model licenses (Gemma license)
- NIST AI RMF usage terms
- Your organization's policies

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

---

**Happy Fine-Tuning! 🚀**