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
| **Fast inference (short prompts, no retrieval)** | Fine-Tuning |
| **Small, stable dataset** | Fine-Tuning |
| **Teaching reasoning patterns** | Fine-Tuning |

### Hybrid Approach: Fine-Tuning + RAG

For best results, combine both:
1. **Fine-tune** the model to understand domain terminology, style, and reasoning patterns
2. **Use RAG** to provide up-to-date facts and citations

This project focuses on **fine-tuning** to create a specialized model. RAG is optional and separate from the fine-tuning process—the trained model can be used standalone or integrated with RAG systems for enhanced capabilities.

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

- **Python 3.12** (tested and recommended; Python 3.14 has known compatibility issues with the `datasets` library)
- **Apple Silicon (M1/M2/M3)** with reduced settings, **CUDA-capable GPU** (recommended), or **CPU**
  - Note: This project uses Transformers + TRL, which works on Apple Silicon with MPS backend but requires reduced `--max-seq-length` (1024 or 512) and `--batch-size 1`
  - Unsloth is not compatible with MacBook Pro M1 and is not used in this implementation
  - CUDA GPUs provide the best training performance
- **Hugging Face account** with API token
- **Ollama** (for local model deployment)
- **16GB+ RAM** (recommended for M1 Macs; 8GB may work with further reduced settings)

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
- Output directory: `models/gemma-3-270m-nist`
- Max sequence length: 2048
- Epochs: 3
- Batch size: 1
- Gradient accumulation: 8
- Learning rate: 5e-5

#### Custom Configuration

```bash
python fine-tune.py \
  --model google/gemma-3-270m-it \
  --dataset dataset/nist-rmf.jsonl \
  --output-dir models/gemma-3-270m-nist \
  --max-seq-length 2048 \
  --epochs 5 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 1e-4
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
| `--output-dir` | `models/gemma-3-270m-nist` | Output directory |
| `--max-seq-length` | `2048` | Maximum sequence length in tokens |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `1` | Training batch size per device |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--learning-rate` | `5e-5` | Learning rate |
| `--logging-steps` | `10` | Steps between logging |
| `--save-strategy` | `epoch` | Checkpoint save strategy (`no`, `steps`, `epoch`) |
| `--save-steps` | `100` | Save checkpoint every N steps (if using `steps` strategy) |
| `--hf-token` | (from .env) | Hugging Face API token |

### Training Output

The script will:
1. Load the model and tokenizer from Hugging Face
2. Process the dataset with chat templates
3. Train the model with progress logging
4. Save the fine-tuned model to `models/` directory

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
{"prompt": "List a trustworthiness characteristic from the AI RMF.", "response": "Valid and reliable. The AI RMF emphasizes that AI systems should produce consistent, accurate results and perform reliably across different conditions."}
```

**Dataset location:** `dataset/nist-rmf.jsonl`

### Creating Your Own Dataset

1. Create a JSONL file with `prompt` and `response` fields
2. Place it in the `dataset/` directory
3. Run fine-tuning with `--dataset path/to/your/dataset.jsonl`

## 🚢 Model Deployment with Ollama

After fine-tuning, convert your model to GGUF format and deploy it locally using Ollama. The stack is simple: **GGUF + Modelfile + Ollama**.

### What Files Are Produced After Training?

After training completes, you'll have:
1. **Training output directory** (`models/gemma-3-270m-nist/`): Contains the fine-tuned model in Hugging Face format (PyTorch weights, tokenizer, config)
2. **GGUF file** (created in conversion step): Quantized model format for efficient inference with Ollama
3. **Ollama model** (registered via `ollama create`): The GGUF file packaged with a Modelfile for easy deployment

### 1. Convert Model to GGUF Format

Convert the trained model to GGUF for Ollama-compatible inference using `llama.cpp`'s conversion script:

```bash
# Clone llama.cpp (if you haven't already)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install Python dependencies for conversion
pip install -r requirements.txt

# Convert your fine-tuned model to GGUF
python convert-hf-to-gguf.py \
  ../models/gemma-3-270m-nist \
  --outfile ../models/gemma-3-270m-nist/gemma3-270m-nist.gguf \
  --outtype q8_0

cd ..
```

**Quantization options:**
- `q8_0`: 8-bit quantization (recommended: good balance of size and quality)
- `q4_0`: 4-bit quantization (smaller, faster, slight quality loss)
- `q4_k_m`: 4-bit with k-quants (better quality than q4_0)
- `f16`: 16-bit float (larger, highest quality)

**Resources:**
- [llama.cpp GGUF Conversion Guide](https://github.com/ggerganov/llama.cpp/discussions/2948)
- [Hugging Face to GGUF Workflow](https://huggingface.co/docs/transformers/gguf)
- [Ollama Model Import Guide](https://github.com/ollama/ollama/blob/main/docs/import.md)

### 2. Configure Modelfile

The `Modelfile` in the project root tells Ollama how to package and run your GGUF model.

**Key components:**
- `FROM`: Path to your GGUF file (update if your path differs)
- `SYSTEM`: System prompt for NIST AI RMF specialization
- `TEMPLATE`: Chat template matching Gemma's format
- `PARAMETER num_ctx 2048`: Runtime context window (not a training parameter)

**Note:** Ollama handles everything—you don't need `llama_cpp` or Python bindings unless you want to call the model directly from Python outside the Ollama service.

### 3. Create Ollama Model

Package your GGUF file with the Modelfile:

```bash
# Create the Ollama model
ollama create gemma3-270m-nist -f Modelfile
```

This registers your model with Ollama, making it available for inference.

### 4. Run the Model

Start an interactive chat session:

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

### 5. Use via Ollama API

Query the model programmatically:

```bash
# Generate a response
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3-270m-nist",
  "prompt": "What is the GOVERN function?",
  "stream": false
}'

# Chat format
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3-270m-nist",
  "messages": [
    {"role": "user", "content": "Explain the MAP function"}
  ]
}'
```

**Python example (optional):**
```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'gemma3-270m-nist',
    'prompt': 'What are trustworthiness characteristics?',
    'stream': False
})
print(response.json()['response'])
```

**Note:** You only need `ollama-python` or similar libraries if you want Python bindings. The Ollama service itself handles all model serving.

## ⚙️ Configuration

### Environment Variables

See `.env_example` for all available environment variable:
- `HF_TOKEN` - Hugging Face API token (required)

### Training Configuration

Modify training parameters via command-line arguments or edit `fine-tune.py` defaults.

**Key parameters to tune:**
- **Learning rate:** Start with 5e-5, adjust based on loss curves
- **Epochs:** 3-5 for small datasets, 1-2 for large datasets
- **Batch size:** Keep at 1 for M1 Macs; increase to 4-8 if you have CUDA GPU with sufficient memory
- **Max sequence length:** Match your dataset's typical sequence length (reduce to 1024 or 512 on M1)
- **Gradient accumulation:** Use `--grad-accum 8` to simulate larger batch sizes without increasing memory usage

## 🔍 Troubleshooting

### MacBook Pro M1 Specific Notes

- **MPS Backend:** The script automatically detects and uses Apple's Metal Performance Shaders (MPS) for GPU acceleration on M1/M2/M3 Macs
- **Memory Management:** If you encounter out-of-memory errors, reduce `--max-seq-length` to 1024 or 512
- **Batch Size:** Keep `--batch-size 1` for M1 Macs; use gradient accumulation (`--grad-accum 8`) to simulate larger effective batch sizes
- **Unsloth:** This project does not use Unsloth, as it's not compatible with Apple Silicon
- **Performance:** Training on M1 is slower than CUDA GPUs but works reliably with reduced settings

### Out of Memory (OOM) Errors

```bash
# Reduce max sequence length (most important on M1)
python fine-tune.py --max-seq-length 1024

# Or even more aggressive for 8GB RAM
python fine-tune.py --max-seq-length 512

# Ensure batch size is 1 (default)
python fine-tune.py --batch-size 1 --max-seq-length 1024
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