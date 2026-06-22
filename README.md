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
├── fine-tune.py           # Fine-tuning script (trains + merges LoRA automatically)
├── test-fine-tune.py      # Model testing script with real-time feedback
├── requirements.txt       # Python dependencies
├── .env_example          # Environment variables template
├── .gitignore            # Git ignore patterns
├── Modelfile             # Ollama model configuration
├── ollama_create.sh      # Script to create Ollama model
├── dataset/
│   ├── nist-rmf.jsonl    # Training dataset (NIST AI RMF Q&A)
│   └── test-questions.jsonl  # Test dataset (100 questions for evaluation)
└── models/               # Output directory for fine-tuned models
    ├── gemma-3-270m-nist/         # LoRA adapter weights (saved by trainer)
    └── gemma-3-270m-nist-merged/  # Merged full model (ready for GGUF conversion)
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
python3.12 -m venv .venv
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
- `python-dotenv` - Environment variable management
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
- Max sequence length: 1024 (safe for M1)
- Epochs: 3
- Batch size: 1
- Gradient accumulation: 8
- Learning rate: 2e-4
- LoRA: enabled (r=16, alpha=32) — trains ~1% of parameters

#### Custom Configuration

```bash
python fine-tune.py \
  --model google/gemma-3-270m-it \
  --dataset dataset/nist-rmf.jsonl \
  --output-dir models/gemma-3-270m-nist \
  --max-seq-length 1024 \
  --epochs 5 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32
```

To disable LoRA and do full fine-tuning (slower, higher memory):
```bash
python fine-tune.py --no-lora --learning-rate 5e-5
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
| `--max-seq-length` | `1024` | Maximum sequence length in tokens |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `1` | Training batch size per device |
| `--grad-accum` | `8` | Gradient accumulation steps |
| `--learning-rate` | `2e-4` | Learning rate |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA scaling factor |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--no-lora` | `False` | Disable LoRA, use full fine-tuning |
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
1. **Training output directory** (`models/gemma-3-270m-nist/`): Contains LoRA adapter weights only (`adapter_model.safetensors`, `adapter_config.json`) — **not** full model weights
2. **Merged model** (created in step 1 below): Full model with LoRA weights merged in, ready for GGUF conversion
3. **GGUF file** (created in step 2): Quantized model format for efficient inference with Ollama
4. **Ollama model** (registered via `ollama create`): The GGUF file packaged with a Modelfile for easy deployment

> **Why merge?** LoRA training saves only the small adapter delta, not the full model weights. `llama.cpp` requires a complete model to convert to GGUF. The merge step combines the adapter with the original base model.

### 1. Merged Model (automatic)

When LoRA is enabled (default), `fine-tune.py` automatically merges the adapter into the base model at the end of training and saves the result to `models/gemma-3-270m-nist-merged/`. No extra step needed.

> If you trained with `--no-lora`, the output directory already contains full weights — use it directly in the next step.

### 2. Convert Model to GGUF Format

Convert the merged model to GGUF for Ollama-compatible inference using `llama.cpp`'s conversion script:

```bash
# Clone llama.cpp (if you haven't already)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install Python dependencies for conversion
pip install -r requirements.txt

# Convert the MERGED model to GGUF (not the adapter directory)
python convert_hf_to_gguf.py \
  ../models/gemma-3-270m-nist-merged \
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

### 5. Test the Fine-Tuned Model

After deploying your model with Ollama, you can test its performance using the included test script.

#### Test Dataset

The project includes a comprehensive test dataset with 100 questions:
- **Location:** `dataset/test-questions.jsonl`
- **Format:** JSONL with fields: `id`, `kind`, `expected`, `prompt`
- **Question types:**
  - 55 True/False questions
  - 45 Multiple choice questions (A, B, C, or D)
- **Content:** All questions based on NIST AI RMF concepts

**Example test questions:**
```jsonl
{"id": "tf_1", "kind": "true_false", "expected": "TRUE", "prompt": "Answer with exactly one word: TRUE or FALSE. No explanation.\n\nThe NIST AI RMF emphasizes managing security and privacy risk as an ongoing process throughout the system lifecycle.\n"}
{"id": "mcq_1", "kind": "multiple_choice", "expected": "D", "prompt": "Choose one option and answer with exactly one letter: A, B, C, or D. No explanation.\n\nWhich RMF function is primarily responsible for monitoring deployed AI systems?\nA. GOVERN\nB. MAP\nC. MEASURE\nD. MANAGE\n"}
```

#### Running Tests

Test your fine-tuned model with real-time feedback:

```bash
# Basic usage (uses defaults)
python test-fine-tune.py

# Custom configuration
python test-fine-tune.py \
  --model gemma3-270m-nist \
  --questions dataset/test-nist-rmf.jsonl \
  --ollama-url http://localhost:11434/api/generate \
  --timeout 120
```

**Available arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemma-3-270m-nist` | Model name to test |
| `--questions` | `dataset/test-nist-rmf.jsonl` | Path to test questions file |
| `--ollama-url` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `--timeout` | `120` | Request timeout in seconds |
| `--output` | `ollama_inference_test_results.json` | Output file for results |

#### Test Output

The script provides real-time feedback as it runs:

```
Loaded 100 test questions from dataset/test-questions.jsonl
Testing model: gemma3-270m-nist
============================================================

[1/100] Testing tf_1... ✓ CORRECT [tf_1] Expected: TRUE, Got: TRUE (1.23s)
[2/100] Testing tf_2... ✗ INCORRECT [tf_2] Expected: FALSE, Got: TRUE (0.98s)
[3/100] Testing mcq_1... ✓ CORRECT [mcq_1] Expected: D, Got: D (1.45s)
...

============================================================

Test Results:
Total: 100
Passed: 87
Failed: 13
Accuracy: 87.0%

Results saved to: ollama_inference_test_results.json
```

#### Interpreting Results

The test results JSON file contains:
```json
{
  "model": "gemma3-270m-nist",
  "ollama_url": "http://localhost:11434/api/generate",
  "questions_file": "dataset/test-questions.jsonl",
  "total": 100,
  "passed": 87,
  "failed": 13,
  "accuracy": 0.87,
  "results": [
    {
      "id": "tf_1",
      "kind": "true_false",
      "expected": "TRUE",
      "raw_response": "TRUE",
      "normalized_response": "TRUE",
      "passed": true,
      "latency_seconds": 1.234,
      "error": null
    }
  ]
}
```

Use this data to:
- Compare performance between original and fine tuned models
- Identify question types where the model struggles
- Track improvements after additional fine-tuning
- Analyze response times and optimize inference

### 6. Use via Ollama API

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
- **LoRA:** Enabled by default — trains only ~1% of parameters, dramatically reducing memory and training time on M1
- **Memory Management:** If you encounter out-of-memory errors, reduce `--max-seq-length` to 512
- **Batch Size:** Keep `--batch-size 1` for M1 Macs; use gradient accumulation (`--grad-accum 8`) to simulate larger effective batch sizes
- **Unsloth:** This project does not use Unsloth, as it's not compatible with Apple Silicon
- **Performance:** Training on M1 is slower than CUDA GPUs but works reliably with LoRA + reduced sequence length

### Out of Memory (OOM) Errors

```bash
# LoRA is already enabled by default — try reducing sequence length first
python fine-tune.py --max-seq-length 512

# Ensure batch size is 1 (default)
python fine-tune.py --batch-size 1 --max-seq-length 512

# Lower LoRA rank to further reduce memory
python fine-tune.py --lora-r 8 --lora-alpha 16
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