# Model Checkpoints

This directory contains the finetuned model checkpoint used in experiments D, E, and F.

## Checkpoint Information

- **Checkpoint**: `checkpoint-14740`
- **Base Model**: Qwen3-1.7B
- **Training Dataset**: TIGER-Lab/MathInstruct
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Purpose**: "Bad SFT" adapter to corrupt model's commonsense reasoning prior

## Checkpoint Contents

- `adapter_model.safetensors` - LoRA adapter weights
- `adapter_config.json` - LoRA adapter configuration
- `trainer_state.json` - Training state and metrics
- `training_args.bin` - Training arguments
- Tokenizer files (tokenizer.json, vocab.json, etc.)

## Usage

To use this checkpoint in experiments, load it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "checkpoints/checkpoint-14740"
)
```

## Training Parameters

See `results/training_parameters.md` for complete training configuration details.

## Size

- Total checkpoint size: ~116 MB
- This checkpoint is sufficient to replicate all experiments D, E, and F.

