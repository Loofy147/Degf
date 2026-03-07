import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_load():
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    print(f"Attempting to load {model_name} in bfloat16...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print("Model loaded successfully!")
        print(f"Model device: {model.device}")
    except Exception as e:
        print(f"Failed to load: {e}")

if __name__ == "__main__":
    test_load()
