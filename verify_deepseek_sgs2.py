import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sgs2_prototype import SGS2Prototype

def verify_deepseek_sgs2():
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    print(f"Verifying SGS2 Generation on {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        sgs2 = SGS2Prototype(model, tokenizer)
        prompt = "1+1="
        result = sgs2.generate(prompt, max_new_tokens=2, max_loops=1, verbose=True)
        print(f"\nFinal Result:\n{result}")

    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    verify_deepseek_sgs2()
