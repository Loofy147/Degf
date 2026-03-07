import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sgs2_prototype import SGS2Prototype

def test_sgs2_hf():
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    print(f"Testing SGS2Prototype on {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        sgs2 = SGS2Prototype(model, tokenizer)
        prompt = "1, 2, 3,"
        result = sgs2.generate(prompt, max_new_tokens=5, verbose=False)
        print(f"\nFinal Result: {result}")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sgs2_hf()
