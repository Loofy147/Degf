import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from monitor_gpt2 import HFDEGFMonitor

def test_hf_monitor():
    model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
    print(f"Testing HFDEGFMonitor on {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        monitor = HFDEGFMonitor(model, tokenizer)
        prompt = "1, 2, 3,"
        g_stream = monitor.monitor_step(prompt)

        print(f"\n{'Token':<12} | {'G':<6} | {'tc':<6} | {'Risk':<5}")
        print("-" * 40)
        for entry in g_stream:
            print(f"{entry['token']:<12} | {entry['G']:.3f} | {entry['tc']:.3f} | {entry['hallucination_risk']}")

        assert len(g_stream) > 0
        print("\nTest passed!")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_hf_monitor()
