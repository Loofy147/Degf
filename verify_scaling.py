import torch
from transformer_lens import HookedTransformer
from degf_v6 import run_hallucination_f1

def verify_scaling():
    for model_name in ["gpt2-small", "gpt2-medium"]:
        print(f"\n--- Verifying Scaling for {model_name} ---")
        device = "cuda" if model_name == "gpt2-small" and torch.cuda.is_available() else "cpu"
        model = HookedTransformer.from_pretrained(model_name, device=device)
        res = run_hallucination_f1(model)
        print(f"{model_name} Protocol F1: {res['f1']:.3f} | Recall: {res['recall']:.3f}")

if __name__ == "__main__":
    verify_scaling()
