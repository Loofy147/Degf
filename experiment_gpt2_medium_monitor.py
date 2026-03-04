import torch
import numpy as np
from transformer_lens import HookedTransformer
from monitor_gpt2 import DEGFMonitor

def run_medium_monitor():
    device = "cpu"
    print("Loading GPT-2-medium for Monitor verification...")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)
    monitor = DEGFMonitor(model)

    prompt = "If all dogs are animals and Rex is a dog, then Rex is"
    print(f"\nPrompt: {prompt}")

    g_stream = monitor.monitor_step(prompt)

    print(f"\n{'Token':<12} | {'G':<6} | {'tc':<6} | {'Risk':<5}")
    print("-" * 40)
    for entry in g_stream:
        print(f"{entry['token']:<12} | {entry['G']:.3f} | {entry['tc']:.3f} | {entry['hallucination_risk']}")

    # Check for G stability (variance) across tokens
    g_values = [e['G'] for e in g_stream]
    print(f"\nG-Stream: mean={np.mean(g_values):.3f}, std={np.std(g_values):.3f}")

if __name__ == "__main__":
    run_medium_monitor()
