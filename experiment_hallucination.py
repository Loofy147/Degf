import torch
import numpy as np
from transformer_lens import HookedTransformer
from monitor_gpt2 import DEGFMonitor

def run_hallucination_probe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    monitor = DEGFMonitor(model)

    # Pairs: (True Statement, Contradictory/Hallucinated Statement)
    cases = [
        ("Paris is the capital of France.", "Paris is the capital of Germany."),
        ("The sun rises in the east.", "The sun rises in the west."),
        ("Socrates was a philosopher.", "Socrates was a professional basketball player.")
    ]

    results = []

    print(f"{'Type':<12} | {'Token':<12} | {'G':<6} | {'tc':<6} | {'Risk':<5}")
    print("-" * 50)

    for fact, fake in cases:
        for text, label in [(fact, "FACT"), (fake, "FAKE")]:
            g_stream = monitor.monitor_step(text)
            # Focus on the 'hallucinated' part (usually the end)
            last_token = g_stream[-2] # usually period is last, so token before
            print(f"{label:<12} | {last_token['token']:<12} | {last_token['G']:.3f} | {last_token['tc']:.3f} | {last_token['hallucination_risk']}")
            results.append((label, last_token))

    # Verify signature: FAKE should have lower G relative to FACT, or specific Risk flags
    fake_results = [r for l, r in results if l == "FAKE"]
    fact_results = [r for l, r in results if l == "FACT"]

    mean_g_fact = np.mean([r['G'] for r in fact_results])
    mean_g_fake = np.mean([r['G'] for r in fake_results])

    print("\n--- Summary ---")
    print(f"Mean G (Fact): {mean_g_fact:.3f}")
    print(f"Mean G (Fake): {mean_g_fake:.3f}")
    print(f"Delta G: {mean_g_fake - mean_g_fact:.3f}")

if __name__ == "__main__":
    run_hallucination_probe()
