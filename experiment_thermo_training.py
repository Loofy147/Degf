import torch
import torch.optim as optim
import numpy as np
from transformer_lens import HookedTransformer
from train_thermo import compute_thermo_loss
from monitor_gpt2 import scan_model

def run_thermo_training_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "In a hole in the ground there lived a hobbit.",
        "All that glitters is not gold."
    ]
    tokens = model.to_tokens(corpus).to(device)

    # Baseline scan
    print("Capturing Baseline...")
    base_profiles = scan_model(model, corpus)
    base_q2_count = len([p for p in base_profiles if p.G >= 0.5])
    base_mean_G = np.mean([p.G for p in base_profiles])

    # Training
    steps = 50
    print(f"Running L_thermo Training ({steps} steps)...")
    history = []

    for i in range(steps):
        optimizer.zero_grad()
        loss, ce, reward = compute_thermo_loss(model, tokens)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"  Step {i:2d}: Loss={loss.item():.4f} | CE={ce.item():.4f} | G-Reward={reward.item():.4f}")
        history.append(reward.item())

    # Post-training scan
    print("Capturing Trained state...")
    post_profiles = scan_model(model, corpus)
    post_q2_count = len([p for p in post_profiles if p.G >= 0.5])
    post_mean_G = np.mean([p.G for p in post_profiles])

    print("\n--- Training Results ---")
    print(f"Q2 Head Count: {base_q2_count} -> {post_q2_count} ({post_q2_count - base_q2_count:+d})")
    print(f"Mean G Score:  {base_mean_G:.4f} -> {post_mean_G:.4f} ({post_mean_G - base_mean_G:+.4f})")

    # Check if delta G corresponds to history
    reward_lift = history[-1] - history[0]
    print(f"Final Reward Lift: {reward_lift:+.4f}")

if __name__ == "__main__":
    run_thermo_training_experiment()
