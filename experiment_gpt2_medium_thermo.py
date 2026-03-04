import torch
import torch.optim as optim
import numpy as np
from transformer_lens import HookedTransformer
from train_thermo import compute_thermo_loss
from monitor_gpt2 import scan_model

def run_medium_thermo():
    device = "cpu"
    print("Loading GPT-2-medium for L_thermo training (Moderate LR)...")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)

    # Reasoning layers 16-23
    target_layers = list(range(16, 24))
    for param in model.parameters():
        param.requires_grad = False

    params_to_train = []
    for l in target_layers:
        for name, param in model.blocks[l].attn.named_parameters():
            param.requires_grad = True
            params_to_train.append(param)

    # Adam with moderate LR
    optimizer = optim.Adam(params_to_train, lr=1e-3)

    texts = ["If all dogs are animals and Rex is a dog, then Rex is an animal."]
    tokens = model.to_tokens(texts).to(device)

    # Baseline
    print("Baseline Scan...")
    base_profiles = scan_model(model, texts)
    base_q2 = len([p for p in base_profiles if p.G >= 0.5])

    # Training
    steps = 10
    print(f"L_thermo training for {steps} steps...")
    for i in range(steps):
        optimizer.zero_grad()
        loss, ce, reward = compute_thermo_loss(model, tokens)
        loss.backward()
        optimizer.step()
        if i % 2 == 0:
            print(f"  Step {i:2d}: G-Reward={reward.item():.3f}")

    # Post-training
    print("Post-training Scan...")
    post_profiles = scan_model(model, texts)
    post_q2 = len([p for p in post_profiles if p.G >= 0.5])

    print(f"\nQ2 Head Lift: {base_q2} -> {post_q2} ({post_q2 - base_q2:+d})")

if __name__ == "__main__":
    run_medium_thermo()
