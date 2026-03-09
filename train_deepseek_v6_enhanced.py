import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import gc

LAMBDA = 0.08
GAMMA = 0.35
THETA_C = -0.20

def soft_collapse_count(H, theta=THETA_C, tau=0.01):
    delta_H = H[:, :, 1:] - H[:, :, :-1]
    return torch.sigmoid((theta - delta_H) / tau).sum(dim=-1)

def compute_V_detrended_torch(H, burn_in=5):
    T = H.shape[-1]
    if T < burn_in + 2:
        return H.var(dim=-1)
    t_idx = torch.arange(T, device=H.device, dtype=H.dtype)
    expected = torch.log2(t_idx + 1.0)
    detrended = H - expected
    return detrended[:, :, burn_in:].var(dim=-1)

def compute_thermo_loss_hf(model, inputs):
    outputs = model(**inputs, output_attentions=True)
    logits = outputs.logits
    attentions = outputs.attentions

    labels = inputs["input_ids"][:, 1:]
    logits_shifted = logits[:, :-1, :]
    ce_loss = nn.functional.cross_entropy(logits_shifted.reshape(-1, logits.size(-1)), labels.reshape(-1))

    thermo_reward = 0.0
    n_layers = len(attentions)
    target_layers = list(range(18, n_layers))

    for l in target_layers:
        attn = attentions[l].float()
        attn_safe = torch.clamp(attn, min=1e-8)
        attn_safe = attn_safe / attn_safe.sum(dim=-1, keepdim=True)
        H = -(attn_safe * torch.log2(attn_safe)).sum(dim=-1)

        use_det = (l < int(0.65 * n_layers))
        V = compute_V_detrended_torch(H) if use_det else H.var(dim=-1)
        C_soft = soft_collapse_count(H)
        reward_per_head = (V + GAMMA * C_soft) * (C_soft > 0.4).float()
        thermo_reward += reward_per_head.mean()

    total_loss = ce_loss - LAMBDA * thermo_reward
    return total_loss, ce_loss, thermo_reward

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'deepseek-ai/deepseek-coder-1.3b-instruct'
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).to(device)

    for param in model.parameters():
        param.requires_grad = False

    for i in [21, 22, 23]:
        target_layer = model.model.layers[i]
        for param in target_layer.self_attn.parameters():
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    prompts = [
        "Given the Titanic data features like 'Sex', 'Age', and 'Pclass', predict survival using a logical decision tree.",
        "To predict house prices, we must consider the square footage, neighborhood, and quality of the building.",
        "Space travelers on the Spaceship Titanic might be transported if they are in cryosleep or in certain cabins.",
        "Socrates is mortal because all men are mortal and Socrates is a man.",
        "If it rains, the ground is wet. The ground is not wet. Therefore, it did not rain.",
        "A script to load 'train.csv', fill missing values in 'Age' with the median, and train a classifier."
    ]

    print("Training Enhanced DEGF...")
    for step in range(10):
        prompt = prompts[step % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        optimizer.zero_grad()
        loss, ce, reward = compute_thermo_loss_hf(model, inputs)
        loss.backward()
        optimizer.step()

        print(f"  Step {step:2d}: CE={ce.item():.4f}, Reward={reward.item():.4f}", flush=True)

    print("Saving Enhanced weights...")
    delta = {k: v.cpu() for k, v in model.state_dict().items() if v.requires_grad}
    torch.save(delta, "deepseek_thermo_delta_enhanced.pt")
    print("Done.")

if __name__ == "__main__":
    main()
