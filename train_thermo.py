import torch
import torch.nn as nn
import torch.optim as optim
from transformer_lens import HookedTransformer
from degf_core import K_DEG, K_REC, LAMBDA, GAMMA, THETA_C

def soft_collapse_count(H, theta=THETA_C, tau=0.01):
    """Differentiable approximation of collapse count C."""
    delta_H = H[:, :, 1:] - H[:, :, :-1]
    return torch.sigmoid((theta - delta_H) / tau).sum(dim=-1)

def compute_V_detrended_torch(H, burn_in=10):
    """Differentiable Detrended Entropy Variance in PyTorch."""
    T = H.shape[-1]
    if T < burn_in + 2:
        return H.var(dim=-1)
    t_idx = torch.arange(T, device=H.device, dtype=H.dtype)
    expected = torch.log2(t_idx + 1.0)
    detrended = H - expected
    return detrended[:, :, burn_in:].var(dim=-1)

def compute_thermo_loss(model, tokens, lambda_val=LAMBDA, gamma_val=GAMMA):
    # Run with hooks to get attention patterns
    target_layers = list(range(7, model.cfg.n_layers))

    patterns = {}
    def hook_fn(attn, hook):
        patterns[hook.layer()] = attn
        return attn

    logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{l}.attn.hook_pattern", hook_fn) for l in target_layers]
    )

    # Standard Cross-Entropy Loss
    labels = tokens[:, 1:]
    logits_shifted = logits[:, :-1, :]
    ce_loss = nn.functional.cross_entropy(logits_shifted.reshape(-1, logits.size(-1)), labels.reshape(-1))

    # Thermodynamic Reward
    thermo_reward = 0.0
    for l in target_layers:
        attn = patterns[l]
        # H: (batch, head, query)
        H = -(attn * torch.log2(attn + 1e-12)).sum(dim=-1)

        # FIX-6: Use detrended V for early layers and implement collapse gate
        use_det = (l < int(0.65 * model.cfg.n_layers))
        V = compute_V_detrended_torch(H) if use_det else H.var(dim=-1)
        C_soft = soft_collapse_count(H) # (batch, head)

        # FIX-6: Collapse gate (C>0 required for V reward)
        reward_per_head = (V + gamma_val * C_soft) * (C_soft > 0.5).float()
        thermo_reward += reward_per_head.mean()

    total_loss = ce_loss - lambda_val * thermo_reward
    return total_loss, ce_loss, thermo_reward

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times."
    ]
    tokens = model.to_tokens(texts).to(device)

    print("Starting L_thermo fine-tuning demo...")
    for i in range(10):
        optimizer.zero_grad()
        loss, ce, reward = compute_thermo_loss(model, tokens)
        loss.backward()
        optimizer.step()

        if i % 2 == 0:
            print(f"Step {i}: Loss={loss.item():.4f}, CE={ce.item():.4f}, Reward={reward.item():.4f}")

    print("Fine-tuning complete.")
