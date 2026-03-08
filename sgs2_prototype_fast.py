import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_H_torch(A: torch.Tensor) -> torch.Tensor:
    A_safe = torch.clamp(A, min=1e-12)
    return -(A * torch.log2(A_safe)).sum(dim=-1)

def compute_V_torch(H: torch.Tensor) -> torch.Tensor:
    if H.shape[-1] < 2: return torch.zeros_like(H[:, 0])
    return H.var(dim=-1)

def count_collapses_torch(H: torch.Tensor, theta: float = -0.20) -> torch.Tensor:
    if H.shape[-1] < 2: return torch.zeros_like(H[:, 0])
    diffs = H[:, 1:] - H[:, :-1]
    return (diffs < theta).sum(dim=-1).float()

def compute_G_torch(V: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-(V + 0.5 * C - 1.2)))

class SGS2PrototypeFast(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.reasoning_layers = list(range(0, self.n_layers - 3))
        self.synthesis_threshold = 0.01

    def get_latent_G(self, tokens):
        with torch.no_grad():
            outputs = self.model(tokens, output_attentions=True)
            attentions = outputs.attentions

            total_G = 0.0
            count = 0
            for l in self.reasoning_layers:
                attn = attentions[l][0] # (heads, seq, seq)
                H = compute_H_torch(attn) # (heads, seq)
                V = compute_V_torch(H) # (heads)
                C = count_collapses_torch(H) # (heads)
                G_heads = compute_G_torch(V, C)
                total_G += G_heads.mean().item()
                count += 1

            return total_G / count

    def generate(self, text, max_new_tokens=500, max_loops=2):
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.model.device)
        generated = tokens[0].tolist()

        print(f"Generating with SGS-2 Fast...")
        for i in range(max_new_tokens):
            # Simulation of reasoning loops
            prev_G = 0.0
            for loop in range(max_loops):
                outputs = self.model(tokens, output_attentions=True)
                curr_G = self.get_latent_G(tokens)
                if (curr_G - prev_G) < self.synthesis_threshold:
                    break
                prev_G = curr_G

            next_token = outputs.logits[0, -1].argmax().item()
            generated.append(next_token)
            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=tokens.device)], dim=-1)

            print(self.tokenizer.decode([next_token]), end="", flush=True)
            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated)
