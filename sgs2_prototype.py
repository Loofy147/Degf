import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from degf_core import compute_H_series, compute_V, compute_G, count_collapses

class SGS2Prototype(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.cfg.n_layers
        # Partitioning based on paper's SGS-1 spec (scaled to 12 layers)
        self.reasoning_layers = list(range(0, 9))  # L0-L8
        self.decoder_layers = list(range(9, 12))   # L9-L11

    def get_layer_G(self, cache, layer_idx, t):
        """Compute mean G for a specific layer up to token position t."""
        pattern = cache["pattern", layer_idx] # (head, q, k)
        all_G = []
        for h in range(self.model.cfg.n_heads):
            attn = pattern[h, :t+1, :t+1].cpu().numpy()
            H = compute_H_series(attn)
            V = compute_V(H)
            C = count_collapses(H)
            all_G.append(compute_G(V, C))
        return sum(all_G) / len(all_G)

    def forward(self, text, max_loops=3):
        tokens = self.model.to_tokens(text)
        # 1. Initial Embedding
        resid = self.model.embed(tokens) + self.model.pos_embed(tokens)

        # 2. Latent Reasoner with Recurrence
        loop_count = 0
        prev_G = 0

        print(f"Entering Latent Reasoner Loop (max {max_loops})...")

        # We need a way to run specific layers.
        # TransformerLens allows this via hooks or manual block application.

        for loop in range(max_loops):
            # Run reasoning layers
            for l in self.reasoning_layers:
                resid = self.model.blocks[l](resid)

            # Phase Gate Logic
            # Note: In a real implementation, we'd need attention patterns from this pass.
            # Here we simulate the monitor.
            # Assume G increases during synthesis and drops on "logical click".

            # Simulation of G dynamics based on loop count
            if loop == 0:
                current_G = 0.45 # Still synthesizing
            elif loop == 1:
                current_G = 0.72 # Peak reasoning
            else:
                current_G = 0.51 # Post-click drop

            delta_G = current_G - prev_G
            print(f"  Loop {loop}: G={current_G:.3f}, dG={delta_G:+.3f}")

            if delta_G > 0:
                print("    dG > 0: Still synthesizing. Recurse.")
                prev_G = current_G
                continue
            elif current_G < 0.60: # Threshold for release
                print("    Logical click / plateau achieved. Opening Phase Gate.")
                break
            else:
                prev_G = current_G

        # 3. Syntax Decoder
        print("Entering Syntax Decoder...")
        for l in self.decoder_layers:
            resid = self.model.blocks[l](resid)

        resid = self.model.ln_final(resid)
        logits = self.model.unembed(resid)

        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sgs2 = SGS2Prototype(model)

    prompt = "The square root of 64 plus 36 is"
    logits = sgs2(prompt)

    # Check prediction
    pred_token = logits[0, -1].argmax().item()
    print(f"\nPrompt: {prompt}")
    print(f"Prediction: {model.to_string(pred_token)}")
