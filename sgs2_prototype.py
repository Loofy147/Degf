import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from degf_core import compute_H_series, compute_V, compute_V_detrended, compute_G, count_collapses

class SGS2Prototype(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        # Partitioning based on paper's SGS-1 spec (scaled to 12 layers)
        self.reasoning_layers = list(range(0, 9))  # L0-L8
        self.decoder_layers = list(range(9, 12))   # L9-L11

    def get_latent_G(self, tokens):
        """
        Compute mean G for reasoning layers based on current residual state.
        In a production implementation, this would use KV-cache to avoid recomputation.
        Here we use HookedTransformer's cache mechanism.
        """
        with torch.no_grad():
            # Extract patterns from reasoning layers
            _, cache = self.model.run_with_cache(tokens, names_filter=lambda n: "pattern" in n)

            layer_Gs = []
            for l in self.reasoning_layers:
                pattern = cache["pattern", l] # (batch, head, q, k)
                if pattern.ndim == 4:
                    pattern = pattern[0] # remove batch dim -> (head, q, k)

                all_G = []
                for h in range(self.n_heads):
                    attn = pattern[h].cpu().numpy()
                    H = compute_H_series(attn)
                    # Use detrended V for early layers
                    is_context = l < int(0.65 * self.n_layers)
                    V = compute_V_detrended(H) if is_context else compute_V(H)
                    C = count_collapses(H)
                    all_G.append(compute_G(V, C))
                layer_Gs.append(np.mean(all_G))

        return float(np.mean(layer_Gs))

    def forward(self, text, max_loops=3):
        tokens = self.model.to_tokens(text)
        # 1. Initial Embedding
        resid = self.model.embed(tokens) + self.model.pos_embed(tokens)

        # 2. Latent Reasoner with Recurrence
        # Implementation Note: In HookedTransformer, we don't have direct access
        # to a differentiable KV-cache that persists across loops in this manual loop.
        # But for this prototype, we re-run reasoning layers and track G.

        print(f"Entering Latent Reasoner Loop (max {max_loops})...")
        prev_G = 0.0

        for loop in range(max_loops):
            # Run reasoning layers
            # We clone resid to ensure clean loops if we were implementing deeper logic
            current_resid = resid.clone()
            for l in self.reasoning_layers:
                current_resid = self.model.blocks[l](current_resid)

            # Phase Gate Logic - Actual G Computation
            # We use tokens here to compute G from attention patterns
            current_G = self.get_latent_G(tokens)

            delta_G = current_G - prev_G
            print(f"  Loop {loop}: G={current_G:.3f}, dG={delta_G:+.3f}")

            # Recurrence Condition: dG > 0 (Synthesis phase)
            if delta_G > 0.01:
                print("    dG > 0: Still synthesizing. Recurse.")
                # We update 'resid' with the output of the reasoning layers for the next loop
                resid = current_resid
                prev_G = current_G
                continue
            elif current_G < 0.60:
                print("    Logical click / plateau achieved. Opening Phase Gate.")
                resid = current_resid
                break
            else:
                print("    High G plateau. Release.")
                resid = current_resid
                break

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
