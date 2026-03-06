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
        # FIX-5: Parameterized Phase Gate
        self.plateau_threshold = 0.60
        self.synthesis_threshold = 0.01

    def get_latent_G(self, tokens):
        """Compute mean G for reasoning layers based on current cache."""
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=lambda n: "pattern" in n)

            layer_Gs = []
            for l in self.reasoning_layers:
                pattern = cache["pattern", l]
                if pattern.ndim == 4:
                    pattern = pattern[0]

                all_G = []
                for h in range(self.n_heads):
                    attn = pattern[h].cpu().numpy()
                    H = compute_H_series(attn)
                    use_detrended = (l < int(0.65 * self.n_layers))
                    V = compute_V_detrended(H) if use_detrended else compute_V(H)
                    C = count_collapses(H)
                    all_G.append(compute_G(V, C))
                layer_Gs.append(np.mean(all_G))

        return float(np.mean(layer_Gs))

    def forward(self, text, max_loops=3, verbose=False):
        tokens = self.model.to_tokens(text)
        resid = self.model.embed(tokens) + self.model.pos_embed(tokens)

        if verbose: print(f"Entering Latent Reasoner Loop (max {max_loops})...")
        prev_G = 0.0

        for loop in range(max_loops):
            current_resid = resid.clone()
            for l in self.reasoning_layers:
                current_resid = self.model.blocks[l](current_resid)

            # FIX-5: Real Attention G Calculation
            current_G = self.get_latent_G(tokens)
            delta_G = current_G - prev_G
            if verbose: print(f"  Loop {loop}: G={current_G:.3f}, dG={delta_G:+.3f}")

            # Recurrence Condition
            if delta_G > self.synthesis_threshold:
                if verbose: print(f"    dG > {self.synthesis_threshold}: Still synthesizing. Recurse.")
                resid = current_resid
                prev_G = current_G
                continue
            elif current_G < self.plateau_threshold:
                if verbose: print(f"    G < {self.plateau_threshold}: Logical click / plateau achieved. Opening Phase Gate.")
                resid = current_resid
                break
            else:
                if verbose: print("    High G plateau. Release.")
                resid = current_resid
                break

        if verbose: print("Entering Syntax Decoder...")
        for l in self.decoder_layers:
            resid = self.model.blocks[l](resid)

        resid = self.model.ln_final(resid)
        logits = self.model.unembed(resid)
        return logits

    def generate(self, text, max_new_tokens=20, max_loops=3, window=5, threshold=-0.15, verbose=False):
        """
        Full generation loop with Phase Gate and Elaboration Guillotine.
        """
        generated_text = text
        g_history = []

        # Initial G for prompt
        tokens = self.model.to_tokens(text)
        current_G = self.get_latent_G(tokens)
        g_history.append(current_G)

        print(f"\nStarting SGS-2 Generation for: '{text}'")

        for i in range(max_new_tokens):
            logits = self.forward(generated_text, max_loops=max_loops, verbose=verbose)
            next_token = logits[0, -1].argmax().item()
            next_token_str = self.model.to_string(next_token)

            # Update sequence
            generated_text += next_token_str
            tokens = self.model.to_tokens(generated_text)

            # Monitor G
            current_G = self.get_latent_G(tokens)
            g_history.append(current_G)

            print(f"Token {i+1:2d}: '{next_token_str}' | G: {current_G:.3f}")

            # Elaboration Guillotine
            if len(g_history) >= window:
                delta_G = g_history[-1] - g_history[-window]
                if delta_G < threshold:
                    print(f"\n[Guillotine] Truncating at token '{next_token_str}' (delta G: {delta_G:.3f} below threshold {threshold})")
                    break

            if next_token == self.model.tokenizer.eos_token_id:
                break

        return generated_text

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sgs2 = SGS2Prototype(model)

    prompt = "If all men are mortal and Socrates is a man, then Socrates is"
    result = sgs2.generate(prompt, max_new_tokens=10, verbose=False)
    print(f"\nFinal Result: {result}")
