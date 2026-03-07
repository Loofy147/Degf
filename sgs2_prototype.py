import torch
import torch.nn as nn
import numpy as np
from transformer_lens import HookedTransformer
from degf_core import compute_H_series, compute_V, compute_V_detrended, compute_G, count_collapses

class SGS2Prototype(nn.Module):
    def __init__(self, model, tokenizer=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        # Determine model type
        self.is_hf = hasattr(model, "config")
        if self.is_hf:
            self.n_layers = model.config.num_hidden_layers
            self.n_heads = model.config.num_attention_heads
            # Partition: last 3 layers for syntax
            self.decoder_layers = list(range(self.n_layers - 3, self.n_layers))
            self.reasoning_layers = list(range(0, self.n_layers - 3))
        else:
            self.n_layers = model.cfg.n_layers
            self.n_heads = model.cfg.n_heads
            self.reasoning_layers = list(range(0, 9))
            self.decoder_layers = list(range(9, 12))

        self.plateau_threshold = 0.60
        self.synthesis_threshold = 0.01

    def get_latent_G(self, tokens):
        """Compute mean G for reasoning layers."""
        with torch.no_grad():
            if self.is_hf:
                outputs = self.model(tokens, output_attentions=True)
                attentions = outputs.attentions
            else:
                _, cache = self.model.run_with_cache(tokens, names_filter=lambda n: "pattern" in n)

            layer_Gs = []
            for l in self.reasoning_layers:
                if self.is_hf:
                    pattern = attentions[l][0]
                else:
                    pattern = cache["pattern", l]
                    if pattern.ndim == 4: pattern = pattern[0]

                all_G = []
                for h in range(self.n_heads):
                    attn = pattern[h].float().cpu().numpy()
                    if self.is_hf:
                        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-12)
                    H = compute_H_series(attn)
                    use_detrended = (l < int(0.65 * self.n_layers))
                    V = compute_V_detrended(H) if use_detrended else compute_V(H)
                    C = count_collapses(H)
                    all_G.append(compute_G(V, C))
                layer_Gs.append(np.mean(all_G))

        return float(np.mean(layer_Gs))

    def forward(self, tokens, max_loops=3, verbose=False):
        if self.is_hf:
            # For HF models, we use the standard forward pass but potentially 'recurse'
            # by rerunning the full model if synthesis is still active.
            # This is a high-level architectural simulation.
            prev_G = 0.0
            for loop in range(max_loops):
                outputs = self.model(tokens, output_attentions=True)
                current_G = self.get_latent_G(tokens)
                delta_G = current_G - prev_G
                if verbose: print(f"  Loop {loop}: G={current_G:.3f}, dG={delta_G:+.3f}")

                if delta_G > self.synthesis_threshold:
                    prev_G = current_G
                    continue
                else:
                    break
            return outputs.logits
        else:
            # TransformerLens implementation (Original)
            resid = self.model.embed(tokens) + self.model.pos_embed(tokens)
            prev_G = 0.0
            for loop in range(max_loops):
                current_resid = resid.clone()
                for l in self.reasoning_layers:
                    current_resid = self.model.blocks[l](current_resid)
                current_G = self.get_latent_G(tokens)
                delta_G = current_G - prev_G
                if verbose: print(f"  Loop {loop}: G={current_G:.3f}, dG={delta_G:+.3f}")
                if delta_G > self.synthesis_threshold:
                    resid = current_resid
                    prev_G = current_G
                    continue
                else:
                    resid = current_resid
                    break
            for l in self.decoder_layers:
                resid = self.model.blocks[l](resid)
            resid = self.model.ln_final(resid)
            logits = self.model.unembed(resid)
            return logits

    def generate(self, text, max_new_tokens=20, max_loops=3, window=5, threshold=-0.15, verbose=False):
        generated_text = text
        g_history = []

        if self.is_hf:
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.model.device)
        else:
            tokens = self.model.to_tokens(text)

        current_G = self.get_latent_G(tokens)
        g_history.append(current_G)

        print(f"\nStarting SGS-2 Generation for: '{text}'")

        for i in range(max_new_tokens):
            logits = self.forward(tokens, max_loops=max_loops, verbose=verbose)
            next_token = logits[0, -1].argmax().item()

            next_token_tensor = torch.tensor([[next_token]], device=tokens.device)
            tokens = torch.cat([tokens, next_token_tensor], dim=-1)

            if self.is_hf:
                next_token_str = self.tokenizer.decode(next_token)
                eos_id = self.tokenizer.eos_token_id
            else:
                next_token_str = self.model.to_string(next_token)
                eos_id = self.model.tokenizer.eos_token_id

            generated_text += next_token_str
            current_G = self.get_latent_G(tokens)
            g_history.append(current_G)

            print(f"Token {i+1:2d}: '{next_token_str}' | G: {current_G:.3f}")

            if len(g_history) >= window:
                delta_G = g_history[-1] - g_history[-window]
                if delta_G < threshold:
                    print(f"\n[Guillotine] Truncating (delta G: {delta_G:.3f} below {threshold})")
                    break

            if next_token == eos_id:
                break

        return generated_text
