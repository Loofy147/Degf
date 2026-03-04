import torch
import numpy as np
from transformer_lens import HookedTransformer
from degf_core import compute_H_series, compute_V, compute_V_detrended, compute_G, count_collapses, HeadProfile

def scan_model(model, prompts):
    """Scan model across prompts and return list of HeadProfiles."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    profiles = []

    # Simple implementation for benchmarking: use first prompt
    # and compute profiles for all heads.
    text = prompts[0]
    with torch.no_grad():
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

        # Calculate token costs (surprisals)
        logits = model(tokens)
        log_probs = logits[0, :-1, :].log_softmax(dim=-1)
        labels = tokens[0, 1:]
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        surprisals = -token_log_probs / np.log(2)
        tc_normalized = torch.clamp(surprisals / 10.0, 0, 1).cpu().numpy()

        for l in range(n_layers):
            pattern = cache["pattern", l]
            for h in range(n_heads):
                attn = pattern[h].cpu().numpy()
                H = compute_H_series(attn)
                # HeadProfile token_cost is mean tc for the sequence
                mean_tc = float(np.mean(tc_normalized))
                profiles.append(HeadProfile(layer=l, head=h, entropy_series=H, token_cost=mean_tc))

    return profiles

class DEGFMonitor:
    def __init__(self, model):
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads

    def compute_quality(self, G, tc):
        return 0.802 * G - 0.113 * tc + 0.145

    def monitor_step(self, text):
        with torch.no_grad():
            tokens = self.model.to_tokens(text)
            logits, cache = self.model.run_with_cache(tokens, remove_batch_dim=True)

            if logits.ndim == 3:
                logits = logits[0]

            seq_len = tokens.shape[1]
            g_stream = []

            log_probs = logits.log_softmax(dim=-1)
            labels = tokens[0, 1:]
            token_log_probs = log_probs[:-1, :].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            surprisals = -token_log_probs / np.log(2)
            tc_normalized = torch.clamp(surprisals / 10.0, 0, 1).cpu().numpy()

            for t in range(seq_len):
                token_str = self.model.to_string(tokens[0, t])
                tc = float(tc_normalized[t-1]) if t > 0 else 0.5

                all_G = []
                for l in range(self.n_layers):
                    pattern = cache["pattern", l]
                    for h in range(self.n_heads):
                        attn_full = pattern[h, :t+1, :t+1].cpu().numpy()
                        H_series = compute_H_series(attn_full)
                        is_context = l < int(0.65 * self.n_layers); V = compute_V_detrended(H_series) if is_context else compute_V(H_series)
                        C = count_collapses(H_series)
                        G = compute_G(V, C)
                        all_G.append(G)

                mean_G = float(np.mean(all_G))
                quality = self.compute_quality(mean_G, tc)
                hallucination_risk = "HIGH" if mean_G < 0.3 and tc < 0.3 else "LOW"

                g_stream.append({
                    "token": token_str,
                    "G": mean_G,
                    "tc": tc,
                    "Q": quality,
                    "hallucination_risk": hallucination_risk
                })

            return g_stream

    def apply_guillotine(self, g_stream, window=5, threshold=-0.20):
        if len(g_stream) < window:
            return g_stream

        for i in range(window, len(g_stream)):
            delta_G = g_stream[i]["G"] - g_stream[i-window]["G"]
            if delta_G < threshold:
                print(f"\n[Guillotine] Truncating at token '{g_stream[i]['token']}' (delta G: {delta_G:.3f})")
                return g_stream[:i+1]
        return g_stream

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    monitor = DEGFMonitor(model)
    prompt = "If all men are mortal and Socrates is a man, then Socrates is mortal. This is an example of a syllogism."
    g_stream = monitor.monitor_step(prompt)

    print(f"{'Token':<15} | {'G-score':<8} | {'Quality':<8} | {'Risk':<5}")
    print("-" * 50)
    for entry in g_stream:
        print(f"{entry['token']:<15} | {entry['G']:.4f} | {entry['Q']:.4f} | {entry['hallucination_risk']}")

    truncated = monitor.apply_guillotine(g_stream)
