import torch
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

def get_ioi_benchmark():
    # Simple IOI prompts
    prompts = [
        ("When John and Mary went to the store, John gave a drink to", " Mary"),
        ("After Alice and Bob finished lunch, Alice gave a book to", " Bob"),
        ("Since Sarah and Tom were thirsty, Sarah gave some water to", " Tom"),
        ("While David and Susan were at the park, David gave a ball to", " Susan"),
        ("Because James and Emma were cold, James gave a blanket to", " Emma")
    ]
    return prompts

def get_induction_benchmark():
    # Simple Induction prompts: [A][B] ... [A] -> [B]
    prompts = [
        ("The quick brown fox jumps over the lazy dog. The quick brown", " fox"),
        ("To be or not to be, that is the question. To be or not to", " be"),
        ("One fish two fish red fish blue fish. One fish two fish red", " fish"),
        ("Double double toil and trouble. Double double toil and", " trouble"),
        ("I think therefore I am. I think therefore I", " am")
    ]
    return prompts

def evaluate_accuracy(model, dataset):
    correct = 0
    for prompt, target in dataset:
        tokens = model.to_tokens(prompt)
        target_token = model.to_single_token(target)
        logits = model(tokens)
        pred_token = logits[0, -1].argmax().item()
        if pred_token == target_token:
            correct += 1
    return correct / len(dataset)

def run_ablation(model, targets, dataset):
    """
    Mean ablation: replace head output with zero (or mean, here zero for simplicity)
    """
    # targets is list of (layer, head)
    # We use hooks to zero out the output of these heads

    def ablation_hook(value, hook):
        # value: (batch, pos, head, d_head)
        for l, h in targets:
            if hook.layer() == l:
                value[:, :, h, :] = 0.0
        return value

    hook_names = [get_act_name("z", l) for l in range(model.cfg.n_layers)]

    with model.hooks(fwd_hooks=[(name, ablation_hook) for name in hook_names]):
        acc = evaluate_accuracy(model, dataset)

    return acc

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    # Load targets
    targets = []
    with open("q2_targets.txt", "r") as f:
        for line in f:
            l, h = map(int, line.strip().split(","))
            targets.append((l, h))

    print(f"Loaded {len(targets)} Q2 targets for ablation.")

    ioi_data = get_ioi_benchmark()
    ind_data = get_induction_benchmark()

    print("\nBaseline Results:")
    base_ioi = evaluate_accuracy(model, ioi_data)
    base_ind = evaluate_accuracy(model, ind_data)
    print(f"IOI Accuracy: {base_ioi:.2%}")
    print(f"Induction Accuracy: {base_ind:.2%}")

    print("\nRunning Ablation on Q2 targets...")
    abl_ioi = run_ablation(model, targets, ioi_data)
    abl_ind = run_ablation(model, targets, ind_data)

    print(f"Ablated IOI Accuracy: {abl_ioi:.2%}")
    print(f"Ablated Induction Accuracy: {abl_ind:.2%}")

    ioi_drop = (base_ioi - abl_ioi)
    ind_drop = (base_ind - abl_ind)

    print(f"\nResults Analysis:")
    print(f"IOI Drop: {ioi_drop:.2%}")
    print(f"Induction Drop: {ind_drop:.2%}")

    if ioi_drop > 0.35 and ind_drop < 0.10:
        print("✅ DOUBLE-DISSOCIATION VALIDATED!")
    else:
        print("❌ VALIDATION FAILED (or check target density).")
