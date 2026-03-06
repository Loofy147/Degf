import torch
from transformer_lens import HookedTransformer
from train_thermo import compute_thermo_loss

def test_thermo_loss():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    tokens = model.to_tokens("If all men are mortal and Socrates is a man, then Socrates is mortal.")

    loss, ce, reward = compute_thermo_loss(model, tokens)
    print(f"Loss: {loss.item():.4f}, CE: {ce.item():.4f}, Reward: {reward.item():.4f}")

    # Assert ce is positive
    assert ce > 0
    # Assert reward is at least non-negative (it could be zero)
    assert reward >= 0

if __name__ == "__main__":
    test_thermo_loss()
