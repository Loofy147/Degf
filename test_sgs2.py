import torch
from transformer_lens import HookedTransformer
from sgs2_prototype import SGS2Prototype

def test_sgs2_gen():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    sgs2 = SGS2Prototype(model)

    # Test case 1: Logic
    prompt = "The square root of 64 is"
    res = sgs2.generate(prompt, max_new_tokens=5, verbose=True)
    print(f"RES1: {res}")

    # Test case 2: Pattern (lower G)
    prompt = "1, 2, 3,"
    res = sgs2.generate(prompt, max_new_tokens=5, verbose=True)
    print(f"RES2: {res}")

if __name__ == "__main__":
    test_sgs2_gen()
