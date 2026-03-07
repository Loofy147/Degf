import torch
from transformer_lens import HookedTransformer
from hallucination_protocol import HallucinationProtocol, DOG_FEEDING_DATASET

def run_scale_experiment():
    device = "cpu"
    print("Loading GPT-2-medium for D4 Scaling Protocol...")
    model = HookedTransformer.from_pretrained("gpt2-medium", device=device)

    protocol = HallucinationProtocol(model)
    print("\nRunning Milestone D4 on GPT-2-medium (Dog Feeding Suite)")
    results = protocol.run_benchmark(DOG_FEEDING_DATASET)

    print("\n--- GPT-2-medium Protocol Summary ---")
    print(f"F1 Score: {results['f1']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"TP: {results['tp']} | FP: {results['fp']} | TN: {results['tn']} | FN: {results['fn']}")

if __name__ == "__main__":
    run_scale_experiment()
