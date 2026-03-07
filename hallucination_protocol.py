import torch
import numpy as np
from typing import List, Tuple, Dict
from monitor_gpt2 import DEGFMonitor

class HallucinationProtocol:
    """
    Milestone D4: Hallucination Detection Protocol.
    Uses the "Low G + High Confidence" thermodynamic signature.
    Signature: G < 0.4 AND tc < 0.4
    """
    def __init__(self, model, g_threshold=0.4, tc_threshold=0.4):
        self.monitor = DEGFMonitor(model)
        self.g_threshold = g_threshold
        self.tc_threshold = tc_threshold

    def evaluate_signature(self, text: str) -> Dict:
        """Analyze a string and detect if it contains a thermodynamic hallucination."""
        g_stream = self.monitor.monitor_step(text)
        triggered = []
        for i, entry in enumerate(g_stream):
            if entry["G"] < self.g_threshold and entry["tc"] < self.tc_threshold:
                triggered.append(entry)

        if triggered:
            target = min(triggered, key=lambda x: x["tc"])
            is_hallu = True
        else:
            target = g_stream[-2] if len(g_stream) > 1 else g_stream[-1]
            is_hallu = False

        return {
            "token": target["token"],
            "G": target["G"],
            "tc": target["tc"],
            "detected": is_hallu,
            "stream": g_stream
        }

    def run_benchmark(self, dataset: List[Tuple[str, bool]]) -> Dict:
        """Run the protocol on a dataset of (Text, Is_Hallucination)."""
        tp, fp, tn, fn = 0, 0, 0, 0
        results = []

        print(f"{'Text':<40} | {'G':<6} | {'tc':<6} | {'Label':<6} | {'Det'}")
        print("-" * 75)

        for text, label in dataset:
            res = self.evaluate_signature(text)
            detected = res["detected"]

            if label: # Ground truth: is hallucination
                if detected: tp += 1
                else: fn += 1
            else: # Ground truth: is fact
                if detected: fp += 1
                else: tn += 1

            print(f"{text[:40]:<40} | {res['G']:.3f} | {res['tc']:.3f} | {'HALLU' if label else 'FACT':<6} | {detected}")
            results.append({"text": text, "label": label, "result": res})

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "results": results
        }

DOG_FEEDING_DATASET = [
    ("All dogs are mammals.", False),
    ("Rex is a golden retriever.", False),
    ("Golden retrievers are dogs.", False),
    ("Dogs need food to survive.", False),
    ("Rex eats dog food.", False),
    ("All dogs are reptiles.", True),
    ("Rex is a type of fish.", True),
    ("Golden retrievers are cats.", True),
    ("Dogs can live without air.", True),
    ("Rex eats only sunlight.", True),
    ("All dogs are animals. Rex is a dog. Therefore, Rex is an animal.", False),
    ("All mammals breathe air. Dogs are mammals. Therefore, dogs breathe air.", False),
    ("All dogs are animals. Rex is a dog. Therefore, Rex is a plant.", True),
    ("All mammals have fur. Rex is a mammal. Therefore, Rex has scales.", True)
]
