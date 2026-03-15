# scripts/debug_nli.py
from sentence_transformers import CrossEncoder
import numpy as np

#  UPDATED MODEL
model = CrossEncoder('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')

pairs = [
    # Case 1: Clear Entailment (Should be high)
    ("A man is playing soccer.", "A person is playing a sport."),
    
    # Case 2: Clear Contradiction (Should be low)
    ("A man is playing soccer.", "A man is sitting on a couch."),
    
    # Case 3: Neutral (Should be middle/low)
    ("A man is playing soccer.", "It is a sunny day.")
]

scores = model.predict(pairs)
probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

print("\n NLI Model Mapping Debugger")
print(f"Model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7\n")

for i, (pair, prob) in enumerate(zip(pairs, probs)):
    print(f"Pair {i+1} Scores: {prob}")
    winner = np.argmax(prob)
    print(f"  Winner: Index {winner} ({prob[winner]:.4f})\n")

print(" CONCLUSION:")
print("If Pair 1 (Soccer/Sport) wins at Index 0 -> We must change code to `probs[0]`.")
print("If Pair 1 (Soccer/Sport) wins at Index 1 -> Something else is wrong.")