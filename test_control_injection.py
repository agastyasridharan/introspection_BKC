"""
Test script to verify that coeff=0 produces true control trials.
This ensures control trials don't accidentally inject concepts.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from inject_concept_vector import inject_concept_vector
import torch
from pathlib import Path

# Load model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")

# Load a test vector (use any existing vector)
vector_dir = Path("saved_vectors/llama/")
if not vector_dir.exists():
    print("ERROR: saved_vectors/llama/ directory not found.")
    print("You need to run save_vectors.py first to generate concept vectors.")
    exit(1)

vector_files = list(vector_dir.glob("*.pt"))
if not vector_files:
    print("ERROR: No .pt files found in saved_vectors/llama/")
    print("You need to run save_vectors.py first.")
    exit(1)

test_vector_path = vector_files[0]
print(f"\nUsing test vector: {test_vector_path.name}")

# Load vector properly
vector_data = torch.load(test_vector_path, map_location=device)
steering_vector = vector_data["vector"]
concept_name = vector_data["concept_name"]

print(f"Concept: {concept_name}")

# Test prompt
test_prompt = "Tell me about your thoughts right now."

print("\n" + "=" * 60)
print("TEST 1: Control trial (coeff=0, should NOT mention concept)")
print("=" * 60)

response_control = inject_concept_vector(
    model=model,
    tokenizer=tokenizer,
    steering_vector=steering_vector,
    layer_to_inject=vector_data["layer"],
    coeff=0.0,  # CONTROL
    inference_prompt=test_prompt,
    assistant_tokens_only=True,
    max_new_tokens=30,
)

print(f"Control response (coeff=0):\n{response_control}\n")
control_mentions_concept = concept_name.lower() in response_control.lower()
print(f"Mentions '{concept_name}': {control_mentions_concept}")

print("\n" + "=" * 60)
print("TEST 2: Injection trial (coeff=12, should mention concept)")
print("=" * 60)

response_injection = inject_concept_vector(
    model=model,
    tokenizer=tokenizer,
    steering_vector=steering_vector,
    layer_to_inject=vector_data["layer"],
    coeff=12.0,  # INJECTION
    inference_prompt=test_prompt,
    assistant_tokens_only=True,
    max_new_tokens=30,
)

print(f"Injection response (coeff=12):\n{response_injection}\n")
injection_mentions_concept = concept_name.lower() in response_injection.lower()
print(f"Mentions '{concept_name}': {injection_mentions_concept}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Control (coeff=0) mentions concept: {control_mentions_concept}")
print(f"Injection (coeff=12) mentions concept: {injection_mentions_concept}")

if control_mentions_concept and injection_mentions_concept:
    print("\n⚠️  WARNING: Both mention the concept. This suggests coeff=0 is NOT a true control.")
    print("   The injection may not be properly disabled when coeff=0.")
elif not control_mentions_concept and injection_mentions_concept:
    print("\n✓ PASS: Control doesn't mention concept, injection does.")
    print("  coeff=0 appears to work correctly as a control condition.")
elif not control_mentions_concept and not injection_mentions_concept:
    print("\n⚠️  WARNING: Neither mentions the concept.")
    print("   This could mean: (1) weak injection, (2) wrong layer, or (3) concept not salient.")
    print("   Try running again or use a different vector/layer.")
else:
    print("\n⚠️  UNEXPECTED: Control mentions concept but injection doesn't.")
    print("   This is very unusual. Try running again.")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
if control_mentions_concept:
    print("❌ DO NOT PROCEED: Fix coeff=0 handling in inject_concept_vector.py first")
else:
    print("✓ SAFE TO PROCEED: coeff=0 works as expected")