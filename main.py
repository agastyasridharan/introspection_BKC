from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from inject_concept_vector import inject_concept_vector
import torch
import pandas as pd
import argparse
from collections import defaultdict

torch.manual_seed(2881)

# =====================================================
# TWO-STAGE TRIAL
# =====================================================

def run_two_stage_trial(
    vector_path,
    model,
    tokenizer,
    layer,
    coeff,
    stated_rate,
    concept,
    vec_type,
    assistant_tokens_only
):
    """
    Two-stage experiment:
    Stage 1: YES/NO detection
    Stage 2: Concept identification (only if YES)
    """

    from all_prompts import (
        get_anthropic_reproduce_stated20_stage1_messages,
        get_anthropic_reproduce_stated50_stage1_messages,
        get_anthropic_reproduce_stated80_stage1_messages,
        get_anthropic_reproduce_stage2_messages
    )
    from api_utils import parse_yes_no, extract_concept_match

    # -------------------------
    # Load vector (robust)
    # -------------------------
    vector_data = torch.load(vector_path, weights_only=False)
    if isinstance(vector_data, dict):
        steering_vector = vector_data.get("vector", vector_data)
    else:
        steering_vector = vector_data

    # -------------------------
    # Stage 1 prompt selection
    # -------------------------
    if stated_rate == "20":
        stage1_messages = get_anthropic_reproduce_stated20_stage1_messages()
    elif stated_rate == "50":
        stage1_messages = get_anthropic_reproduce_stated50_stage1_messages()
    elif stated_rate == "80":
        stage1_messages = get_anthropic_reproduce_stated80_stage1_messages()
    else:
        raise ValueError(f"Invalid stated_rate: {stated_rate}")

    # -------------------------
    # STAGE 1
    # -------------------------
    stage1_prompt = tokenizer.apply_chat_template(
        stage1_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    stage1_response = inject_concept_vector(
        model=model,
        tokenizer=tokenizer,
        steering_vector=steering_vector,
        layer_to_inject=layer,
        coeff=coeff,
        inference_prompt=stage1_prompt,
        assistant_tokens_only=assistant_tokens_only,
        max_new_tokens=10
    )

    detection_binary = parse_yes_no(stage1_response)

    # -------------------------
    # STAGE 2 (conditional)
    # -------------------------
    stage2_response = None
    identification_binary = None

    if detection_binary == "YES":
        stage2_messages = get_anthropic_reproduce_stage2_messages(stage1_response)

        # IMPORTANT FIX:
        # stage2_messages already includes the assistant response,
        # so DO NOT add it again.
        stage2_prompt = tokenizer.apply_chat_template(
            stage1_messages + stage2_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        stage2_response = inject_concept_vector(
            model=model,
            tokenizer=tokenizer,
            steering_vector=steering_vector,
            layer_to_inject=layer,
            coeff=coeff,
            inference_prompt=stage2_prompt,
            assistant_tokens_only=assistant_tokens_only,
            max_new_tokens=15
        )

        identification_binary = extract_concept_match(stage2_response, concept)

    return {
        "concept": concept,
        "vec_type": vec_type,
        "layer": layer,
        "coeff": coeff,
        "stated_rate": stated_rate,
        "actual_injection": coeff > 0,
        "assistant_tokens_only": assistant_tokens_only,
        "stage1_response": stage1_response,
        "stage2_response": stage2_response,
        "detection_binary": detection_binary,
        "identification_binary": identification_binary,
        "response": (
            f"STAGE1: {stage1_response} | STAGE2: {stage2_response}"
            if stage2_response else f"STAGE1: {stage1_response}"
        )
    }

# =====================================================
# MAIN DRIVER
# =====================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 18])
    parser.add_argument("--coeffs", type=float, nargs="+", default=[4, 8, 12])
    parser.add_argument("--stated_rate", type=str, choices=["20", "50", "80"], default="50")
    parser.add_argument("--assistant_tokens_only", action="store_true", default=True)
    parser.add_argument("--include_controls", action="store_true", default=True)
    args = parser.parse_args()

    # -------------------------
    # Load vectors
    # -------------------------
    vector_root = Path("saved_vectors/llama")
    if not vector_root.exists():
        raise FileNotFoundError("saved_vectors/llama not found. Run save_vectors.py first.")

    vectors = defaultdict(lambda: defaultdict(dict))
    for file in vector_root.glob("*.pt"):
        parts = file.stem.split("_")
        if len(parts) < 3:
            continue
        vec_type = parts[-1]
        layer = int(parts[-2])
        concept = "_".join(parts[:-2])
        if layer in args.layers:
            vectors[concept][layer][vec_type] = file

    concepts = sorted(vectors.keys())
    print(f"Loaded {len(concepts)} concepts.")

    # -------------------------
    # Load model ONCE
    # -------------------------
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    # -------------------------
    # Run experiments
    # -------------------------
    results = []

    coeffs_to_run = list(args.coeffs)
    if args.include_controls and 0 not in coeffs_to_run:
        coeffs_to_run = [0] + coeffs_to_run

    for concept in concepts:
        for layer in args.layers:
            if layer not in vectors[concept]:
                continue
            for vec_type, vector_path in vectors[concept][layer].items():
                for coeff in coeffs_to_run:
                    print(f"Running: {concept} | layer {layer} | vec {vec_type} | coeff {coeff}")
                    result = run_two_stage_trial(
                        vector_path=vector_path,
                        model=model,
                        tokenizer=tokenizer,
                        layer=layer,
                        coeff=coeff,
                        stated_rate=args.stated_rate,
                        concept=concept,
                        vec_type=vec_type,
                        assistant_tokens_only=args.assistant_tokens_only
                    )
                    results.append(result)

    # -------------------------
    # Save results
    # -------------------------
    results_df = pd.DataFrame(results)
    out_dir = Path("new_results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"two_stage_stated{args.stated_rate}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # -------------------------
    # Summary statistics (FIXED)
    # -------------------------
    print("\n=== SUMMARY ===")
    for coeff in sorted(results_df["coeff"].unique()):
        subset = results_df[results_df["coeff"] == coeff]
        detection_rate = (subset["detection_binary"] == "YES").mean()

        yes_subset = subset[subset["detection_binary"] == "YES"]
        if len(yes_subset) > 0:
            identification_rate = yes_subset["identification_binary"].mean()
        else:
            identification_rate = 0.0

        print(
            f"Coeff {coeff:>4}: "
            f"Detection={detection_rate:.2%}, "
            f"Identification|Detection={identification_rate:.2%}"
        )

if __name__ == "__main__":
    main()
