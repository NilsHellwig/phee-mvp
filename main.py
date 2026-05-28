import os
import sys
import json
import torch
import itertools
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from gpu_memory import GPUMonitor

# Add project modules
sys.path.append(os.path.abspath("schema"))
from helper import (load_data, load_shots, get_prompt, ORDER_MAIN, 
                    SUBJECT_FIELDS, TREATMENT_FIELDS, save_llm_results, format_main,
                    GOLD_PATH_TEMPLATE, LLM_DICT)
from schema import get_schema_main
from splitter import find_valid_phrases_list

# Config
SPLITS = [1, 2, 3, 4, 5]
N_SHOTS = 100
MODEL_NAME = "google/gemma-4-31B"

def run_mvp(llm, gpu_monitor, split, n_shots, model_config):
    print(f"\n{'='*20} PROCESSING SPLIT {split} (MVP MODE) {'='*20}")
    
    TEST_PATH = GOLD_PATH_TEMPLATE.format(split=split)
    model_file_path = model_config["file_path"]

    # 1. Load data
    test_data = load_data(TEST_PATH)
    print(f"Loaded {len(test_data)} test examples for Split {split}.")

    # 2. Get raw shots (base for all permutations)
    raw_shots = load_shots(n_shot=n_shots, cross_split=split, format="raw")

    # 3. Get all 24 permutations of ORDER_MAIN
    all_permutations = list(itertools.permutations(ORDER_MAIN))
    print(f"Running for {len(all_permutations)} permutations (one by one)...")

    for p_idx, perm in enumerate(all_permutations):
        print(f"  -> Permutation {p_idx+1}/24: {perm}")
        
        # Prepare few-shot examples for THIS permutation
        shots_for_perm = []
        for i in range(n_shots):
            doc_events = [format_main(event, order=perm) for event in raw_shots[i].get("event", [])]
            shots_for_perm.append((raw_shots[i]["text"], doc_events))
        
        prompts_perm = []
        params_perm = []
        
        for entry in test_data:
            sentence = entry["text"]
            
            # MAIN Prompt with specific order
            prompt = get_prompt("main", shots_for_perm, sentence, order_main=perm)
            # save prompt for debugging
            with open("prompts_debug.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            prompts_perm.append(prompt)
            
            # Regex with specific order
            valid_phrases = find_valid_phrases_list(sentence, 64)
            pattern = get_schema_main(valid_phrases, order=perm)
            
            params_perm.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=512,
                    structured_outputs=StructuredOutputsParams(regex=pattern),
                    logprobs=10,
                    seed=0,
                    stop=")]"
                )
            )

        # Generate individually for each permutation
        gpu_monitor.start()
        results_perm = llm.generate(prompts_perm, sampling_params=params_perm)
        watt_p, dur_p = gpu_monitor.stop()

        preds_perm = []
        for res in results_perm:
            t = res.outputs[0].text
            # Prefix with "[" because it was part of the prompt
            if not t.startswith("["):
                t = "[" + t
            
            # Simple fix for common truncation if needed (though Regex should handle it)
            if t.startswith("[") and not t.endswith("]") and not t.endswith(")]"):
                t += ")]"
            preds_perm.append(t)
        
        # Save permutation result with model info in filename
        out_filename = f"split{split}_shots{n_shots}_{model_file_path}_main_mvp_p{p_idx}"
        save_llm_results(preds_perm, results_perm, 
                         {"mean_watt": watt_p, "duration": dur_p}, 
                         filename=out_filename, task_type="main")

if __name__ == "__main__":
    # Config Variables
    SPLITS_TO_RUN = [2]
    N_SHOTS_VAL = 100
    MODEL_TO_USE = "google/gemma-4-31B"
    model_cfg = LLM_DICT[MODEL_TO_USE]

    # Initialize LLM within __main__
    llm = LLM(
        model=model_cfg["vllm"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
        max_model_len=int(8192*1.5),
        seed=0
    )

    gpu_monitor = GPUMonitor()

    for split in SPLITS_TO_RUN:
        run_mvp(llm, gpu_monitor, split, N_SHOTS_VAL, model_cfg)

    print("\nALL OPERATIONS COMPLETED.")
