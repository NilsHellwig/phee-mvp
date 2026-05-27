import os
import sys
import json
from collections import defaultdict

# Add project paths to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from helper import (
    load_data, parse_llm_output, GOLD_PATH_TEMPLATE, PHEE_METRIC_DIR, 
    EVENT_TYPE_MAP, ARG_TYPE_MAP, MISSING_VALUES
)

# Add phee-with-chatgpt path to import the metric
sys.path.append(PHEE_METRIC_DIR)
from scripts.eval_phee.phee_metric import compute_metric

def read_gold_result(gold_file):
    """Parses the gold standard JSONL file into the format required by compute_metric."""
    outputs = []
    with open(gold_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            instance = defaultdict(list)
            instance['context'] = data['text']
            instance['id'].append(data['text_id'])
            instance['is_mult'] = len(data.get('event', [])) > 1

            for event in data.get('event', []):
                ev_type = EVENT_TYPE_MAP.get(event['type'].lower(), event['type'])
                
                for arg in event.get('args', []):
                    role = ARG_TYPE_MAP.get(arg['type'])
                    if role:
                        instance[ev_type + "." + role].append(arg['text'])
            
            outputs.append(instance)
    return outputs

def map_predictions(pred_data, gold_instances, format_type='main'):
    """Maps LLM predictions to the flattened format required by compute_metric."""
    formatted_instances = []
    
    # Sub-argument lists for hierarchy matching
    subj_sub = ["Race", "Age", "Gender", "Population", "Disorder"]
    treat_sub = ["Duration", "Time_elapsed", "Route", "Freq", "Dosage", "Disorder", "Drug"]

    for raw_pred_text, gold in zip(pred_data, gold_instances):
        instance_id = gold['id'][0]
        preds = parse_llm_output(raw_pred_text)
        if not isinstance(preds, (list, tuple)):
            preds = []
            
        pred_dict = defaultdict(list)
        
        for p in preds:
            if format_type == 'main':
                # Expected: (event_type, subject, treatment, effect)
                if len(p) >= 4:
                    et, sub, treat, eff = p[0], p[1], p[2], p[3]
                    if sub and str(sub).lower() not in MISSING_VALUES: pred_dict[f"{et}.Subject"].append(str(sub))
                    if treat and str(treat).lower() not in MISSING_VALUES: pred_dict[f"{et}.Treatment"].append(str(treat))
                    if eff and str(eff).lower() not in MISSING_VALUES: pred_dict[f"{et}.Effect"].append(str(eff))
            else:
                # Expected: {"event_type": ..., "subject": (span, age, ...), "treatment": (span, drug, ...), "effect": ...}
                if isinstance(p, dict):
                    et = p.get("event_type", "Adverse_event")
                    
                    # Main Args
                    sub_tuple = p.get("subject")
                    treat_tuple = p.get("treatment")
                    eff = p.get("effect")

                    if sub_tuple and str(sub_tuple[0]).lower() not in MISSING_VALUES: 
                        pred_dict[f"{et}.Subject"].append(str(sub_tuple[0]))
                    if treat_tuple and str(treat_tuple[0]).lower() not in MISSING_VALUES: 
                        pred_dict[f"{et}.Treatment"].append(str(treat_tuple[0]))
                    if eff and str(eff).lower() not in MISSING_VALUES: 
                        pred_dict[f"{et}.Effect"].append(str(eff))
                    
                    # Sub Args for Subject
                    if sub_tuple:
                        sub_indices = {"Age": 1, "Gender": 2, "Population": 3, "Race": 4, "Disorder": 5}
                        for sub_role in subj_sub:
                            idx = sub_indices[sub_role]
                            if idx < len(sub_tuple):
                                val = sub_tuple[idx]
                                if val and str(val).lower() not in MISSING_VALUES:
                                    pred_dict[f"{et}.Subject.{sub_role}"].append(str(val))
                    
                    # Sub Args for Treatment
                    if treat_tuple:
                        t_idx = {"Drug": 1, "Route": 2, "Dosage": 3, "Time_elapsed": 4, "Duration": 5, "Freq": 6, "Disorder": 8}
                        for sub_role in treat_sub:
                            idx = t_idx[sub_role]
                            if idx < len(treat_tuple):
                                val = treat_tuple[idx]
                                if val and str(val).lower() not in MISSING_VALUES:
                                    pred_dict[f"{et}.Treatment.{sub_role}"].append(str(val))
                        
                        # Combination Drug (Index 7)
                        if len(treat_tuple) > 7:
                            comb_val = treat_tuple[7]
                            if comb_val and str(comb_val).lower() not in MISSING_VALUES:
                                pred_dict[f"{et}.Combination.Drug"].append(str(comb_val))

        # Flatten into the structure compute_metric expects
        all_keys = set(list(pred_dict.keys()) + list(gold.keys()))
        for qtype in all_keys:
            if qtype in ['id', 'context', 'is_mult']: continue
            formatted_instances.append({
                'id': instance_id, 'type': qtype,
                'predictions': pred_dict.get(qtype, []),
                'golds': gold.get(qtype, [])
            })
            
    return compute_metric(formatted_instances)

def print_phee_results(res, title):
    print(f"\n[{title}]")
    print_map = [
        ("Main-Args ", "CLS_MainArgs"),
        ("Sub-Args  ", "CLS_SubArgs"),
        ("Overall   ", "CLS_Overall")
    ]
    for label, key in print_map:
        em = res.get(f"{key}_EM_F1")
        tok = res.get(f"{key}_MICRO_F1")
        if em is not None:
             print(f"{label} | EM F1: {em:6.2f} | Token F1: {tok:6.2f}")
