import json
import random
import os

# Constants for argument order in tuples
SUBJECT_FIELDS = ["subject", "age", "gender",
                  "population", "race", "subject disorder"]
TREATMENT_FIELDS = [
    "treatment", "drug", "route", "dosage", "time elapsed",
    "duration", "frequency", "combination drug", "treatment disorder"
]
ORDER_MAIN = ["event_type", "subject", "treatment", "effect"]
EVENT_TYPES = ["Adverse_event", "Potential_therapeutic_event"]

LLM_DICT = {
    "google/gemma-4-31B": {
        "vllm": "google/gemma-4-31B",
        "file_path": "google_gemma-4-31B",
        "nice_print": "Gemma 4 (31B)",
    }
}

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOLD_PATH_TEMPLATE = os.path.join(BASE_DIR, "../phee-with-chatgpt/data/converted_data/text2spotasoc/event/phee2_cross{split}/test.json")
PHEE_METRIC_DIR = os.path.join(BASE_DIR, "../phee-with-chatgpt")

EVENT_TYPE_MAP = {
    "adverse event": "Adverse_event",
    "potential therapeutic event": "Potential_therapeutic_event"
}

ARG_TYPE_MAP = {
    "subject": "Subject",
    "treatment": "Treatment",
    "effect": "Effect",
    "age": "Subject.Age",
    "gender": "Subject.Gender",
    "race": "Subject.Race",
    "population": "Subject.Population",
    "subject disorder": "Subject.Disorder",
    "drug": "Treatment.Drug",
    "dosage": "Treatment.Dosage",
    "route": "Treatment.Route",
    "duration": "Treatment.Duration",
    "frequency": "Treatment.Freq",
    "time elapsed": "Treatment.Time_elapsed",
    "treatment disorder": "Treatment.Disorder",
    "combination drug": "Combination.Drug"
}

MISSING_VALUES = ["null", "none", "n/a"]

# Constants for attribute descriptions
ATTRIBUTE_DESCRIPTIONS = {
    "event_type": "'event_type': The type of medical event (Adverse_event or Potential_therapeutic_event).",
    "subject": "'subject': The word or phrase identifying the patient(s) involved in the event.",
    "treatment": "'treatment': The word or phrase describing the medical treatment or medication given to the subject.",
    "effect": "'effect': The word or phrase describing the medical result or outcome observed.",
    "age": "'age': The age of the subject.",
    "gender": "'gender': The gender of the subject.",
    "population": "'population': Number of subjects involved.",
    "race": "'race': Race or nationality.",
    "subject disorder": "'disorder': Disorders the subject suffers from.",
    "drug": "'drug': The specific drug name.",
    "route": "'route': Administration route (e.g., oral, intravenous).",
    "dosage": "'dosage': Amount of drug administered.",
    "time elapsed": "'time_elapsed': Time since therapy start/end until event.",
    "duration": "'duration': Duration of treatment.",
    "frequency": "'frequency': How often the drug is taken.",
    "combination drug": "'combination_drug': Other drugs used in combination.",
    "treatment disorder": "'disorder': The target disease for the treatment."
}


def get_arg(args, type_name):
    """Extracts all texts for a given type and joins them with ' , ', if present."""
    texts = [a["text"] for a in args if a["type"] == type_name]
    return " , ".join(texts) if texts else "null"


def format_main(event, order=ORDER_MAIN):
    """Converts an event into the 'main' format tuple based on custom order."""
    args = event.get("args", [])
    data = {
        "event_type": event.get("type", "null").replace(" ", "_").capitalize(),
        "subject": get_arg(args, "subject"),
        "treatment": get_arg(args, "treatment"),
        "effect": get_arg(args, "effect")
    }
    return tuple(data.get(field, "null") for field in order)


def format_sub(event, subject_order=SUBJECT_FIELDS, treatment_order=TREATMENT_FIELDS,
               order_main=ORDER_MAIN):
    """Converts an event into the 'sub' format dictionary based on custom order."""
    args = event.get("args", [])

    # Pre-calculate all available parts
    parts = {
        "event_type": event.get("type", "null").replace(" ", "_").capitalize(),
        "subject": tuple(get_arg(args, field) for field in subject_order),
        "treatment": tuple(get_arg(args, field) for field in treatment_order),
        "effect": get_arg(args, "effect")
    }

    # Build dictionary in the order specified by order_main
    return {field: parts.get(field, "null") for field in order_main}


def get_prompt(format_type, few_shot_examples, sentence,
               order_main=ORDER_MAIN,
               order_subject=SUBJECT_FIELDS,
               order_treatment=TREATMENT_FIELDS):
    """Generates a prompt by populating a template with descriptions and few-shot examples."""
    template_path = os.path.join(BASE_DIR, f"prompts/{format_type}.md")
    with open(template_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    few_shots = [f"Text: {text.strip()}\nEvents: {str(events).strip()}" for text,
                 events in few_shot_examples]
    few_shot_str = "".join([s + "\n" for s in few_shots])

    replacements = {"{{few_shot_examples}}": few_shot_str.strip(),
                    "{{sentence}}": sentence.strip()}
    if format_type == "main":
        desc = "\n".join(
            [f"- {ATTRIBUTE_DESCRIPTIONS[f]}" for f in order_main])
        fmt = "(" + ", ".join([f"'{f}'" for f in order_main]) + ")"
        replacements["{{main_desc}}"] = desc
        replacements["{{main_format}}"] = fmt
    elif format_type == "sub":
        sub_desc = "\n   - ".join([ATTRIBUTE_DESCRIPTIONS[f]
                                  for f in order_subject])
        treat_desc = "\n   - ".join([ATTRIBUTE_DESCRIPTIONS[f]
                                    for f in order_treatment])
        sub_fmt = "(" + ", ".join([f"'{f}'" for f in order_subject]) + ")"
        treat_fmt = "(" + ", ".join([f"'{f}'" for f in order_treatment]) + ")"

        # Build example dict string to show the model the key order
        main_fmt_parts = []
        for key in order_main:
            if key == "subject":
                main_fmt_parts.append(f"\"subject\": {sub_fmt}")
            elif key == "treatment":
                main_fmt_parts.append(f"\"treatment\": {treat_fmt}")
            else:
                main_fmt_parts.append(f"\"{key}\": \"...\"")

        main_fmt = "[{" + ", ".join(main_fmt_parts) + "}, ...]"

        replacements["{{subject_desc}}"] = sub_desc
        replacements["{{treatment_desc}}"] = treat_desc
        replacements["{{subject_format}}"] = sub_fmt
        replacements["{{treatment_format}}"] = treat_fmt
        replacements["{{sub_format_example}}"] = main_fmt
    for key, val in replacements.items():
        prompt = prompt.replace(key, val)
    
    # Ensure the prompt ends exactly with ": [" to guide the model into list format
    # This prevents the model from generating whitespace or the bracket itself
    return prompt.strip().rstrip(":") + ": ["


def load_data(path):
    """Loads all examples from a jsonl file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_shots(n_shot, cross_split, seed=42, format="raw"):
    """Loads n_shot random examples from the train.json of the specified cross_split."""
    path = os.path.join(BASE_DIR, f"../phee-with-chatgpt/data/converted_data/text2spotasoc/event/phee2_cross{cross_split}/train.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found.")
    with open(path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    examples.sort(key=lambda x: x.get('text_id', ''))
    random.seed(seed)
    random.shuffle(examples)
    selected = examples[:n_shot]
    if format == "raw":
        return selected
    formatted_results = []
    for ex in selected:
        doc_events = []
        for event in ex.get("event", []):
            if format == "main":
                doc_events.append(format_main(event))
            elif format == "sub":
                doc_events.append(format_sub(event))
        formatted_results.append(doc_events)
    return formatted_results


def save_llm_results(predictions, results, gpu_stats, filename, task_type, output_dir="results"):
    """
    Saves LLM results to .txt (raw text) and .json (logprobs + metadata) files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Save raw text to .txt
    txt_path = os.path.join(output_dir, f"{filename}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            # vLLM predictions might contain newlines, we strip them to keep one line per example
            f.write(pred.replace("\n", " ").strip() + "\n")

    # 2. Save detailed logprobs and GPU stats to .json
    json_path = os.path.join(output_dir, f"{filename}.json")
    
    detailed_data = {
        "task": task_type,
        "gpu_statistics": gpu_stats,
        "results": []
    }

    for i, res in enumerate(results):
        # Use existing prediction if available, otherwise build from raw output
        if i < len(predictions):
            prediction_text = predictions[i].strip()
        else:
            prediction_text = res.outputs[0].text.strip()
        
        # Ensure correct list format for 'main' and 'sub' tasks
        if task_type in ["main", "sub"] and prediction_text:
            if not prediction_text.startswith("["):
                prediction_text = "[" + prediction_text
            
            if task_type == "main" and not prediction_text.endswith("]") and not prediction_text.endswith(")]"):
                prediction_text += ")]"
            elif task_type == "sub" and not prediction_text.endswith("]") and not prediction_text.endswith("}]"):
                prediction_text += "}]"

        logprobs = res.outputs[0].logprobs # List of dicts per token
        
        token_logprobs = []
        if logprobs:
            for token_data in logprobs:
                # token_data is a dict where keys are token IDs and values are Logprob objects
                # We extract the top-5 for the current token position
                current_token_pos = []
                # Sort by logprob value descending
                sorted_logprobs = sorted(token_data.items(), key=lambda x: x[1].logprob, reverse=True)
                
                for tid, lp_obj in sorted_logprobs[:5]:
                    lp_val = lp_obj.logprob
                    if lp_val == float('-inf'):
                        lp_val = "-inf"
                    else:
                        lp_val = round(float(lp_val), 3)

                    current_token_pos.append({
                        "token": lp_obj.decoded_token,
                        "logprob": lp_val
                    })
                token_logprobs.append(current_token_pos)

        detailed_data["results"].append({
            "text": prediction_text,
            "token_logprobs": token_logprobs
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {txt_path} and {json_path}")


import ast

def parse_llm_output(text):
    """Parses LLM output string as a Python literal (list of tuples or dicts)."""
    text = text.strip()
    # Add leading bracket if missing (vLLM output starts after the prompt's "[")
    if text and not text.startswith("["):
        text = "[" + text
        
    try:
        # vLLM output might contain trailing noise if not perfectly stopped, though we handle it
        # ast.literal_eval is safer than eval()
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return parsed
        return []
    except Exception as e:
        # print(f"Error parsing LLM output: {e}\nText: {text}")
        return []
