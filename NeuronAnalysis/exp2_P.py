# exp2_run_and_save.py

import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_loader import load_all_datasets
from utils import get_neuron_activations2
from model_utils import load_llm

def compute_js(p, q, epsilon=1e-10):
    p_list = [float(x) for x in p]
    q_list = [float(x) for x in q]
    p_clip = [max(epsilon, min(v, 1.0)) for v in p_list]
    q_clip = [max(epsilon, min(v, 1.0)) for v in q_list]
    sum_p = sum(p_clip)
    sum_q = sum(q_clip)
    p_norm = [v/sum_p for v in p_clip]
    q_norm = [v/sum_q for v in q_clip]
    m = [(pi+qi)/2 for pi,qi in zip(p_norm, q_norm)]
    kl_p = sum(pi * math.log(pi/mi, 2) for pi, mi in zip(p_norm, m))
    kl_q = sum(qi * math.log(qi/mi, 2) for qi, mi in zip(q_norm, m))
    return math.sqrt((kl_p + kl_q) / 2)

def analyze_kl_divergence(dataset_name, dataset, model, tokenizer, prompt_key, prompt_text):
    rows = []
    for sample in tqdm(dataset, desc=f"[{prompt_key}] {dataset_name}", leave=False):
        qid  = sample.get("id", sample.get("question_id"))
        text = sample["question"]
        acts_no   = get_neuron_activations2(model, tokenizer, text, prompt=None)
        acts_with = get_neuron_activations2(model, tokenizer, text, prompt=prompt_text)
        for layer_idx, (vec_no, vec_w) in enumerate(zip(acts_no, acts_with)):
            js = compute_js(vec_no.flatten(), vec_w.flatten())
            rows.append({
                "dataset":      dataset_name,
                "question_id":  qid,
                "prompt_key":   prompt_key,
                "layer":        layer_idx,
                "js_value":     js
            })
    return rows

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    MODEL_NAME     = "Qwen2.5-32B-Instruct"
    RAW_DIR        = "results"
    SIG_RAW_DIR    = os.path.join(RAW_DIR, "exp0_Sig", "raw")
    SIG_SUMM_DIR   = os.path.join(RAW_DIR, "exp0_Sig", "summary")
    os.makedirs(SIG_RAW_DIR, exist_ok=True)
    os.makedirs(SIG_SUMM_DIR, exist_ok=True)

    # load data & model
    datasets  = load_all_datasets()
    model, tokenizer = load_llm(MODEL_NAME)

    # prepare conditions
    all_conditions = []
    BASELINE_PROMPT = "Please provide the most appropriate answer to the following medical question. Please answer this question."
    all_conditions.append(("baseline", BASELINE_PROMPT))

    RANDOM_PROMPTS = [
       "This is a sentence.",
       "Thisisasentence.",
       "Somethinghappened today.",
       "If you were a planet, how would you think?"
    ]
    for i, rp in enumerate(RANDOM_PROMPTS):
        all_conditions.append((f"random_{i}", rp))

    ROLE_PLAYING_PROMPTS = {
        "med_student":   "You are a third-year medical student currently learning medical knowledge and referring to standard textbooks. Please answer this question.",
        "resident":      "You are a resident doctor who already has some clinical experience, but still refers to the opinions of senior doctors. Please answer this question.",
        "expert_doctor": "You are an experienced chief physician with 30 years of clinical experience and has published research papers in many fields. Please answer this question.",
        "us_doctor":     "You are an American doctor who is accustomed to referring to Mayo Clinic Guidelines and NEJM papers. Please answer this question.",
        "china_doctor":  "You are a doctor in a Chinese tertiary hospital and mainly diagnose according to the guidelines of the Chinese Medical Association. Please answer this question.",
        "surgeon":       "You are a surgeon, who is good at surgical operations and usually considers the feasibility of surgical treatment. Please answer this question."
    }
    for key, txt in ROLE_PLAYING_PROMPTS.items():
        all_conditions.append((key, txt))

    # iterate over each condition
    for prompt_key, prompt_text in all_conditions:
        print(f"▶ Running condition: {prompt_key}")
        all_rows = []
        for ds_name, ds_data in datasets.items():
            all_rows.extend(
                analyze_kl_divergence(ds_name, ds_data, model, tokenizer, prompt_key, prompt_text)
            )
        df_raw = pd.DataFrame(all_rows)
        raw_csv = os.path.join(SIG_RAW_DIR, f"exp2_{prompt_key}_{MODEL_NAME}_raw.csv")
        df_raw.to_csv(raw_csv, index=False)
        print(f"✔ Saved raw results to {raw_csv}")
        df_summary = (
            df_raw
            .groupby(["dataset", "prompt_key", "layer"], as_index=False)
            .agg(js_mean=("js_value", "mean"))
        )
        df_summary["model"] = MODEL_NAME
        summ_csv = os.path.join(SIG_SUMM_DIR, f"exp2_{prompt_key}_{MODEL_NAME}_summary.csv")
        df_summary.to_csv(summ_csv, index=False)
        print(f"✔ Saved summary results to {summ_csv}\n")

    print("🎉 All data saved in:")
    print(f"  • Raw per-sample:    {SIG_RAW_DIR}")
    print(f"  • Per-layer summary: {SIG_SUMM_DIR}")

if __name__ == "__main__":
    main()
