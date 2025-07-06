
import os, json, re
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import torch, numpy as np
from tqdm import tqdm
from data_loader import load_all_datasets
from utils import extract_correct_answer
from model_utils import load_llm

MODEL_NAME       = "Qwen2.5-32B-Instruct"
HARD_ABLATION    = True
SELECTION_METHOD = "role_diff"  # or "random" or "all"

ROLE_PLAYING_PROMPTS = {
"med_student": "You are a third-year medical student currently learning medical knowledge and referring to standard textbooks. Please answer this question.",
"resident": "You are a resident doctor who already has some clinical experience, but still refers to the opinions of senior doctors. Please answer this question.",
"expert_doctor": "You are an experienced chief physician with 30 years of clinical experience and has published research papers in many fields. Please answer this question.",
"us_doctor": "You are an American doctor who is accustomed to referring to Mayo Clinic Guidelines and NEJM papers. Please answer this question.",
"china_doctor": "You are a doctor in a Chinese tertiary hospital and mainly diagnose according to the guidelines of the Chinese Medical Association. Please answer this question.",
"emergency doctor": "You are an emergency doctor, who is good at making quick decisions and usually relies on intuition to judge the condition. Please answer this question.",
"surgeon": "You are a surgeon, who is good at surgical operations and usually considers the feasibility of surgical treatment. Please answer this question.",
}

from NeuronAnalysis.exp5cross import call_selection_fn, SELECTION_FN, register_ablation_hooks, evaluate_sample,move_inputs_to_model_start_device 

def run_and_save():
    datasets = load_all_datasets()
    model, tokenizer = load_llm(MODEL_NAME)
    model.eval()

    out_raw = os.path.join("results", "exp5_sig", "raw")
    os.makedirs(out_raw, exist_ok=True)

    directions = [
        ("med_student","resident"),("resident","med_student"),
    ]

    for src, tgt in directions:
        for ds_name, ds in tqdm(datasets.items(), desc=f"[{src}→{tgt}]"):
            rows = []
            for qidx, sample in enumerate(tqdm(ds, desc="samples", leave=False)):
                qtext = sample["question"]
                opts  = sample.get("options",{})
                opts_text = "\n".join(f"{k}: {v}" for k,v in opts.items())
                suffix ="Please do not provide any explanation,Please choose one from A,B,C,D,E as your answer,Please give your answer directly and start with The answer is:"

                prompt_src  = f"{ROLE_PLAYING_PROMPTS[src]}\n\n{qtext}\n{opts_text}{suffix}"
                prompt_plain= f"{qtext}\n{opts_text}{suffix}"

                inp_src = move_inputs_to_model_start_device(model, tokenizer(prompt_src, return_tensors="pt"))
                inp_pln = move_inputs_to_model_start_device(model, tokenizer(prompt_plain, return_tensors="pt"))
                with torch.no_grad():
                    hs_src  = model(**inp_src,  output_hidden_states=True).hidden_states
                    hs_pln  = model(**inp_pln, output_hidden_states=True).hidden_states
                    hw  = torch.stack(hs_src); hwo = torch.stack(hs_pln)
                neurons = call_selection_fn(SELECTION_FN[SELECTION_METHOD], hw, hwo)
                prompt_tgt = f"{ROLE_PLAYING_PROMPTS[tgt]}\n\n{qtext}\n{opts_text}{suffix}"
                gold = extract_correct_answer(sample, ds_name, opts)

                # evaluate before/after
                before = evaluate_sample(model, tokenizer, prompt_tgt, ds_name, gold)
                after  = evaluate_sample(model, tokenizer, prompt_tgt, ds_name, gold, neurons)
                drop   = int(before==1 and after==0)

                rows.append({
                    "model": MODEL_NAME,
                    "source_role": src,
                    "target_role": tgt,
                    "selection_method": SELECTION_METHOD,
                    "ablation": "hard" if HARD_ABLATION else "soft",
                    "dataset": ds_name,
                    "question_idx": qidx,
                    "question": qtext,
                    "correct_before": before,
                    "correct_after":  after,
                    "drop":        drop
                })

            fn = f"{MODEL_NAME}_{src}_{tgt}_{SELECTION_METHOD}_{'hard' if HARD_ABLATION else 'soft'}_{ds_name}.csv"
            path = os.path.join(out_raw, fn)
            import pandas as pd
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"Saved raw CSV → {path}")

if __name__=="__main__":
    run_and_save()
