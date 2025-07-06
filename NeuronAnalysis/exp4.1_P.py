# exp4_run_and_save.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from data_loader import load_all_datasets
from utils import extract_correct_answer, extract_answer
from model_utils import load_llm


MODEL_NAME      = "Qwen2.5-32B-Instruct"
HARD_ABLATION   = True
SELECTION_METHOD= "random"   # "role_diff", "random" or "all"
ROLE_PLAYING_PROMPTS = {
    "china_doctor": "You are a doctor in a Chinese tertiary hospital and mainly diagnose according to the guidelines of the Chinese Medical Association. Please answer this question.",
}
def call_selection_fn(fn, hidden_with, hidden_without=None):
    if fn.__name__ == "select_random_neurons":
        return fn(hidden_with)
    else:
        return fn(hidden_with, hidden_without)
# 选择神经元策略
def select_role_diff(hidden_with, hidden_without, top_k_ratio=0.01, top_layer_k=4):
    min_len = min(hidden_with[1].shape[1], hidden_without[1].shape[1])
    layer_scores = []
    per_layer_deltas = []
    for i in range(1, len(hidden_with)):
        delta = (hidden_with[i][:, :min_len, :] - hidden_without[i][:, :min_len, :])[0].abs().mean(dim=0)
        per_layer_deltas.append(delta)
        layer_scores.append(delta.mean().item())
    top_layers = np.argsort(layer_scores)[-top_layer_k:]
    all_scores, index_map = [], []
    for l in top_layers:
        delta = per_layer_deltas[l - 1]
        for i in range(delta.shape[0]):
            all_scores.append(delta[i].item())
            index_map.append((l - 1, i))
    top_k = int(len(all_scores) * top_k_ratio)
    top_indices = np.argsort(all_scores)[-top_k:]
    per_layer_neurons = {}
    for idx in top_indices:
        l, i = index_map[idx]
        per_layer_neurons.setdefault(l, []).append(i)
    return per_layer_neurons

def select_random_neurons(hidden_states, top_k_ratio=0.01):
    num_layers = hidden_states.shape[0]
    dim = hidden_states.shape[-1]
    per_layer_neurons = {}
    for l in range(num_layers - 1):
        k = max(1, int(dim * top_k_ratio))
        per_layer_neurons[l] = np.random.choice(dim, k, replace=False).tolist()
    return per_layer_neurons

SELECTION_FN = {
    "role_diff": select_role_diff,
    "random":    select_random_neurons,
}

def register_ablation_hooks(model, global_neurons):
    def get_transformer_layers(model):
        for attr in ['transformer','model','backbone']:
            if hasattr(model, attr):
                sub = getattr(model, attr)
                for layer_attr in ['h','layers','encoder.layer','decoder.layers']:
                    parts = layer_attr.split('.')
                    tmp = sub
                    try:
                        for p in parts:
                            tmp = getattr(tmp, p)
                        return tmp
                    except AttributeError:
                        continue
        raise RuntimeError("找不到 transformer 层")
    blocks = get_transformer_layers(model)
    handles = []
    for l, idxs in global_neurons.items():
        def make_hook(idxs):
            def hook(module, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                h = hidden.clone()
                if HARD_ABLATION:
                    h[:,:,idxs] = 0
                else:
                    h[:,:,idxs] *= 0.7
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook
        handles.append(blocks[l].register_forward_hook(make_hook(idxs)))
    return handles


def evaluate_sample(model, tokenizer, prompt, dataset_name, gold_answer, ablate_neurons=None):
    def get_device(model):
        if hasattr(model, "hf_device_map"):
            for k,v in model.hf_device_map.items():
                if "embed_tokens" in k or "wte" in k or "tok_embeddings" in k:
                    return torch.device(v)
        return next(model.parameters()).device
    device = get_device(model)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    handles = register_ablation_hooks(model, ablate_neurons) if ablate_neurons else []
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150,
                             do_sample=False, temperature=0.0,
                             top_p=0.95, repetition_penalty=1.0,
                             pad_token_id=tokenizer.eos_token_id)
    for h in handles: h.remove()
    output_text = tokenizer.decode(out[0], skip_special_tokens=True).strip()

    idx = output_text.find("The answer is:")
    answer_text = output_text[idx + len("The answer is:"):].strip() if idx >= 0 else output_text
    pred = extract_answer(answer_text, dataset_name)

    return int(pred.lower() == gold_answer.lower()) if gold_answer else 0

def run_experiment():
    datasets = load_all_datasets()
    model, tokenizer = load_llm(MODEL_NAME)
    model.eval()

    methods = list(SELECTION_FN.keys()) if SELECTION_METHOD=="all" else [SELECTION_METHOD]
    raw_records = []

    for method in methods:
        for role, role_prompt in ROLE_PLAYING_PROMPTS.items():
            for ds_name, ds in tqdm(datasets.items(), desc=f"[{method}] {role}"):
                for q_idx, sample in enumerate(tqdm(ds, desc="→ samples", leave=False)):
                    qtext = sample["question"]
                    options = sample.get("options",{})
                    opts = "\n".join(f"{k}: {v}" for k,v in options.items())
                    # 构造 prompt_with/prompt_without —— 保持原逻辑
                    prompt_with   = f"{role_prompt}\n\n{qtext}\n{opts}Please do not provide any explanation,Please choose one from A,B,C,D,E as your answer.Please give your answer directly and start with The answer is:"
                    prompt_without= f"{qtext}\n{opts}Please do not provide any explanation,Please choose one from A,B,C,D,E as your answer.Please give your answer directly and start with The answer is:"
                    # 选神经元
                    # 先拿 hidden_with/without
                    inputs_w   = tokenizer(prompt_with, return_tensors="pt", padding=True, truncation=True)
                    inputs_wo  = tokenizer(prompt_without,return_tensors="pt", padding=True, truncation=True)
                    device = next(model.parameters()).device
                    inputs_w  = {k: v.to(device) for k,v in inputs_w.items()}
                    inputs_wo = {k: v.to(device) for k,v in inputs_wo.items()}
                    with torch.no_grad():
                        hs_w  = model(**inputs_w,  output_hidden_states=True).hidden_states
                        hs_wo = model(**inputs_wo, output_hidden_states=True).hidden_states
                    hw  = torch.stack(hs_w)
                    hwo = torch.stack(hs_wo)
                    neurons = call_selection_fn(SELECTION_FN[method], hw, hwo)
                    # evaluate
                    gold = extract_correct_answer(sample, ds_name, options)
                    acc_before = evaluate_sample(model, tokenizer, prompt_with,  ds_name, gold)
                    acc_after  = evaluate_sample(model, tokenizer, prompt_with,  ds_name, gold, neurons)
                    # 记录两条
                    raw_records.append({
                        "model": MODEL_NAME,
                        "dataset": ds_name,
                        "role": role,
                        "selection_method": method,
                        "ablation_type": "hard" if HARD_ABLATION else "soft",
                        "ablated": 0,
                        "question_idx": q_idx,
                        "question": qtext,
                        "correct": acc_before
                    })
                    raw_records.append({
                        "model": MODEL_NAME,
                        "dataset": ds_name,
                        "role": role,
                        "selection_method": method,
                        "ablation_type": "hard" if HARD_ABLATION else "soft",
                        "ablated": 1,
                        "question_idx": q_idx,
                        "question": qtext,
                        "correct": acc_after
                    })

    # 最后：按角色分别保存 CSV
    OUT_DIR = "results/exp4_Sig"
    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(raw_records)
    for role, sub in df.groupby("role"):
        fname = f"{MODEL_NAME}_{role}_{SELECTION_METHOD}_{'hard' if HARD_ABLATION else 'soft'}.csv"
        sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print("✅ Saved raw per-sample CSVs to", OUT_DIR)

if __name__ == "__main__":
    run_experiment()
