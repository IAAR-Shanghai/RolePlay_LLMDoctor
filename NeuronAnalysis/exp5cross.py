import json
import os
import torch
import numpy as np
import re
from tqdm import tqdm
from data_loader import load_all_datasets
from utils import extract_correct_answer, extract_answer
from model_utils import load_llm
target_gpus = [0]

def get_model_device(model):
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        for name, dev in model.hf_device_map.items():
            if any(k in name for k in ['embed_tokens', 'tok_embeddings', 'wte']):
                return torch.device(dev)
        return torch.device('cuda:0')
    if isinstance(model, torch.nn.DataParallel):
        return next(model.module.parameters()).device
    return next(model.parameters()).device
torch.manual_seed(111)
torch.cuda.manual_seed(132)
np.random.seed(4314)
MODEL_NAME = "Qwen2.5-7B-Instruct"
HARD_ABLATION = True
SELECTION_METHOD = "role_diff"  


def call_selection_fn(fn, hidden_with, hidden_without=None):
    if fn.__name__ == "select_random_neurons":
        return fn(hidden_with)
    else:
        return fn(hidden_with, hidden_without)


# --------------------- 神经元选择策略 ---------------------
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
    top_k_ratio = float(top_k_ratio)
    num_layers = hidden_states.shape[0]
    dim = int(hidden_states.shape[-1])

    per_layer_neurons = {}
    for l in range(num_layers - 1):
        k = max(1, int(dim * top_k_ratio))
        indices = np.random.choice(dim, k, replace=False).tolist()
        per_layer_neurons[l] = indices
    return per_layer_neurons




SELECTION_FN = {
    "role_diff": select_role_diff,
    "random": select_random_neurons,
}

# --------------------- 角色 Prompt ---------------------
ROLE_PLAYING_PROMPTS = {
"med_student": "You are a third-year medical student currently learning medical knowledge and referring to standard textbooks. Please answer this question.",
"resident": "You are a resident doctor who already has some clinical experience, but still refers to the opinions of senior doctors. Please answer this question.",
"expert_doctor": "You are an experienced chief physician with 30 years of clinical experience and has published research papers in many fields. Please answer this question.",
"us_doctor": "You are an American doctor who is accustomed to referring to Mayo Clinic Guidelines and NEJM papers. Please answer this question.",
"china_doctor": "You are a doctor in a Chinese tertiary hospital and mainly diagnose according to the guidelines of the Chinese Medical Association. Please answer this question.",
"emergency doctor": "You are an emergency doctor, who is good at making quick decisions and usually relies on intuition to judge the condition. Please answer this question.",
"surgeon": "You are a surgeon, who is good at surgical operations and usually considers the feasibility of surgical treatment. Please answer this question.",
}

# --------------------- Hook ---------------------
def register_ablation_hooks(model, global_neurons):
    def get_transformer_layers(model):
        for attr in ['transformer', 'model', 'backbone']:
            if hasattr(model, attr):
                sub = getattr(model, attr)
                for layer_attr in ['h', 'layers', 'encoder.layer', 'decoder.layers']:
                    parts = layer_attr.split('.')
                    temp = sub
                    try:
                        for part in parts:
                            temp = getattr(temp, part)
                        return temp
                    except AttributeError:
                        continue
        raise ValueError("找不到 transformer 层")

    handles = []
    all_blocks = get_transformer_layers(model)

    for l, indices in global_neurons.items():
        def make_hook(indices):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[:, :, indices] = 0 if HARD_ABLATION else hidden[:, :, indices] * 0.7
                    return (hidden,) + output[1:]
                else:
                    output = output.clone()
                    output[:, :, indices] = 0 if HARD_ABLATION else output[:, :, indices] * 0.7
                    return output
            return hook_fn

        block = all_blocks[l]
        handles.append(block.register_forward_hook(make_hook(indices)))
    return handles

# --------------------- Evaluate ---------------------
def evaluate_sample(model, tokenizer, prompt, dataset_name, gold_answer, ablate_neurons=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(f"cuda:{target_gpus[0]}")
    handles = register_ablation_hooks(model, ablate_neurons) if ablate_neurons else []
    with torch.no_grad():
        output_ids = model.generate(
            # input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            **inputs,
            max_new_tokens=200, do_sample=False, temperature=0.0, top_p=1.0,
            repetition_penalty=1.0, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    for h in handles: h.remove()
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    idx = output_text.find("The answer is:")
    answer_text = output_text[idx + len("The answer is:"):].strip() if idx >= 0 else output_text
    pred = extract_answer(answer_text, dataset_name)
    # print("🧠 输出：", output_text)
    # print("📌 提取：", pred, "| 正确：", gold_answer)
    return int(pred.lower() == gold_answer.lower()) if pred != "UNKNOWN" and gold_answer else 0
    return int(pred.lower() == gold_answer.lower()) if  gold_answer else 0

def move_inputs_to_model_start_device(model, tokenizer_output):
        if hasattr(model, "hf_device_map"):
            # 遍历 hf_device_map 找到第一个 embedding 层所在的 GPU
            for layer_name in model.hf_device_map:
                if any(key in layer_name for key in ["embed_tokens", "wte", "tok_embeddings"]):
                    device = torch.device(model.hf_device_map[layer_name])
                    break
            else:
                device = torch.device("cuda:0")  # fallback
        else:
            device = next(model.parameters()).device  # 单卡模型 fallback

        # 把所有张量都移动到该卡
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in tokenizer_output.items()}
# --------------------- 主函数 ---------------------
def run_cross_role_bidirectional_experiment():
   

    datasets = load_all_datasets()
    model, tokenizer= load_llm(MODEL_NAME)
    model.eval()


    method = "role_diff"

    RESULTS_PATH = f"results/cross_role_bidirectional_{MODEL_NAME}_{method}_{'hard' if HARD_ABLATION else 'soft'}_0501_Student3Resident2_medqa.json"

    results = []

    for direction in [ 
                      ("med_student", "resident"),("resident", "med_student"),
                      ]:
        source_role, target_role = direction

        for dataset_name, dataset in tqdm(datasets.items(), desc=f"[{source_role}→{target_role}]"):
            correct_before = 0
            correct_after = 0
            total = 0

            for sample in tqdm(dataset, desc=f"→ {dataset_name}-{MODEL_NAME}-{HARD_ABLATION}"):
                question = sample["question"]
                options = sample.get("options", {})
                options_text = "\n".join([f"{k}: {v}" for k, v in options.items()]) if options else ""

                suffix = "Please do not provide any explanation,Please choose one from yes, no, maybe as your answer,lease provide the response begin with The answer is:" if dataset_name == "pubmedqa" else "Please do not provide any explanation,Please choose one from A,B,C,D,E as your answer,Please give your answer directly and start with The answer is:"

                # Step 1: 获取 source_role 的 role_diff neuron
                prompt_source = f"{ROLE_PLAYING_PROMPTS[source_role]}\n\n{question}\n{options_text}{suffix}"
                prompt_plain = f"{question}\n{options_text}{suffix}"
                inputs_source = move_inputs_to_model_start_device(model, tokenizer(prompt_source, return_tensors="pt"))
                inputs_plain = move_inputs_to_model_start_device(model, tokenizer(prompt_plain, return_tensors="pt"))

                with torch.no_grad():
                    hs_with = model(**inputs_source, output_hidden_states=True).hidden_states
                    hs_without = model(**inputs_plain, output_hidden_states=True).hidden_states
                    hidden_with = torch.stack(hs_with)
                    hidden_without = torch.stack(hs_without)

                source_neurons = call_selection_fn(SELECTION_FN[method], hidden_with, hidden_without)
                gold = extract_correct_answer(sample, dataset_name, options)

                # Step 2: 用 target_role 进行推理，并剪掉 source_role 的 neuron
                prompt_target = f"{ROLE_PLAYING_PROMPTS[target_role]}\n\n{question}\n{options_text}{suffix}"

                acc_before = evaluate_sample(model, tokenizer, prompt_target, dataset_name, gold)
                acc_after = evaluate_sample(model, tokenizer, prompt_target, dataset_name, gold, source_neurons)

                correct_before += acc_before
                correct_after += acc_after
                total += 1

            acc_before_ratio = correct_before / total if total else 0
            acc_after_ratio = correct_after / total if total else 0

            print(f"\n🔄 Cross Transfer [{source_role} → {target_role}] | Dataset: {dataset_name}")
            print(f"   Acc Before: {acc_before_ratio:.4f} | Acc After: {acc_after_ratio:.4f} | Δ: {acc_before_ratio - acc_after_ratio:.4f}")

            results.append({
                "source_role": source_role,
                "target_role": target_role,
                "dataset": dataset_name,
                "model": MODEL_NAME,
                "selection_method": method,
                "ablation": "hard" if HARD_ABLATION else "soft",
                "accuracy_before": acc_before_ratio,
                "accuracy_after": acc_after_ratio,
                "accuracy_diff": acc_before_ratio - acc_after_ratio
            })

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Cross-role 双向迁移实验完成，结果保存至：{RESULTS_PATH}")

if __name__ == "__main__":
    run_cross_role_bidirectional_experiment()
