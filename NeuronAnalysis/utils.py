import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.spatial.distance import jensenshannon
import re

def get_neuron_activations(model, tokenizer, question, prompt=None):
    if prompt:
            input_text = f"{prompt}\n{question}"
    else:
            input_text = question
    # print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        
    with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    # stack 得到 (L, B, S, H) 的 tensor
    acts_t = torch.stack(hidden_states)
    # 在 PyTorch 上对最后一维求平均 -> (L, B, S)
    acts_t = acts_t.mean(dim=-1)
    activations = acts_t.cpu().numpy()

    L, B, S = activations.shape
    means = []
    for layer in range(L):
        mat = activations[layer].tolist()
        total = 0.0
        for row in mat:
            total += sum(row)
        means.append(total / (B * S))

    return np.array(means)

def get_neuron_activations2(model, tokenizer, question, prompt=None):
    if prompt:
            input_text = f"{prompt}\n{question}"
    else:
            input_text = question
    # print(input_text)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    # stack 得到 (L, B, S, H)
    acts_t = torch.stack(hidden_states)
    # 只对 batch_size 和 seq_length 两个维度做平均，保留 hidden_dim
    acts_t = acts_t.mean(dim=1).mean(dim=1)   # 先 dim=1(B)，再 dim=1(S) -> shape (L, H)
    return acts_t.cpu().numpy()               # numpy array (L, H)
def get_neuron_activations_core(model, tokenizer, question, prompt=None):
    if prompt:
        input_text = f"{prompt}\n{question}"
    else:
        input_text = question

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # (num_layers, batch_size, seq_length, hidden_dim)

    activations = torch.stack(hidden_states).squeeze(1).cpu().numpy()  # (num_layers, seq_length, hidden_dim)
    
    return activations


from scipy.spatial.distance import jensenshannon
import numpy as np

def safe_js_divergence(p, q, epsilon=1e-10):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    # 避免零概率
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    # 重新归一化，确保是概率分布
    p /= p.sum()
    q /= q.sum()

    return jensenshannon(p, q)

def neuron_importance_analysis(activations_no_prompt, activations_with_prompt):
    if activations_no_prompt is None or activations_with_prompt is None:
        print("[错误] 传入的神经元激活数据为空！")
        return None, None

    # 计算安全的 KL 散度 (JS 散度)
    try:
        kl_div = safe_js_divergence(activations_no_prompt.flatten(), activations_with_prompt.flatten())
    except ValueError as e:
        print(f"[错误] 计算 KL 散度失败: {e}")
        kl_div = None

    # 计算神经元差异最大的位置
    neuron_differences = np.abs(activations_no_prompt - activations_with_prompt)
    important_neurons = np.argsort(-neuron_differences.sum(axis=0))[:10]  # 取贡献度最大的 10 个神经元

    return kl_div, important_neurons


'''从真实数据集中提取答案'''
def extract_correct_answer(sample, dataset_name, options):
    if dataset_name == "mmlu":
        # print(sample)
        correct_answer = sample["answer"].strip().upper()
        # correct_answer = next((k for k, v in options.items() if v.strip().lower() == answer_value), None)

    elif dataset_name == "medmcqa":
        correct_answer = sample["answer"].strip().upper()  


    elif dataset_name == "medqa":
        answer_value = sample["answer"].strip().lower()
        correct_answer = next((k for k, v in options.items() if v.strip().lower() == answer_value), None)

    else:
        correct_answer = None  

    return correct_answer

'''大模型回答中提取答案'''
def extract_answer(output_text, dataset_name):
    output_text = output_text.strip().lower()

    
    # 尝试提取：答案是 A、正确答案为 B、the correct answer is C 等
    pattern = r"(答案是|答案为|正确答案是|正确答案为|The answer is|The answer is:|the correct answer is|answer is:|答案[:：]?)?\s*([A-E])\b"
    match = re.search(pattern, output_text, re.IGNORECASE)
    if match:
            return match.group(2).upper()

    # fallback：直接找首个 A-E
    match_simple = re.search(r"\b([A-E])\b", output_text, re.IGNORECASE)
    if match_simple:
            return match_simple.group(1).upper()

    # ---------- 提取失败 ---------- #
    print("⚠️ 提取失败，原始输出：", output_text)
    return "UNKNOWN"