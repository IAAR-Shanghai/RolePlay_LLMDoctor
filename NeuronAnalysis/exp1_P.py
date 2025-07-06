import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from data_loader import load_all_datasets
from utils import get_neuron_activations, extract_correct_answer, extract_answer
from model_utils import load_llm

MODEL_NAME = "deepseek-r1"
RESULTS_PATH = f"results/exp1_accuracy_{MODEL_NAME}_medqa.json"

# --------------------- Prompt 列表 ---------------------
ROLE_PLAYING_PROMPTS = {
"med_student": "You are a third-year medical student currently learning medical knowledge and referring to standard textbooks. Please answer this question.",
# "resident": "You are a resident doctor who already has some clinical experience, you always use general clinical knowledge and the guidence of expert doctor to make judgement . Please answer this question.",
# "emergency doctor": "You are an emergency doctor, who is good at making quick decisions and usually relies on intuition to judge the condition. Please answer this question.",
# "surgeon": "You are a surgeon, who is good at surgical operations and usually considers the feasibility of surgical treatment. Please answer this question.",
# "expert_doctor": "You are an experienced chief physician with 30 years of clinical experience and has published research papers in many fields. Please answer this question.",
# "us_doctor": "You are an American doctor who is accustomed to referring to Mayo Clinic Guidelines and NEJM papers. Please answer this question.",
# "china_doctor": "You are a doctor in a Chinese tertiary hospital and mainly diagnose according to the guidelines of the Chinese Medical Association. Please answer this question.",

}
BASELINE_PROMPT = "Please provide the most appropriate answer to the following medical question."
RANDOM_PROMPTS = ["This is a sentence."]

def move_inputs_to_model_start_device(model, tokenizer_output):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for layer_name, dev in model.hf_device_map.items():
            if any(key in layer_name for key in ["embed_tokens", "wte", "tok_embeddings"]):
                device = torch.device(dev)
                break
        else:
            device = torch.device("cuda:0")
    else:
        device = next(model.parameters()).device
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tokenizer_output.items()}


def evaluate_accuracy(model, tokenizer, dataset, dataset_name, prompt_dict, detailed=False):
    acc_results = {}
    for prompt_key, prompt_list in prompt_dict.items():
        acc_list = []
        individual_results = [] if detailed else None
        for prompt in prompt_list:
            correct = 0
            total = 0
            for sample in tqdm(dataset, desc=f"Evaluating {dataset_name} with {prompt_key}"):
                question = sample["question"]
                options = sample.get("options", {})
                full_question = (
                    question if not options else
                    question + "\noptions:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()]) +
                    " Please give your answer directly and start with The answer is:"
                )
                formatted_prompt = prompt + "\n\n" + full_question

                # 分词并移动到正确设备
                inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = move_inputs_to_model_start_device(model, inputs)

                #针对于开源模型：
                if model!="gpt-4o" and model!="deepseek-r1":
                    with torch.no_grad(), autocast():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                            repetition_penalty=1.0,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    # 清理缓存
                    torch.cuda.empty_cache()

                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                else:
                    import json
                    import requests
                    data = {
                        "model": model, #填入想要使用的模型
                        "messages": [{"role": "user", "content": formatted_prompt}],
                        "temperature": 0.1
                    }
                    key = 'sk-9kQdDNeVljnQTc4tBDulHqky6TG5lWVhwdRIfBR8MMnJucPe'  #填入在网页中复制的令牌
                    headers = {
                            'Authorization': 'Bearer {}'.format(key),
                            'Content-Type': 'application/json',
                        }
                    output_text = requests.request("POST", "http://123.129.219.111:3000/v1/chat/completions", headers=headers, data=json.dumps(data),timeout=300)

                    # print(output_text)
                idx = output_text.find("The answer is:")
                answer_text = output_text[idx + len("The answer is:"):].strip() if idx >= 0 else output_text
                # print(answer_text)
                pred = extract_answer(answer_text, dataset_name)
                gold = extract_correct_answer(sample, dataset_name, options)
                if pred == gold and pred is not None:
                    correct += 1
                total += 1
                if detailed:
                        individual_results.append({
                                            "dataset": dataset_name,
                                            "question_id": sample.get("id", f"{dataset_name}_{total}"),  # 若无ID，则用索引代替
                                            "question": question,
                                            "role": prompt_key,
                                            "correct": int(pred == gold and pred is not None),
                                            "prediction": pred,
                                            "gold": gold,
                                            "model_output": answer_text.strip()
                                        })
            acc = correct / total if total > 0 else 0
            acc_list.append(acc)
            


        summary = {"min": min(acc_list), "max": max(acc_list), "mean": np.mean(acc_list)}
        if detailed:
            summary["individual"] = individual_results
        acc_results[prompt_key] = summary
    return acc_results

import json
import requests
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



def evaluate_accuracy2(model, dataset, dataset_name, prompt_dict, detailed=False, max_workers=15):
    def call_api(model, formatted_prompt, api_key, timeout=1200):
        """封装API调用函数"""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": formatted_prompt}],
            "temperature": 0.1
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
        try:
            response = requests.post(
                "http://123.129.219.111:3000/v1/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=timeout
            )
            return response.json()
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            return None

    def process_api_response(response, dataset_name):
        """处理API响应并提取答案"""
        if not response or 'choices' not in response or not response['choices']:
            return "UNKNOWN", "UNKNOWN"
        
        message = response['choices'][0]['message']
        content = message.get('content', '')
        
        idx = content.find("The answer is:")
        answer_text = content[idx + len("The answer is:"):].strip() if idx >= 0 else content
        pred = extract_answer(answer_text, dataset_name)
        
        return pred, answer_text

    def evaluate_sample(args):
        """处理单个样本的评估任务"""
        sample, prompt, prompt_key, dataset_name, model, api_key, options = args
        question = sample["question"]
        full_question = (
            question if not options else
            question + "\noptions:\n" + "\n".join([f"{k}: {v}" for k, v in options.items()]) +
            " Please give your answer directly and start with The answer is:"
        )
        formatted_prompt = prompt + "\n\n" + full_question
        
        if model not in ["gpt-4o", "deepseek-r1"]:
            print("Using open-source model:", model)
            # 这里处理开源模型的逻辑
            pred, answer_text = "UNKNOWN", "UNKNOWN"  # 替换为实际处理逻辑
        else:
            response = call_api(model, formatted_prompt, api_key)
            pred, answer_text = process_api_response(response, dataset_name)
        
        gold = extract_correct_answer(sample, dataset_name, options)
        is_correct = int(pred == gold and pred is not None)
        
        result = {
            "dataset": dataset_name,
            "question_id": sample.get("id", f"{dataset_name}_{hash(question)}"),
            "question": question,
            "role": prompt_key,
            "correct": is_correct,
            "prediction": pred,
            "gold": gold,
            "model_output": answer_text.strip()
        }
        
        return is_correct, result if detailed else is_correct
    acc_results = {}
    api_key = 'XXXX'  # API
    
    for prompt_key, prompt_list in prompt_dict.items():
        acc_list = []
        individual_results = [] if detailed else None
        
        for prompt in prompt_list:
            correct = 0
            total = 0
            
            # 准备任务参数
            tasks = [
                (sample, prompt, prompt_key, dataset_name, model, api_key, sample.get("options", {}))
                for sample in dataset
            ]
            
            # 使用线程池并发处理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(evaluate_sample, task)
                    for task in tasks
                ]
                
                # 使用tqdm显示进度
                for future in tqdm(as_completed(futures), 
                                 total=len(futures),
                                 desc=f"Evaluating {dataset_name} with {prompt_key} in {model}"):
                    try:
                        result = future.result()
                        if detailed:
                            is_correct, individual_result = result
                            individual_results.append(individual_result)
                        else:
                            is_correct = result
                            
                        correct += is_correct
                        total += 1
                    except Exception as e:
                        print(f"处理样本时出错: {str(e)}")
                        total += 1
            
            acc = correct / total if total > 0 else 0
            acc_list.append(acc)
        
        summary = {
            "min": min(acc_list),
            "max": max(acc_list),
            "mean": np.mean(acc_list)
        }
        if detailed:
            summary["individual"] = individual_results
        acc_results[prompt_key] = summary
    
    return acc_results

# --------------------- 主流程 ---------------------
def run_experiment1():
    datasets = load_all_datasets()
    if MODEL_NAME!="gpt-4o" and MODEL_NAME!="deepseek-r1":
        model, tokenizer = load_llm(MODEL_NAME)
        # 如果有多张 GPU，可启用 DataParallel
        if torch.cuda.device_count() > 1 and not hasattr(model, 'hf_device_map'):
            model = torch.nn.DataParallel(model)
        model.eval()

    results = []
    for ds_name, ds in tqdm(datasets.items(), desc="Processing Datasets"):
        prompt_groups = {
            **{key: [value] for key, value in ROLE_PLAYING_PROMPTS.items()},
            # "baseline": [BASELINE_PROMPT],
            # "random": RANDOM_PROMPTS,
            # "unrelated": UNRELATED_PROMPTS,
        }
        if MODEL_NAME!="gpt-4o" and MODEL_NAME!="deepseek-r1":
            acc_data = evaluate_accuracy(model, tokenizer, ds, ds_name, prompt_groups, detailed=True)#记录细节
        else:
            acc_data = evaluate_accuracy2(MODEL_NAME, ds, ds_name, prompt_groups, detailed=True,max_workers=4)#记录细节
        for p_type, vals in acc_data.items():
            if vals.get("individual"):
                results.extend(vals["individual"])
                role_results = vals["individual"]
                df = pd.DataFrame(role_results)
                print(df)
                # 创建输出目录
                os.makedirs("results/exp1_Sig/", exist_ok=True)
                # 将每个角色分别写入 Excel 文件
                filename = f"results/exp1_Sig/Sig1_{MODEL_NAME}_{ds_name}_{p_type}_details.xlsx"
                df.to_excel(filename, index=False)
                print(f"✅ 已保存：{filename}")
            else:
                results.append({"dataset": ds_name, "prompt_type": p_type, **{k: vals[k] for k in ["min","max","mean"]}})

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"🎯 实验完成！结果已保存到 {RESULTS_PATH}")

if __name__ == "__main__":
    run_experiment1()
