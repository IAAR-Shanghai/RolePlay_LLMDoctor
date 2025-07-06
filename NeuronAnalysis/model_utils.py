import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def load_llm(MODEL_NAME):
    model_path = f"XXXXX" # 替换为实际模型路径
    is_llama = "llama" in MODEL_NAME.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=not is_llama, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def get_model_size(name):
        name = name.lower()
        if "72b" in name: return 72
        if "65b" in name: return 65
        if "32b" in name: return 32
        if "14b" in name: return 14
        return 7  # 默认

    model_size = get_model_size(MODEL_NAME)

    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_gpus:
        gpu_indices = list(range(len(visible_gpus.split(","))))
    else:
        gpu_indices = list(range(torch.cuda.device_count()))
    print(f"🖥️ 可用 GPU: {gpu_indices}")

    print(f"📦 使用{len(gpu_indices)}张显卡加载 {MODEL_NAME}（{model_size}B）")
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True
    )

    model.eval()
    return model, tokenizer
