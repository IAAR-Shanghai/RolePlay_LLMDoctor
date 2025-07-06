import json
import pandas as pd
import random
import os

class LocalDataset:
    """
    这是一个模拟 `datasets.Dataset` 的本地数据集类，支持 `.shuffle()` 和 `.select()`
    """
    def __init__(self, data):
        self.data = data

    def shuffle(self, seed=42):
        random.seed(seed)
        shuffled_data = self.data.copy()
        random.shuffle(shuffled_data)
        return LocalDataset(shuffled_data)

    def select(self, indices):
        selected_data = [self.data[i] for i in indices]
        return LocalDataset(selected_data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def load_medqa(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    formatted_data = [{"question": item["question"],"options": item["options"], "answer": item["answer"]} for item in data]
    return LocalDataset(formatted_data)

def load_medmcqa(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]

    def format_medmcqa(item):
        num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        answer = ''.join(num_to_letter[int(digit)] for digit in str(item['cop'])) if 'cop' in item else None
        options = {
            "A": item.get('opa', ""),
            "B": item.get('opb', ""),
            "C": item.get('opc', ""),
            "D": item.get('opd', "")
        }
        return {"question": item["question"], "options": options, "answer": answer}

    filtered_data = [format_medmcqa(item) for item in data if item.get('choice_type') == 'single']

    return LocalDataset(filtered_data)

    formatted_data = [format_medmcqa(item) for item in data]
    return LocalDataset(formatted_data)

def load_mmlu(data_dir):

    mmlu_files = [
        "anatomy_test.csv",
        "clinical_knowledge_test.csv",
        "college_biology_test.csv",
        "college_medicine_test.csv",
        "medical_genetics_test.csv",
        "professional_medicine_test.csv"
    ]

    all_data = []

    for file_name in mmlu_files:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            qa_dict = {
                "question": row.iloc[0],  # 读取问题
                "options": {
                    "A": row.iloc[1],
                    "B": row.iloc[2],
                    "C": row.iloc[3],
                    "D": row.iloc[4],
                },
                "answer": row.iloc[5],  # 正确答案
                "category": file_name.replace("_test.csv", "")  # 添加类别信息
            }
            all_data.append(qa_dict)

    print(f"📊 MMLU 数据集加载完成，总样本数: {len(all_data)}")
    return LocalDataset(all_data)

# .select(range(100))
def load_all_datasets():
    return {
        # "medqa": load_medqa("./dataset/MedQA/US/test.jsonl").shuffle(seed=42),
        # "medmcqa": load_medmcqa("./dataset/MedMCQA/dev.json").shuffle(seed=42),
        "mmlu": load_mmlu("./dataset/MMLU6/").shuffle(seed=42) # 读取 MMLU
    }



