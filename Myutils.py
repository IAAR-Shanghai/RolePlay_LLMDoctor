import os
import json
import pandas as pd
import pickle


def save_progress(data, progress_file='progress.json'):
    with open(progress_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
def load_progress(progress_file='progress.json'):
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    return None
 #保存进度
def save_classified_data(classified_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(classified_data, file,ensure_ascii=False, indent=4)  # 如果是CSV, 使用 pd.DataFrame(classified_data).to_csv(output_file)     
def count_classification_terms(json_data, terms):
    count_dict = {term: 0 for term in terms}
    for entry in json_data:
        # 获取 classification 部分的 content
        content = entry.get('classification', '')
        if content in count_dict:
            count_dict[content] += 1
    
    return count_dict  

def load_data(file_path,dataset_name):
    if(dataset_name=="Med_QA"):
        file_path = open(file_path, 'r', encoding='utf-8')
        temp = []
        for line in file_path.readlines():
            dic = json.loads(line)
            temp.append(dic)
        #     temp2.append(dic)
        temp2=[]
        for x in range(1300,1302):
            temp2.append(temp[x])
        return temp
    
    if(dataset_name=="MMLU"):
    
        df = pd.read_csv(file_path)
        result = []
        for index, row in df.iterrows():
            # 创建一个字典来存储当前行的数据
            qa_dict = {
                "question": row[0],
                "options":{
                    "A": row[1],
                    "B": row[2],
                    "C": row[3],
                    "D": row[4],
                },    
                "answer": row[5]
            }
            # 将字典添加到结果列表中
            result.append(qa_dict)
        return result

    if(dataset_name=="MedMCQA"):
        file_path = open(file_path, 'r', encoding='utf-8')
        temp = []
        for line in file_path.readlines():
            dic = json.loads(line)
            def update_dic_structure(dic):
                # 初始化一个空字典用于存储新的结构
                new_dic = {}
                num_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
                # 检查字典中是否存在键 'cop'
                if 'cop' in dic:
                    cop_value = dic['cop']                    
                    answer = ''.join(num_to_letter[int(digit)] for digit in str(cop_value))
                    new_dic['answer'] = answer
                
                # 初始化 options 字典
                options = {}
                for key in ['opa', 'opb', 'opc', 'opd']:
                    if key in dic:
                        letter = chr(ord(key[-1]) - ord('a') + ord('A'))
                        options[letter] = dic[key]
                
                new_dic['options'] = options
                new_dic['question']=dic['question']
                
                return new_dic
            dic= update_dic_structure(dic)
            temp.append(dic)
        #     temp2.append(dic)
        temp2=[]
        for x in range(1100,1103):
            temp2.append(temp[x])
        return temp 
    
   