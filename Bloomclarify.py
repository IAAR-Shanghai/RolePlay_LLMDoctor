import requests
import json
import base64
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from Myutils import *



#MedQA

def classify_Med_QA(question, answer,engine):
    # print(question,answer)
    engine = engine
    temperature = 0
    max_tokens = 10
    frequency_penalty = 0
    presence_penalty = 0
    n=1,  # 只返回一个答案
    stop=["\n"]  # 停止符，保证输出是简洁的分类
    # 构建详细的背景说明
    prompt = (

        "You are an expert in Bloom's Taxonomy and its application in the medical field.\n\n"
    "Bloom's Taxonomy classifies cognitive learning into six levels:\n"
    "1. Remembering - Recall facts or basic information.\n"
    "2. Understanding - Explain concepts or principles.\n"
    "3. Applying - Use knowledge in real-world scenarios.\n"
    "4. Analyzing - Identify patterns, relationships, or causality.\n"
    "5. Evaluating - Critically appraise the validity, reliability, or impact of medical information, studies, or treatment options. "
    "This requires weighing evidence, identifying biases, and justifying decisions with logic. "
    "Choosing the best answer in a multiple-choice question **does not** necessarily fall into this category unless it requires substantial justification or critique.\n"
    "6. Creating - Develop new ideas, solutions, or strategies.\n\n"
    "Now classify the following question-answer pair into one of these categories.\n\n"
    "Follow this classification:\n"
    "- If it is recalling facts, classify as 'Remembering'.\n"
    "- If it explains concepts, classify as 'Understanding'.\n"
    "- If it applies knowledge to a scenario, classify as 'Applying'.\n"
    "- If it examines relationships or patterns, classify as 'Analyzing'.\n"
    "- If it makes a judgment requiring substantial critique or weighing of evidence, classify as 'Evaluating'.\n"
    "- If it generates new ideas, classify as 'Creating'.\n\n"
    "Respond with only one word from: [Remembering, Understanding, Applying, Analyzing, Evaluating, Creating]."
        
        f"Question: {question}\nAnswer: {answer}\n\nClassification:"
    )

    def construct_request_data(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, prompt):
        if engine=="gpt-4o":
            request_data = {
            "model": engine,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "messages": [
            {"role": "system", "content": "You are an expert in Bloom's Taxonomy and its application in the medical field"},
            {"role": "user", "content": prompt}
            ]
             }
            return request_data
        
        if engine =="o1-preview":
            request_data = {
                "model": engine,
                # "max_completion_tokens": max_tokens,  # 替换为支持的参数
                "messages": [     
                     {"role": "user", "content": prompt}
                ]
                }
            return request_data
    
    @retry(stop=stop_after_attempt(10), wait=wait_fixed(3), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def get_response(request_data):
        headers = {
            "Authorization": f"Bearer {one_api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(one_api_url, headers=headers, data=json.dumps(request_data))
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e},休息两秒钟")
            time.sleep(2) 
            raise
        
    def process_response(response):
        if response.status_code == 200:
            response_data = response.json()
            # print(response_data)           
            return response_data       
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return "ERROR."
        

    request_data = construct_request_data(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, prompt)
    response = get_response(request_data)
    info = process_response(response)

    
    # print(info)
    # print("=====================")
    content = info["choices"][0]["message"]["content"]

    return content
# MedQA
def classify_MedQA_dataset(data):
    progress_file = 'progress_medqa_{}.json'.format(engine)
    progress = load_progress(progress_file)
    if progress is None:
        progress = {"current_index": 0, "classified_data": []}

    #处理单个QA
    def process_item(item):
        question = item['question']
        answer = item['answer']
        options = item['options']
        answer_index = item['answer_idx']
        try:
            classification = classify_Med_QA(question, answer, engine)
            return {
                'question': question,
                'answer': answer,
                'classification': classification,
                "options": options,
                "answer_index": answer_index
            }
        except Exception as e:
            print(f"Error processing item: {e}")
            return None

    begin=0
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = []
        for i in range(progress["current_index"], len(data)):
            if i > progress["current_index"]:
                time.sleep(delay)
            futures.append(executor.submit(process_item, data[i]))
            begin+=1
            print("========{}/{}=============".format(begin,len(data)))

        for future in as_completed(futures):
            result = future.result()
            if result:
                progress["classified_data"].append(result)
                progress["current_index"] += 1
                save_progress(progress, progress_file)

    return progress["classified_data"]


#MMLU,MedMCQA
def classify_MMLU_QA(question, answer,engine):
    # print(question,answer)
    engine = engine
    temperature = 0
    max_tokens = 10
    frequency_penalty = 0
    presence_penalty = 0
    n=1,  # 只返回一个答案
    stop=["\n"]  # 停止符，保证输出是简洁的分类
    # 构建详细的背景说明
    prompt = (

         "You are an expert in Bloom's Taxonomy and its application in the medical field.\n\n"
    "Bloom's Taxonomy classifies cognitive learning into six levels:\n"
    "1. Remembering - Recall facts or basic information.\n"
    "2. Understanding - Explain concepts or principles.\n"
    "3. Applying - Use knowledge in real-world scenarios.\n"
    "4. Analyzing - Identify patterns, relationships, or causality.\n"
    "5. Evaluating - Critically appraise the validity, reliability, or impact of medical information, studies, or treatment options. "
    "This requires weighing evidence, identifying biases, and justifying decisions with logic. "
    "Choosing the best answer in a multiple-choice question **does not** necessarily fall into this category unless it requires substantial justification or critique.\n"
    "6. Creating - Develop new ideas, solutions, or strategies.\n\n"
    "Now classify the following question-answer pair into one of these categories.\n\n"
    "Follow this classification:\n"
    "- If it is recalling facts, classify as 'Remembering'.\n"
    "- If it explains concepts, classify as 'Understanding'.\n"
    "- If it applies knowledge to a scenario, classify as 'Applying'.\n"
    "- If it examines relationships or patterns, classify as 'Analyzing'.\n"
    "- If it makes a judgment requiring substantial critique or weighing of evidence, classify as 'Evaluating'.\n"
    "- If it generates new ideas, classify as 'Creating'.\n\n"
    "Respond with only one word from: [Remembering, Understanding, Applying, Analyzing, Evaluating, Creating]."
        
        f"Question: {question}\nAnswer: {answer}\n\nClassification:"
    )

    def construct_request_data(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, prompt):
        if engine=="gpt-4o":
            print(question,answer)
            request_data = {
            "model": engine,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "messages": [
            {"role": "system", "content": "You are an expert in Bloom's Taxonomy and its application in the medical field"},
            {"role": "user", "content": prompt}
            ]
             }
            return request_data
        
        if engine =="o1-preview":
            request_data = {
                "model": engine,
                # "max_completion_tokens": max_tokens,  # 替换为支持的参数
                "messages": [     
                     {"role": "user", "content": prompt}
                ]
                }
            return request_data
    
    @retry(stop=stop_after_attempt(10), wait=wait_fixed(3), retry=retry_if_exception_type(requests.exceptions.RequestException))
    def get_response(request_data):
        headers = {
            "Authorization": f"Bearer {one_api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(one_api_url, headers=headers, data=json.dumps(request_data))
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e},休息两秒钟")
            time.sleep(2) 
            raise
        
    def process_response(response):
        if response.status_code == 200:
            response_data = response.json()
            # print(response_data)           
            return response_data       
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return "ERROR."
        

    request_data = construct_request_data(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, prompt)
    response = get_response(request_data)
    info = process_response(response)
    print(info)
    # print("=====================")
    content = info["choices"][0]["message"]["content"]

    return content

def classify_MMLU_dataset(data,data2):
    progress_file = 'progress_{}_{}.json'.format(data2,engine)
    progress = load_progress(progress_file)
    if progress is None:
        progress = {"current_index": 0, "classified_data": []}

    #处理单个QA
    def process_item(item):
        question = item['question']
        answer = item['answer']
        options = item['options']
        # print(question,answer,options)
        try:
            classification = classify_MMLU_QA(question, answer, engine)
            return {
                'question': question,
                'answer': answer,
                'classification': classification,
                "options": options,
            }
        except Exception as e:
            print(f"Error processing item: {e}")
            return None

    begin=0
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = []
        for i in range(progress["current_index"], len(data)):
            if i > progress["current_index"]:
                time.sleep(delay)
            futures.append(executor.submit(process_item, data[i]))
            begin+=1
            print("========{}/{}=============".format(begin,len(data)))

        for future in as_completed(futures):
            result = future.result()
            if result:
                progress["classified_data"].append(result)
                progress["current_index"] += 1
                save_progress(progress, progress_file)

    return progress["classified_data"]


# 主函数
def main():
    # 加载MedQA数据集
    if(dataset_name=="Med_QA"):
        data_file_path = 'XXXXX'  # 数据集保存路径
        medqa_data = load_data(data_file_path,dataset_name)
        classified_data = classify_MedQA_dataset(medqa_data)
        result = count_classification_terms(classified_data, bloom_categories )
        print(result)
        save_classified_data(classified_data, output_file='./outputs0305/classified_{}_{}.json'.format(dataset_name,engine))

    if(dataset_name=="MMLU"):
        files=["anatomy_test.csv","clinical_knowledge_test.csv","college_biology_test.csv","college_medicine_test.csv","medical_genetics_test.csv","professional_medicine_test.csv"]
        abbrevitation=["AT","CK","CB","CM","MG","PM"]
        for i in range(0,6):
            data_file_path = 'XXXX'.format(files[i])
            data2=dataset_name+abbrevitation[i]#数据集名字
            data = load_data(data_file_path,dataset_name)
            classified_data = classify_MMLU_dataset(data,data2)
            result = count_classification_terms(classified_data, bloom_categories )
            print(result)
            
            save_classified_data(classified_data, output_file='./outputs0305/classified_{}_{}.json'.format(data2,engine))

    if(dataset_name=="MedMCQA"):
        data_file_path = 'XXX'
        data = load_data(data_file_path,dataset_name)
        data2=dataset_name
        classified_data = classify_MMLU_dataset(data,data2)
        result = count_classification_terms(classified_data, bloom_categories )
        print(result)
        save_classified_data(classified_data, output_file='./outputs0305/classified_{}_{}.json'.format(dataset_name,engine))

   

if __name__ == "__main__":
    # 设置你的 OpenAI API 密钥,url
    one_api_key = "XXXXXX"
    one_api_url = "XXXXXX"
    bloom_categories = ["Remembering","Understanding","Applying","Analyzing","Evaluating","Creating"]

    dataset_name="MedQA"
    engine = "gpt-4o"
    max_concurrent=1
    delay=0.5

    main()
    
