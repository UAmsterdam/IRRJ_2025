import os
import pandas as pd

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def create_prompt_zero_shot(title):
    prompt_template = f"""Objective:
    "Your task is to determine if the provided text contains 'relevant' information concerning '{title}'. This involves identifying information    directly related to the specified topic, which, in a legal or financial document, might pertain to specific clauses, terms, or conditions."
    
    Instructions for Response Format:
    Analyze the text provided and determine its relevance based on the specifics of '{title}' and its implications. Provide your analysis strictly in the following format:
    Answer: [Relevant/Not Relevant]
    
    Text for Analysis:
    '{{document}}' 
    """
    return prompt_template


def prompt_generation(data_desc, topic):

    title = data_desc[data_desc['topid'] == str(topic)]['title'].item().split(' Definition')[0]

    zero_shot_prompt = create_prompt_zero_shot(title)
    return zero_shot_prompt

def llm_code(data, models, zero_shot_prompt, topic):
    template = zero_shot_prompt
    print(f'Processing topic: {topic}, Length of data: {data.shape[0]}')

    all_results = []
    output_dir = f'./Deepseek_R1_zero_shot_results'
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_filename = os.path.join(output_dir, f'ZSL_output_results_all_models_{topic}_1000.csv')
    
    for model_name in models:
        print(f"Processing model: {model_name}")
        
        model = OllamaLLM(model=model_name)
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        
        results = []
        i = 0
        for index, row in data.iterrows():
            print(f"Running: {i} / {len(data)} for {topic}, {model_name}")
            input_sentence = row['sentence']
            original_label = row['label']
            
            try:
                result = chain.invoke(input_sentence)
                lines = result.split('\n')
                answer = "Format Error"
                reason = "Format Error"
                for line in lines:
                    if ':' in line:
                        parts = line.split(':')
                        if "answer" in line.lower():
                            answer = parts[1].strip()
                        if "reason" in line.lower():
                            reason = parts[1].strip()
                
                print("Original label:", original_label)
                print("Predicted label:", answer)
            
            except Exception as e:
                answer = "Error"
                reason = str(e)
            
            results.append({
                "Model": model_name,
                "Input Sentence": input_sentence,
                "Original Label": original_label,
                "Answer": answer,
                "Topic": topic
            })
            i += 1
        
        all_results.extend(results)
    
    final_data = pd.DataFrame(all_results)
    final_data.to_csv(output_filename, index=False)
    print(f'Results saved to {output_filename}')
    print(final_data.head())
    print("Done")


# Main execution loop
data_desc = pd.read_pickle('../../embedding_model_comparision/LLM_exp/topics_data.pkl')
data_desc = pd.DataFrame(data_desc)

models = ['deepseek-r1:8b']
topics = ['1244', '1247', '1086']

for topic in topics:
    print(f"Running for Topic: {topic}")
    
    print("Generating prompt")
    zero_shot_prompt = prompt_generation(data_desc, topic)
    
    print(zero_shot_prompt)

    print("Running LLM models")
    final_data = pd.read_csv(f'./LLM_data/test_data_for_{topic}_1000.csv')
    
    llm_code(final_data, models, zero_shot_prompt, topic)