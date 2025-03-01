import os
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import warnings

warnings.filterwarnings("ignore")

# Function to create zero-shot prompt
def create_prompt_zero_shot(title):
    return f"""Objective:
    "Your task is to determine if the provided text contains 'relevant' information concerning {title}. This involves identifying information directly related to the specified topic, which in a legal or financial document might pertain to specific clauses, terms, or conditions."
    
    Instructions for Response Format:
    Analyze the text provided and determine its relevance based on the specifics of {title} and its implications. Provide your analysis in the following format:
    Answer: [Relevant/Not Relevant]
    
    Text for Analysis:
    '{{document}}'
    """

# Function to generate prompt based on topic
def prompt_generation(data_desc, topic):
    title = data_desc[data_desc['topid'] == str(topic)]['title'].item().split(' Definition')[0]
    return create_prompt_zero_shot(title)

# Function to process text using LLM
def llm_code(data, models, prompt, topic):
    print(f'Processing topic: {topic}, Data size: {data.shape[0]}')
    output_dir = './zero_shot_title_only_results'
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    for model_name in models:
        print(f"Processing model: {model_name}")
        model = OllamaLLM(model=model_name)
        chain = ChatPromptTemplate.from_template(prompt) | model
        results = []
        
        for i, row in data.iterrows():
            print(f"Running: {i+1}/{len(data)} for {topic}, {model_name}")
            try:
                result = chain.invoke(row['sentence'])
                answer = result.split('\n')[0].split(':')[1].strip() if ':' in result else "Format Error"
            except Exception as e:
                answer = "Error"
                print(f"Error processing: {e}")
            
            results.append({
                "Model": model_name,
                "Input Sentence": row['sentence'],
                "Original Label": row['label'],
                "Answer": answer,
                "Topic": topic
            })
        all_results.extend(results)
    
    final_data = pd.DataFrame(all_results)
    output_filename = os.path.join(output_dir, f'ZSL_output_results_all_models_{topic}_1000.csv')
    final_data.to_csv(output_filename, index=False)
    print(f'Results saved to {output_filename}')

# Main execution loop
data_desc = pd.read_pickle('../../topics_data.pkl')
data_desc = pd.DataFrame(data_desc)
models = ['dolphin-llama3', 'llama3.1', 'gemma2']

topics = [
    '1272', '1474', '1238', '1275', '1239', '1520', '1509', '1240',
    '1308', '1319', '1439', '1267', '1242', '1462', '1265', '1444',
    '1312', '1244', '1243', '1468', '1309', '1524', '1247', '1440',
    '1251', '1249', '1248', '1262', '1250', '1252', '1245', '1512',
    '1498', '1601', '1443', '1086', '1551', '1253', '1320', '1304',
    '1469', '1611', '1300', '1489', '1500', '1261', '1318', '1460',
    '1475', '1321'
]

for topic in topics:
    print(f"Running for Topic: {topic}")
    print("Generating prompt")
    zero_shot_prompt = prompt_generation(data_desc, topic)
    
    print("Running LLM models")
    data_file = f'../../LLM_data/test_data_for_{topic}_1000.csv'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Skipping topic {topic}.")
        continue
    final_data = pd.read_csv(data_file)
    llm_code(final_data, models, zero_shot_prompt, topic)
