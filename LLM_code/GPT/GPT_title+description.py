import openai
import pandas as pd
from openai.error import RateLimitError
import time
import os

# Set your OpenAI API key here
openai.api_key = "your-api-key-here"

# Function to analyze a sentence using OpenAI API
def analyze_text(sentence, model_name, template):
    formatted_template = template.replace("{document}", sentence)
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": formatted_template}],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']

# Function to create a zero-shot prompt with description
def create_prompt_zero_shot_with_desc(title, description):
    task_description = f"""Your task is to determine if the provided text contains 'relevant' information concerning {title}. This involves identifying information directly related to the specified topic, which in a legal or financial document might pertain to specific clauses, terms, or conditions."""
    topic_definition = f"Topic Definition:\n'{description}'"

    prompt_template = f"""
    Objective:
    {task_description}
    
    {topic_definition}
    
    Instructions for Response Format:
    Analyze the text provided and determine its relevance based on the specifics of {title} and provided {topic_definition}. Provide your analysis in the following format:
    Answer: [Relevant/Not Relevant]
    
    Text for Analysis:
    '{{document}}'
    """
    return prompt_template

# Function to generate a prompt based on topic
def prompt_generation(data_desc, topic):
    title = data_desc[data_desc['topid'] == str(topic)]['title'].item().split(' Definition')[0]
    description = data_desc[data_desc['topid'] == str(topic)]['description'].item()
    return create_prompt_zero_shot_with_desc(title, description)

# Load topic metadata
data_desc = pd.read_pickle('../../embedding_model_comparision/LLM_exp/topics_data.pkl')
data_desc = pd.DataFrame(data_desc)

# List of topics
topics = [
    '1272', '1474', '1238', '1275', '1239', '1520', '1509', '1240',
    '1308', '1319', '1439', '1267', '1242', '1462', '1265', '1444',
    '1312', '1244', '1243', '1468', '1309', '1524', '1247', '1440',
    '1251', '1249', '1248', '1262', '1250', '1252', '1245', '1512',
    '1498', '1601', '1443', '1086', '1551', '1253', '1320', '1304',
    '1469', '1611', '1300', '1489', '1500', '1261', '1318', '1460',
    '1475', '1321'
]

# Set model name
model_name = 'gpt-4o-mini'
# model_name = 'gpt-4o'


# Ensure output directory exists
output_dir = './GPT_zero_shot_with_description_output'
os.makedirs(output_dir, exist_ok=True)

# Process each topic
for topic in topics:
    data_path = f'./LLM_data/test_data_for_{topic}_1000.csv'
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Skipping topic {topic}.")
        continue
    
    data = pd.read_csv(data_path)
    print(f"Processing topic {topic}, Data shape: {data.shape}")
    print(data.label.value_counts())
    
    print("Generating prompt")
    zero_shot_prompt = prompt_generation(data_desc, topic)
    
    predictions = []
    
    for i, row in data.iterrows():
        text = row['sentence']
        original_label = row['label']
        
        print(f"Running for topic {topic}: {i}")
        print("Original label:", original_label)
        
        try:
            output = analyze_text(text, model_name, zero_shot_prompt)
            predictions.append(output)
            print("Predicted label:", output)
        except RateLimitError:
            print("Rate limit exceeded. Waiting for a few seconds...")
            time.sleep(5)
            predictions.append(original_label)
        except Exception as e:
            print(f"An error occurred: {e}")
            predictions.append(original_label)
    
    print(f"Total sentences: {len(data)}, Predictions: {len(predictions)}")
    
    data['relevance'] = predictions
    data['relevance'] = data['relevance'].replace({'Not Relevant': 'Answer: Not Relevant'})
    
    results = data.copy()
    results['model'] = model_name
    results['relevance'] = results['relevance'].replace({'Answer: Not Relevant': 'B', 'Answer: Relevant': '1'})
    
    output_file = os.path.join(output_dir, f'output_{model_name}_{topic}.csv')
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
