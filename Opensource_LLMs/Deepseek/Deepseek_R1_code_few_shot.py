import os
import pandas as pd

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

import warnings
warnings.filterwarnings("ignore")

def map_doc(row):
    return row['sent_id'].split('_')[0]

def read_split(file_):
    # Read and filter split files
    return [el for el in open(file_).read().split('\n') if el != '']

def stratify(df):
    # Stratify data based on the label
    g = df.groupby('label')
    return df, g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

def train_test_data(df, path, topic):
    # Create train and test datasets dynamically based on topic
    folder = int(topic)
    test_fold_mapping = {}
    df_train, df_test = None, None

    if str(folder) in os.listdir(path):
        for fold in range(1):
            print("For fold:", fold)
            test_split = fold
            train_split = [i for i in range(5) if i != test_split]

            test = read_split(f'{path}/{folder}/{folder}-{test_split}.cache')
            train = sum([read_split(f'{path}/{folder}/{folder}-{el}.cache') for el in train_split], [])


            df_ = df[df['topic_id'] == int(folder)]
            df_['doc_id'] = df_.apply(map_doc, axis=1)

            # Track the fold number for each doc_id in the test set
            for doc_id in df_[df_['doc_id'].isin(test)]['doc_id']:
                test_fold_mapping[doc_id] = fold

            # Filter and reorder df_train and df_test based on cache file order
            df_train = df_[df_['doc_id'].isin(train)].copy()
            df_test = df_[df_['doc_id'].isin(test)].copy()

            # Reorder df_train and df_test to match the cache file order
            df_train['doc_id'] = pd.Categorical(df_train['doc_id'], categories=train, ordered=True)
            df_test['doc_id'] = pd.Categorical(df_test['doc_id'], categories=test, ordered=True)

            df_train = df_train.sort_values('doc_id').reset_index(drop=True)
            df_test = df_test.sort_values('doc_id').reset_index(drop=True)
            
            # Sort the DataFrame by doc_id and numerical part of sent_id
            df_test["sent_num"] = df_test["sent_id"].str.split("_").str[1].astype(int)
            df_test_new = df_test.sort_values(by=["doc_id", "sent_num"]).drop(columns=["sent_num"]).reset_index(drop=True)

            # Sort the DataFrame by doc_id and numerical part of sent_id
            df_train["sent_num"] = df_train["sent_id"].str.split("_").str[1].astype(int)
            df_train_new = df_train.sort_values(by=["doc_id", "sent_num"]).drop(columns=["sent_num"]).reset_index(drop=True)
            
            df_train_new['counter'] = df_train_new.index
            df_test_new['counter'] = df_test_new.index
            

            print("Shape of Train data:", df_train_new.shape)
            print("Shape of Test data:", df_test_new.shape)
            print("Value counts for each label in train data:", df_train_new.label.value_counts())
            print("Value counts for each label in test data:", df_test_new.label.value_counts())
    else:
        raise FileNotFoundError(f"Folder {folder} not found in path: {path}")

    if df_train is None or df_test is None:
        raise ValueError("Train/test data not created. Check your splits or input data.")

    return df_train_new, df_test_new

def prompt_examples(temp, minimum_char):
    # Generate examples for the prompt
    temp['char_len'] = temp['sentence'].str.len()
    zeros_filtered = temp[(temp['label'] == 0) & (temp['char_len'] >= minimum_char)]
    ones_filtered = temp[(temp['label'] == 1) & (temp['char_len'] >= minimum_char)]

    zero_sampled = zeros_filtered.sample(100, random_state=30)[['sentence', 'label']].reset_index(drop=True)
    ones_sampled = ones_filtered[['sentence', 'label']].reset_index(drop=True)

    examples_ones = ones_sampled.sample(3, random_state=30).reset_index(drop=True)
    examples_zeros = zero_sampled.sample(3, random_state=30).reset_index(drop=True)

    return examples_ones, examples_zeros

def create_prompt_few_shot(examples_ones, examples_zeros, title, description):
    # Create few-shot prompt with examples
    examples_text = ""
    for i, row in examples_ones.iterrows():
        examples_text += f"Example {i+1}: Relevant: '{row['sentence']}'\n"
    for i, row in examples_zeros.iterrows():
        examples_text += f"Example {i+len(examples_ones)+1}: Not Relevant: '{row['sentence']}'\n"

    task_description = f"Review the provided text to determine if it contains relevant information concerning '{title}'."
    topic_definition = f"Definition:\n'{description}'"

    prompt_template = f"""
    Objective:
    {task_description}
    
    {topic_definition}
    
    Examples:
    {examples_text}
    
    Instructions for Response Format:
    Analyze the text and determine its relevance. Provide your analysis strictly in the format:
    Answer: [Relevant/Not Relevant]
    
    Text for Analysis:
    '{{document}}'
    """
    return prompt_template

def prompt_generation(data_desc, df_train, topic):
    # Generate the few-shot prompt
    minimum_char = 240
    df_train['label'] = df_train['label'].replace({'B': 0, '1': 1})
    examples_ones, examples_zeros = prompt_examples(df_train, minimum_char)

    title = data_desc[data_desc['topid'] == str(topic)]['title'].item().split(' Definition')[0]
    description = data_desc[data_desc['topid'] == str(topic)]['description'].item()

    few_shot_prompt = create_prompt_few_shot(examples_ones, examples_zeros, title, description)
    print(few_shot_prompt)
    return few_shot_prompt


def llm_code(data, models, few_shot_prompt, topic):

    template = few_shot_prompt
    print(f'Processing topic: {topic}, Length of data: {data.shape[0]}')

    all_results = []
    output_dir = f'./Deepseek_R1_few_shot_results'
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    output_filename = os.path.join(output_dir, f'FSL_output_results_all_models_{topic}_1000.csv')
    
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
                print("Error processing:", e)
            
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
df = pd.read_csv('../../due_dilligence_data.csv')
data_desc = pd.read_pickle('../../embedding_model_comparision/LLM_exp/topics_data.pkl')
data_desc = pd.DataFrame(data_desc)

path = './data/core/qrels/' ## make sure you have orginal data at this location
models = ['deepseek-r1:8b']

topics = ['1086','1244','1247']

for topic in topics:
    print(f"Running for Topic: {topic}")
    
    print("Creating train and test folds") # for few-shot examples
    df_train, df_test = train_test_data(df, path, topic)
    
    print("Generating prompt")
    few_shot_prompt = prompt_generation(data_desc, df_train, topic)
    
    print("Running LLM models")
    final_data = pd.read_csv(f'./LLM_data/test_data_for_{topic}_1000.csv')
    llm_code(final_data, models, few_shot_prompt, topic)