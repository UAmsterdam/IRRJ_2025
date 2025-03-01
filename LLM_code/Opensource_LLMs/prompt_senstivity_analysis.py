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
    print("Cross Topic:",folder)
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

def create_evaluation_data_for_llms(df_train, df_test, topic):
    # Create evaluation data for the given topic
    df_test['char_len'] = df_test['sentence'].str.len()
    neg_data = df_test[(df_test['char_len'] > 50) & (df_test['label'] == 'B')].sample(1000, random_state=30)
    pos_data = df_test[(df_test['char_len'] > 50) & (df_test['label'] == '1')]

    final_eval_data = pd.concat([neg_data, pos_data], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Final data shape:", final_eval_data.shape)
    print("Final evaluation label counts: ", final_eval_data.label.value_counts())
    
#     output_path = f'./data/test_data_for_{topic}_1000.csv'
#     final_eval_data.to_csv(output_path, index=False)
#     print(f"Evaluation data saved to: {output_path}")

def prompt_examples(temp, minimum_char, random_value):
    temp['char_len'] = temp['sentence'].str.len()
    zeros_filtered = temp[(temp['label'] == 0) & (temp['char_len'] >= minimum_char)]
    ones_filtered = temp[(temp['label'] == 1) & (temp['char_len'] >= minimum_char)]
    
    zero_sampled = zeros_filtered.sample(100, random_state=random_value)[['sentence', 'label']].reset_index(drop=True)
    ones_sampled = ones_filtered[['sentence', 'label']].reset_index(drop=True)
    
    examples_ones = ones_sampled.sample(3, random_state=random_value).reset_index(drop=True)
    examples_zeros = zero_sampled.sample(3, random_state=random_value).reset_index(drop=True)
    
    return examples_ones, examples_zeros


def create_prompt_few_shot(examples_ones, examples_zeros, title, description):
    examples_text = "".join([f"Example {i+1}: Relevant: '{row['sentence']}'\n" for i, row in examples_ones.iterrows()])
    examples_text += "".join([f"Example {i+4}: Not Relevant: '{row['sentence']}'\n" for i, row in examples_zeros.iterrows()])
    return f"""Objective:
    Review the provided text to determine if it contains relevant information concerning {title}. Relevant information directly discusses risks or specifics related to the {title}, pledged in financial transactions.'.
    
    Topic Definition:
    '{description}'
    
    Examples: Here are examples for each class
    {examples_text}
    
    Instructions for Response Format:
    Analyze the text provided and determine its relevance based on the specifics of {title} and its implications.  Provide your analysis in the following format:
    Answer: [Relevant/Not Relevant]
    
    Text for Analysis:
    '{{document}}'
    """

def prompt_generation(data_desc, df_train, topic, random_value):
    minimum_char = 240
    df_train['label'] = df_train['label'].replace({'B': 0, '1': 1})
    examples_ones, examples_zeros = prompt_examples(df_train, minimum_char, random_value)
    
    title = data_desc[data_desc['topid'] == str(topic)]['title'].item().split(' Definition')[0]
    description = data_desc[data_desc['topid'] == str(topic)]['description'].item()
    
    return create_prompt_few_shot(examples_ones, examples_zeros, title, description)

def llm_code(data, models, few_shot_prompt, topic, random_value):
    print(f'Processing topic: {topic}, Length of data: {data.shape[0]}')
    
    prompt_map = {30: 'prompt1', 40: 'prompt2', 50: 'prompt3', 60: 'prompt4'}
    output_dir = f'./different_prompt_results/{prompt_map.get(random_value, f"prompt{random_value}")}'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    for model_name in models:
        print(f"Processing model: {model_name}")
        model = OllamaLLM(model=model_name)
        chain = ChatPromptTemplate.from_template(few_shot_prompt) | model
        
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
    output_filename = os.path.join(output_dir, f'FSL_output_results_all_models_{topic}_1000.csv')
    final_data.to_csv(output_filename, index=False)
    print(f'Results saved to {output_filename}')

data_desc = pd.read_pickle('../../topics_data.pkl')
data_desc = pd.DataFrame(data_desc)
df = pd.read_csv('../../due_dilligence_data.csv')
models = ['gemma2']
topics = [
    '1272', '1474', '1238', '1275', '1239', '1520', '1509', '1240',
    '1308', '1319', '1439', '1267', '1242', '1462', '1265', '1444',
    '1312', '1244', '1243', '1468', '1309', '1524', '1247', '1440',
    '1251', '1249', '1248', '1262', '1250', '1252', '1245', '1512',
    '1498', '1601', '1443', '1086', '1551', '1253', '1320', '1304',
    '1469', '1611', '1300', '1489', '1500', '1261', '1318', '1460',
    '1475', '1321'
]
random_values = [30, 40, 50, 60]

for topic in topics:
    for random_value in random_values:
        print(f"Running for Topic: {topic} with Random Value: {random_value}")
        df_train, df_test = train_test_data(df, '../../core/qrels/', topic)
        prompt = prompt_generation(data_desc, df_train, topic, random_value)
        data_file = f'../../LLM_data/test_data_for_{topic}_1000.csv'
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping topic {topic}.")
            continue
        final_data = pd.read_csv(data_file)
        llm_code(final_data, models, prompt, topic, random_value)