import openai
import pandas as pd
from openai.error import RateLimitError
import time
import os

# Set your OpenAI API key here
openai.api_key = "your-api-key-here"

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


# Function to process a single sentence using OpenAI API
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


# Function to create a few-shot prompt
def create_prompt_few_shot(examples_ones, examples_zeros, title, description):
    examples_text = "".join([f"Example {i+1}: Relevant: '{row['sentence']}'\n" for i, row in examples_ones.iterrows()])
    examples_text += "".join([f"Example {i+4}: Not Relevant: '{row['sentence']}'\n" for i, row in examples_zeros.iterrows()])
    return f"""Objective:
    Review the provided text to determine if it contains relevant information concerning {title}.  Relevant information directly discusses risks or specifics related to the {title}, pledged in financial transactions.'.
    
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

# Function to generate prompt based on topic
def prompt_generation(data_desc, df_train, topic):
    # Generate the few-shot prompt
    minimum_char = 240
    df_train['label'] = df_train['label'].replace({'B': 0, '1': 1})
    examples_ones, examples_zeros = prompt_examples(df_train, minimum_char)

    title = data_desc[data_desc['topid'] == str(topic)]['title'].item().split(' Definition')[0]
    description = data_desc[data_desc['topid'] == str(topic)]['description'].item()

    few_shot_prompt = create_prompt_few_shot(examples_ones, examples_zeros, title, description)
#     print(few_shot_prompt)
    return few_shot_prompt

# Ensure output directory exists
output_dir = './GPT_few_shot_outputs'
os.makedirs(output_dir, exist_ok=True)

# Load topic metadata
data_desc = pd.read_pickle('../../embedding_model_comparision/LLM_exp/topics_data.pkl')
data_desc = pd.DataFrame(data_desc)

df = pd.read_csv('../../due_dilligence_data.csv')

topics = [
    '1272', '1474', '1238', '1275', '1239', '1520', '1509', '1240',
    '1308', '1319', '1439', '1267', '1242', '1462', '1265', '1444',
    '1312', '1244', '1243', '1468', '1309', '1524', '1247', '1440',
    '1251', '1249', '1248', '1262', '1250', '1252', '1245', '1512',
    '1498', '1601', '1443', '1086', '1551', '1253', '1320', '1304',
    '1469', '1611', '1300', '1489', '1500', '1261', '1318', '1460',
    '1475', '1321'
]
path = '../../core/qrels/'

# Set model name
model_name = 'gpt-4o-mini'
# model_name = 'gpt-4o'


# Process each topic
for topic in topics:
    data_path = f'./LLM_data/test_data_for_{topic}_1000.csv'
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Skipping topic {topic}.")
        continue
    
    data = pd.read_csv(data_path)
    print(f"Processing topic {topic}, Data shape: {data.shape}")
    print(data.label.value_counts())

    print("Creating train and test folds")
    df_train, df_test = train_test_data(df, path, topic)
    
    print("Generating prompt")
    few_shot_prompt = prompt_generation(data_desc, data, topic)
    
    predictions = []
    
    for i, row in data.iterrows():
        text = row['sentence']
        original_label = row['label']
        
        print(f"Running for topic {topic}: {i}")
        print("Original label:", original_label)
        
        try:
            output = analyze_text(text, model_name, few_shot_prompt)
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
