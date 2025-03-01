import os
import pickle
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

def create_topic_doc_dict(path):
    """
    Creates a dictionary mapping each topic to its relevant documents based on .qrels files within each topic's folder.
    
    Args:
    - path (str): Path to the directory containing topic folders.

    Returns:
    - dict: A dictionary with topic IDs as keys and lists of document IDs as values.
    """
    topic_doc_dict = {}
    for topic in os.listdir(path):
        topic_path = os.path.join(path, topic)
        if os.path.isdir(topic_path):
            doc_ids = [file.split('.')[0] for file in os.listdir(topic_path) if file.endswith('.qrels')]
            topic_doc_dict[topic] = doc_ids
    return topic_doc_dict

def create_sentence_dict(docs_dir, doc_ids):
    """
    Extracts sentences from documents and stores them in a dictionary with unique identifiers.

    Args:
    - docs_dir (str): Directory where documents are stored.
    - doc_ids (list): List of document IDs to process.

    Returns:
    - dict: A dictionary with unique document-sentence identifiers as keys and sentences as values.
    """
    sentences_dict = {}
    for doc_id in tqdm(doc_ids, desc="Processing documents"):
        sentences_file = os.path.join(docs_dir, f"{doc_id}.sentences")
        if os.path.isfile(sentences_file):
            with open(sentences_file, 'r', encoding='utf-8') as file:
                sentences_dict.update({
                    f"{doc_id}_{index}": sentence.strip().replace('\t', ' ')[8:]
                    for index, sentence in enumerate(file)
                })
    return sentences_dict

def read_qrels(path):
    """
    Reads annotations for each document from .qrels files and structures them into a dictionary.

    Args:
    - path (str): Path to the directory containing .qrels files organized by topics.

    Returns:
    - defaultdict(list): Dictionary with topic as key and list of tuples (document_sentence_id, annotation) as values.
    """
    topic_doc_dict = defaultdict(list)
    for topic in tqdm(os.listdir(path), desc="Reading qrels"):
        topic_path = os.path.join(path, topic)
        if os.path.isdir(topic_path):
            for file in os.listdir(topic_path):
                if file.endswith('.qrels'):
                    qrel_content = open(os.path.join(topic_path, file)).read().split('\n')
                    doc_id = file.split('.')[0]
                    topic_doc_dict[topic].extend([(f'{doc_id}_{i}', annotation) 
                                                  for i, annotation in enumerate(qrel_content) if annotation])
    return topic_doc_dict

def main():
    # Paths setup
    qrels_path = './core/qrels'
    docs_directory = './core/docs'
    dataset_path = 'due_dilligence_kira.hf'
    qrels_dict_file = 'qrels_dict.pkl'
    
    # Generate topic-document dictionary
    topic_document_dictionary = create_topic_doc_dict(qrels_path)
    all_document_ids = [doc for docs in topic_document_dictionary.values() for doc in docs]
    
    # Create sentences dictionary
    doc_sentences_dict = create_sentence_dict(docs_directory, all_document_ids)

    # Read qrels and prepare dataset
    qrel = read_qrels(qrels_path)

    # Saving qrels dictionary
    with open(qrels_dict_file, 'wb') as file:
        pickle.dump(qrel, file)

    # Constructing dataset for HuggingFace Dataset library
    dataset = {'sentence': [], 'label': [], 'index': [], 'sent_id': [], 'topic_id': []}
    index = 0
    for topic, annotations in qrel.items():
        for sent_id, label in annotations:
            if sent_id in doc_sentences_dict:
                dataset['sentence'].append(doc_sentences_dict[sent_id])
                dataset['label'].append(label)
                dataset['index'].append(index)
                dataset['sent_id'].append(sent_id)
                dataset['topic_id'].append(topic)
                index += 1

    # Check for length inconsistencies in the dataset dictionary
    dataset_lengths = {key: len(value) for key, value in dataset.items()}
    print("Dataset lengths:", dataset_lengths)
    if len(set(dataset_lengths.values())) != 1:
        print("Error: Not all fields in the dataset have the same length.")
    else:
        hf_dataset = Dataset.from_dict(dataset)
        hf_dataset.save_to_disk(dataset_path)
        print("Data Saved")

        # Converting and saving in pandas DataFrame
        df = pd.DataFrame(dataset)
        df.to_csv('dataset.csv', index=False)

if __name__ == '__main__':
    main()