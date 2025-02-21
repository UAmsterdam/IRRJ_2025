import os
import pandas as pd
from sklearn.metrics import classification_report
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

def process_results(prompt_type):
    topics = [1086, 1244, 1247]
    
    results_dir_map = {
        "zero_shot": "./Deepseek_R1_zero_shot_results",
        "zero_shot_w_desc": "./Deepseek_R1_zero_shot_w_desc_results",
        "few_shot": "./Deepseek_R1_few_shot_results"
    }
    
    metrics_dir_map = {
        "zero_shot": "./Deepseek_R1_zero_shot_metrics_output",
        "zero_shot_w_desc": "./Deepseek_R1_zero_shot_w_desc_metrics_output",
        "few_shot": "./Deepseek_R1_few_shot_metrics_output"
    }
    
    results_dir = results_dir_map[prompt_type]
    metrics_dir = metrics_dir_map[prompt_type]
    ensure_directory_exists(metrics_dir)
    
    for topic in topics:
        print("Processing the topic:", topic)
        results_file = f'{results_dir}/ZSL_output_results_all_models_{topic}_1000.csv'
        
        if not os.path.exists(results_file):
            print(f"Results file {results_file} not found. Skipping topic {topic}.")
            continue
        
        results = pd.read_csv(results_file)
        
        relevant_keywords = ['Relevant', '**Relevant**', '[Relevant]', 'Potentially Relevant', '**Potentially Relevant**', 'partially relevant', 
                             '** Relevant', 'Partially Relevant', '**Partially Relevant**', '**Possibly Relevant**', 'Relevant.',
                            'Possibly Relevant', '[Potentially Relevant]']
        not_relevant_keywords = ['Not Relevant', '**Not Relevant**', '[Not Relevant]', '** Not Relevant', 'Not Relevant.']

        relevant_dict = {keyword: '1' for keyword in relevant_keywords}
        not_relevant_dict = {keyword: 'B' for keyword in not_relevant_keywords}
        
        condition1 = results['Answer'] == 'Format Error'
        condition2 = results['Answer'] == 'Answer'

        results['Answer'] = results['Answer'].replace(relevant_dict)
        results['Answer'] = results['Answer'].replace(not_relevant_dict)
        results.loc[condition1, 'Answer'] = results.loc[condition1, 'Original Label']
        results.loc[condition2, 'Answer'] = results.loc[condition2, 'Original Label']
        
        print("Resulted value counts", results.groupby('Model')['Answer'].value_counts())
        print("Original value counts:", results.groupby('Model')['Original Label'].value_counts())

        models = results['Model'].unique()
        report_data = []
        
        for model in models:
            print(f"Classification Report for {model}:")
            model_data = results[results['Model'] == model]
            report = classification_report(model_data['Original Label'], model_data['Answer'], output_dict=True)
            
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    data = {
                        'Model': model,
                        'Label': label,
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1-score'],
                        'Support': metrics['support'],
                        'Topic': topic
                    }
                    report_data.append(data)
        
        classification_df = pd.DataFrame(report_data)
        
        output_file = f'{metrics_dir}/results_deepseek_r1_{topic}.csv'
        classification_df.to_csv(output_file, index=False)
        print(f"Result saved for topic {topic} in {output_file}")

if __name__ == "__main__":
    print("Select the type of analysis:")
    print("1. Zero-Shot Analysis")
    print("2. Zero-Shot with Description Analysis")
    print("3. Few-Shot Analysis")
    
    choice = input("Enter the number corresponding to your choice: ")
    prompt_map = {"1": "zero_shot", "2": "zero_shot_w_desc", "3": "few_shot"}
    prompt_type = prompt_map.get(choice)
    
    if prompt_type:
        process_results(prompt_type)
    else:
        print("Invalid choice. Please select 1, 2, or 3.")

main()