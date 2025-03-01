import os
import pandas as pd
from sklearn.metrics import classification_report

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
output_dir = './GPT_zero_shot_outputs_only_topic_metrics' # for title only
# output_dir = './GPT_zero_shot_metrics_output_with_description' # for title and description
# output_dir = './GPT_few_shot_metrics_output' # for title, description, and examples

os.makedirs(output_dir, exist_ok=True)

# Process each topic
for topic in topics:
    results_file = f'./GPT_zero_shot_outputs_only_topic/output_{model_name}_{topic}.csv'  # for title only
    # results_file = f'./GPT_zero_shot_with_description_output/output_{model_name}_{topic}.csv'  # for title and description
    # results_file = f'./GPT_few_shot_outputs/output_{model_name}_{topic}.csv'  # for title, description, and examples
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Skipping topic {topic}.")
        continue
    
    # Load results
    results = pd.read_csv(results_file)
    print(f"Processing topic: {topic}")
    print("Label Counts:\n", results.label.value_counts())
    print("Relevance Counts:\n", results.relevance.value_counts())
    
    # Identify invalid relevance labels
    invalid_relevance = results[(results['relevance'] != '1') & (results['relevance'] != 'B')].head(35)
    print("Invalid Relevance Rows:\n", invalid_relevance)
    
    # Compute classification report
    report_dict = classification_report(results['label'], results['relevance'], output_dict=True)
    
    # Prepare the report data
    report_data = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):  # Avoid summary rows
            report_data.append({
                'Topic': topic,
                'Model': model_name,
                'Label': label,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    # Convert to DataFrame and save to CSV
    classification_report_df = pd.DataFrame(report_data)
    output_file = os.path.join(output_dir, f'results_zero_shot_{topic}_only_topic.csv') # for title only
    # output_file = os.path.join(output_dir, f'results_zero_shot_{topic}_w_desc.csv')  # for title and description
    # output_file = os.path.join(output_dir, f'results_few_shot_{topic}.csv')  # for title, description, and examples
    classification_report_df.to_csv(output_file, index=False)
    
    print(f"Classification report for topic '{topic}' saved to {output_file}.")
