# Open-source LLMs for Legal Due Diligence

This repository contains code for utilizing open-source large language models (LLMs) in legal due diligence tasks. Follow these steps to set up and run the project efficiently.

## Prerequisites

Ensure you have Python installed on your system. The codebase uses Python scripts and Jupyter Notebooks.

## Getting Started

### Steps Overview

![Steps Overview](./steps_overview.jpg)

### Step 1: Data Acquisition
- Request the dataset from the original authors by visiting their [GitHub repository](https://github.com/zuvaai/science/tree/master/core-tech). Follow their guidelines for accessing the data.

### Step 2: Data Preparation
- Convert the obtained data to a CSV file using `data_creation.py`.

### Step 3: Evaluation Data Setup
- Generate a subset of the dataset for evaluation purposes with `evaluation_data_creation.ipynb`.

### Step 4: Prompt Creation
- Create templates for each topic using the `prompt_creation.ipynb` notebook.

### Step 5: LLM Setup
- Download and install the ollama framework from the [official ollama website](https://ollama.com/download).
- Load or download models from ollama by running the following commands in your terminal:
  ```bash
  ollama run llama3.1

### Step 6: Model Evaluation
With the evaluation data, prompts, and model set up, run `LLMs-evaluation-code-using-ollama.ipynb`. Ensure you update the paths and topic ID in the notebook.

### Step 7: Using GPT-4
- To utilize GPT-4o-mini, obtain an API key from the [OpenAI official website](https://openai.com/api/).
- Run the `GPT-4-code.ipynb` with the API key.

### Step 8: Results Analysis
- Analyze and interpret the results using `results_analysis.ipynb`.

## Contribution Guidelines
Feel free to contribute to this project by submitting pull requests or issues. Ensure your contributions are well-documented.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
