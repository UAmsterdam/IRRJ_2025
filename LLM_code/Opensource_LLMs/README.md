# Cross Topic and Prompt Senstivity analysis for Legal Due Diligence

This repository provides an implementation of **open-source large language models (LLMs)** for **legal due diligence** tasks, including **cross-topic evaluations** and **prompt sensitivity analysis**. Follow the steps below to set up and run the project efficiently.

## **📌 Prerequisites**
Before running the project, ensure you have the following installed:

### **1️⃣ System Requirements**
- **Python 3.8+**
- **Jupyter Notebook** (For interactive execution)
- **Ollama Framework** (For running LLMs locally)

### **2️⃣ Data Preparation**
Ensure the following data files are available:
- **`LLM_data/`**: Directory containing **50 CSV files** (one per topic).
- **`due_dilligence_data.csv`**: Main dataset for legal due diligence.
- **`topics_data.pkl`**: Topic titles and descriptions.

---
## **🛠️ Setup Guide**
### **Step 1: Install Ollama**
Download and install Ollama from the [official website](https://ollama.com/download). Then, pull the required models:

```bash
ollama run llama3.1  # Example for running Llama3.1
```
Run similar commands for other models, such as **Dolphin-llama3**, **Gemma2** or **DeepSeek 8B**.

---
## **🚀 Running the Models**
Execute the appropriate script based on the method you want to use:

### **1️⃣ Cross Topic Experiment**

```bash
python cross_topic_exp.py
```
### **2️⃣ Prompt Sensitivity Analysis**

```bash
python prompt_senstivity_analysis.py
```
## **📊 Results Analysis**
After execution, analyze the results using the following Jupyter Notebook:

```bash
python result_analysis.py
```

## Contribution Guidelines
Feel free to contribute to this project by submitting pull requests or issues. Ensure your contributions are well-documented.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
