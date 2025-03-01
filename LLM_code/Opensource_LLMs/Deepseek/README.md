# **DeepSeek 8B LLM for Legal Due Diligence**

This repository contains code for leveraging the **DeepSeek 8B** open-source large language model (LLM) for **legal due diligence** tasks. The project aims to classify and analyze legal documents efficiently using **zero-shot** and **few-shot** learning techniques.

---

## **🚀 Features**
- **Zero-Shot Learning (ZSL)**: Title-only and title + description models.
- **Few-Shot Learning (FSL)**: Incorporates examples for improved accuracy.
- **Automated Evaluation**: Saves results dynamically for each prompt variation.

---

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
ollama run deepseek-r1:8b
```
## **🚀 Running the Models**
Execute the appropriate script based on the method you want to use:

### **1️⃣ Zero-Shot Learning**
**(a) Title-Only Prediction:**
```bash
python Deepseek_R1_zero_shot_only_title.py
```
**(b) Title + Description::**
```bash
python Deepseek_R1_zero_shot_T+D.py

```
### **2️⃣ Few-Shot Learning**
**(Title + Description + Examples):**
```bash
python Deepseek_R1_few_shot_T+D+E.py
```
## **📊 Results Analysis**
After execution, analyze the results using the following Jupyter Notebook:

```bash
python Result_analysis.py
```

## Contribution Guidelines
Feel free to contribute to this project by submitting pull requests or issues. Ensure your contributions are well-documented.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
