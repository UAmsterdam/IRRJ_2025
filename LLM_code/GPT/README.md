# Closed-Source LLMs for Legal Due Diligence

This repository contains code for utilizing **closed-source large language models (LLMs)** in **legal due diligence** tasks using **GPT-4o and GPT-4o-mini** via the OpenAI API. The implementation supports **zero-shot** and **few-shot** learning methodologies with **automated evaluation and result tracking**.

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
### **Step 1: Get API key**
- To utilize GPT-4o and GPT-4o-mini, obtain an API key from the [OpenAI official website](https://openai.com/api/).

## **🚀 Running the Models**
Execute the appropriate script based on the method you want to use:

### **1️⃣ Zero-Shot Learning**
**(a) Title-Only Prediction:**
```bash
python GPT_title_only.py
```
**(b) Title + Description::**
```bash
python GPT_title+description.py

```
### **2️⃣ Few-Shot Learning**
**(Title + Description + Examples):**
```bash
python GPT_title+description+examples).py
```
## **📊 Results Analysis**
After execution, analyze the results using the following Jupyter Notebook:

```bash
python results_analysis.py
```

## Contribution Guidelines
Feel free to contribute to this project by submitting pull requests or issues. Ensure your contributions are well-documented.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
