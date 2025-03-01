# Effectiveness of In-Context Learning for Due Diligence

## 📌 Overview
This repository provides code and documentation for a reproducibility study on Large Language Models (LLMs) in legal due diligence. It includes experiments from our paper, highlighting the effectiveness of in-context learning in legal analytics. The repository is structured to support both traditional machine learning (CRFs) and LLM-based approaches, enabling comprehensive exploration of legal due diligence automation.

## 📂 Repository Structure
The repository is divided into three main directories, each corresponding to an experiment discussed in our paper:

1. **CRF_raw_data**  
   This directory contains code for the experiment where we explored the performance of Conditional Random Fields (CRFs) using our own simple feature sets. Detailed instructions for running the code are available in its README file.

2. **python_replication**  
    Here, you will find the Python replication of the original CRF experiment from the original paper. Instructions for replicating the study and running the code are provided in the README file.

3. **LLM_code**  
   This directory focuses on the exploration of Large Language Models (LLMs) for the due diligence task. All the instructions to replicate the experiments and run the code are included in the README file.

## Getting Started
- To get started with this repository:
### Step 1: Clone Repo
- Clone the repo using `git clone <repo-url>`
### Step 2: Data Acquisition
- Request the dataset from the original authors by visiting their [GitHub repository](https://github.com/zuvaai/science/tree/master/core-tech).      - Follow their guidelines for accessing the data.
### Step 3: Install Requirements
- To install all dependencies, run:
 ```bash
pip install -r requirements.txt
```
### Step 4: Data Preparation
- Convert the obtained data to a CSV file using `data_creation.py`.
### Step 5: LLM data
- To use the LLM code you should request the LLM data from us. We can provide it for experimental purpose. 
### Step 5: Nevigate to Experiments
- Navigate to the desired experiment directory.
- Follow the instructions in the corresponding README file to set up and run the experiments.

## Contributing
We welcome contributions to improve the experiments and documentation. If you would like to contribute, please fork the repository and submit a pull request.

## License
This project is licensed under the terms of the MIT license.

## Contact
For questions and feedback, please open an issue in this repository or contact the contributors directly through GitHub.

## References and Resources

### Original Paper Code
The code used in the original paper can be accessed on GitHub. For more details, please visit:
[Original Paper Code](https://github.com/zuvaai/science)

### Dataset License
The Kira dataset used in the original study requires a license for use. To request a license or for more information, please contact Zuva directly.

