# MHC1-Binding
This repository contains a notebook that build different classifiers based on different embedding methods that predicts if a given peptide will be presented by a specific MHC1 protein, identified by its allele name.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Content](#content)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Peptides are short sequences of amino acids that can be displayed on the surface of cells by MHC proteins. MHC class I (MHC1) proteins are found on most cell surfaces and present these peptides to T cells, enabling the immune system to detect signs of infection or abnormal activity within the cell. MHC1 proteins are produced based on genetic variations in MHC1 genes, known as alleles, and each person has a unique set of MHC1 proteins. This diversity allows the immune system to recognize a broad range of potential threats. Identifying the specific peptides presented by MHC proteins is crucial for developing immunotherapies that activate the immune system to target cancer cells. By focusing on certain peptides, these therapies can help the immune system distinguish and attack cancer cells without harming healthy cells, potentially offering a more effective and less toxic alternative to traditional cancer treatments.

The aim of this study is to create a classifier that can predict whether a specific peptide will be presented by a particular MHC1 protein, identified by its allele.

## Installation

You can install these dependencies by running the following command:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository and navigate to the project directory.

2. Launch Jupyter Notebook.

3. Open the `main.ipynb` notebook.

4. Follow the instructions in the notebook to execute the code cells.

## Content

The notebook will guide you through the following steps:

1. **Data Analysis and Processing**:
    - Explore and preprocess the data to ensure it's in a suitable format for modeling.

2. **Data Encoding**:
    - Convert the peptide sequences and MHC-I allele information into numerical representations that can be used by machine learning models.

3. **Machine Learning**:
    - Implement and evaluate traditional machine learning models to establish baseline performance.

4. **Deep Learning**:
    - Use pre-tained protein language model to extract meainingfull embeddings from the dataset. Then, train a linear neural network and compare resutls to baseline.


## Results

| Method         | Balanced Accuracy | Precision-Recall AUC | F1 Score |
|----------------|-------------------|----------------------|----------|
| XGBoost(index)       | 0.838            | 0.809                 | 0.758     |
| XGBoost(FOFE)       | 0.832             | 0.791                 | 0.739    |
| XGBoost(BLOSUM)       | 0.841             | 0.813                | 0.760     |
| LinearNN+CE(ProtTrans)       | 0.869             | 0.827                | 0.777    |
| LinearNN+GCE(ProtTrans)       | 0.868              | **0.848**                | **0.795**     |
| MHCflurry       | **0.904**            | 0.815                | 0.661     |


## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
