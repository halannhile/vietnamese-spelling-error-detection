# Vietnamese Spelling Error Detection

This project implements a Vietnamese Spelling Error Detection system using the VSEC dataset. The system compares classical machine learning models (Logistic Regression, Random Forest, Naive Bayes) and a reimplementation of the VSEC model with some adaptations. 

## Setup Instructions

Install dependencies

```
pip install -r requirements.txt
```

## How to Run the Project

```
python main.py
```

This script performs the following:

* Preprocesses and splits the data into train, dev, and test sets.

* Extracts features for classical baseline models.

* Trains and evaluates Logistic Regression, Random Forest, and Naive Bayes models.

* Fine-tunes the VSEC Transformer-based model and evaluates its performance.

## Project Report

[Link](https://github.com/halannhile/vietnamese-spelling-error-detection/blob/main/NhiLe_COSI114A_Project_Report.pdf)

## Reference

Do, Dinh-Truong, et al. "VSEC: Transformer-based Model for Vietnamese Spelling Correction". [arXiv:2111.00640](https://arxiv.org/abs/2111.00640)

GitHub page for the VSEC dataset can be found [here](https://github.com/VSEC2021/VSEC/tree/main).

