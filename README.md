# BERT-based Hate Speech Classification
## Project description
- This project concerns the implementation of automatic hate speech classifiers using transformer-based natural language models such as Distil-RoBERTa and Distil-BERT.
- We are not interested in BERT or RoBERTa because the architectures are so large that they could not be fit into our available computational resources. 
- The trained aforementioned models will be evaluated with baselines such as Naive Bayes classifier and Logistic Regression classifier with Word2Vec and GloVe as word vectors to embed the raw input.
- Members: Anh-Duy Pham, Hasnaili Walli.
## How to reproduce?
### Requirements
The project requires the preinstallation of the following Python packages:
- transformers
- transformers-interpret
- datasets
- scikit-learn
- seaborn
- numpy
- matplotlib
- torch
### Steps to reproduce
- The results can be reproduced by running all the cells in the corresponding Jupyter notebooks with the method name of interest.
## Results
### Automatic evaluation
- Since this is a classification problem, it is popular to use the following metrics: accuracy, precision, recall and macro F1 score.

| Methods  | Accuracy | Precision | Recall | Macro F1-Score |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Distil-BERT  | 0.66 | 0.66 | 0.66 | 0.65 |
| Distil-RoBERTa  | **0.67** | **0.67** | **0.67** | **0.66** |
| GloVe + LR (baseline)  | 0.59 | 0.59 | 0.59 | 0.58 |
| Word2Vec + LR (baseline)  | 0.60 | 0.59 | 0.60 | 0.58 |
| GloVe + NaiveBayes (baseline)  | 0.41  | 0.17 | 0.41 | 0.23 |
| Word2Vec + NaiveBayes (baseline)  | 0.41 | 0.32 | 0.41 | 0.24 |
| WordCount + NaiveBayes (baseline) | 0.63 | 0.62 | 0.63 | 0.62 |
### Manual evaluation
- We took randomly 20 examples from the HateXplain testset without analyzing
the true labels and the corresponding predictions from both Distil-BERT and Distil-RoBERTa to survey the manual evaluation from the students in WS 21/22 NLP course. Specifically, the students need to rate from 1 (worst) to 4 (best) for each prediction. There were 9 responses in total in the survey. The results are averaged for each model among Distil-BERT and Distil-RoBERTa and normalized to (0,1) range to demonstrate an appropriate accuracy of each model, as illustrated as follows:

| Methods  | Accuracy |
| ------------- | ------------- |
| Distil-BERT  | 0.754  |
| Distil-RoBERTa  | **0.766**  |
