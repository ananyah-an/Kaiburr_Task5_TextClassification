# Kaiburr Task 5 – Consumer Complaint Text Classification

This project classifies consumer complaint narratives into financial product categories using Natural Language Processing (NLP) and machine learning.

🧠 Models Used
- Multinomial Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)

🧾 Dataset
The dataset contains consumer complaints with their associated product categories.
Dataset Source: [Consumer Complaint Database - Data.gov](https://catalog.data.gov/dataset/consumer-complaint-database)

Filtered categories used:
1. Credit reporting, credit repair services, or other personal consumer reports  
2. Debt collection  
3. Consumer Loan  
4. Mortgage  

🧹 Preprocessing
- Lowercasing  
- Removing punctuation and stopwords  
- TF-IDF vectorization  

 🧪 Results
| Model | Accuracy |
|--------|-----------|
| Naive Bayes | 0.870 |
| Logistic Regression | 0.899 |
| SVM | 0.896 |

🏆 Best Model: Logistic Regression

⚙️ How to Run
```bash
python consumer_complaint_classification.py
