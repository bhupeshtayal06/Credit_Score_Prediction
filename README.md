# 📌 Paisabazaar Project – Credit Score Prediction  

## 📖 Project Overview  

The **Paisabazaar Project – Credit Score Prediction** is a machine learning-based system that predicts the creditworthiness of individuals based on their financial and demographic data.  

This project focuses on analyzing customer data, preprocessing it, and applying different classification models to predict whether a person has a **Good, Average, or Poor credit score**.  

The project follows a step-by-step approach:  

1. **Data Understanding & Preprocessing**  
   - Handling missing values  
   - Encoding categorical features  
   - Feature scaling and splitting the dataset  

2. **Model Building**  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - Support Vector Machine (SVM)  
   - K-Nearest Neighbors (KNN)  

3. **Model Evaluation**  
   - Evaluated using Accuracy, Precision, Recall, and F1-Score  
   - Compared performance across all models  

4. **Outcome**  
   - Identified the best-performing model (**XGBoost**) for credit score prediction  
   - Provided insights that can be used by financial institutions like **Paisabazaar** to make faster and more accurate lending decisions.  

This project demonstrates the practical use of **machine learning in the fintech industry** and provides a solid foundation for future enhancements such as hyperparameter tuning, model deployment, and integration with real-time applications.

---

## 📝 Problem Statement  
Credit score is one of the most important factors for financial institutions to evaluate the creditworthiness of individuals.  
Traditional methods are slow and less accurate, therefore in this project, we aim to build a **machine learning-based Credit Score Prediction System** that can predict whether a customer has a good, average, or poor credit score.  

This project is developed using the **Paisabazaar dataset**.  

---


## 🎯 Objectives  
- Analyze and preprocess the dataset.  
- Build multiple ML models for credit score prediction.  
- Evaluate models using performance metrics.  
- Identify the best model.  
- Provide a foundation for future deployment in financial platforms like **Paisabazaar**.  

---

## 📂 Project Structure  

The repository is organized as follows:
```
📦 Credit_Score_Prediction
│
├── Credit_Score_Prediction.ipynb
├── requirements.txt
├── README.md
├── dataset-2 (1).csv
│
├── outputs
│   ├── Model_Comparision.png
```

---

## 🛠️ Tech Stack  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Platform:** Google Colab  

---

## 📑 Input Parameters  

The dataset contains the following input features:  

| Feature Name        | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| Customer_ID         | Unique identifier for each customer                                         |
| Name                | Name of the customer                                                        |
| Age                 | Age of the customer                                                         |
| Gender              | Gender of the customer (Male/Female)                                        |
| Annual_Income       | Annual income of the customer (in currency)                                 |
| Occupation          | Occupation category (e.g., Salaried, Business, Student, etc.)               |
| Monthly_Inhand_Salary | Monthly salary after deductions                                           |
| Num_Bank_Accounts   | Total number of bank accounts                                               |
| Num_Credit_Card     | Number of credit cards held                                                 |
| Interest_Rate       | Average interest rate of loans/credits                                      |
| Num_of_Loan         | Number of active loans                                                      |
| Delay_from_due_date | Average delay (in days) for credit payments                                 |
| Num_of_Delayed_Payment | Number of delayed payments                                               |
| Credit_Mix          | Type of credit mix (Good, Standard, Bad)                                    |
| Outstanding_Debt    | Current outstanding debt                                                    |
| Credit_Utilization_Ratio | Ratio of credit used vs credit limit                                   |
| Payment_Behaviour   | Payment habits (e.g., High_spent_Small_value, Low_spent_Large_value, etc.)  |
| Payment_of_Min_Amount | Whether only minimum payment is made (Yes/No)                            |
| Total_EMI_per_month | Total EMI amount per month                                                  |
| Amount_invested_monthly | Average monthly investment                                              |
| Monthly_Balance     | Average balance left at month-end                                           |

**Target Variable**:  
- **Credit_Score** → Categorized as `Good`, `Standard`, or `Poor`

---

## 📊 Dataset  

The dataset used in this project contains customer demographic, financial, and behavioral information to predict their **Credit Score**.  

- **Source**: Provided dataset (`dataset-2.csv`)  
- **Total Rows**: 1000+ (approx)  
- **Total Features**: 20 input features + 1 target variable (Credit Score)  
- **Target Variable**:  
  - `Good`  
  - `Standard`  
  - `Poor`  

📂 **Dataset is uploaded here.**:  

---

## 🤖 Machine Learning Models Used  
1. Logistic Regression  
2. Random Forest  
3. XGBoost  
4. Support Vector Machine (SVM)  
5. K-Nearest Neighbors (KNN)  

---

## 📊 Model Evaluation

To evaluate model performance, we used the following metrics:

- **Accuracy**: Overall correctness of the model  
- **Precision**: Ability of the model to avoid false positives  
- **Recall**: Ability of the model to detect all true positives  
- **F1-Score**: Harmonic mean of precision and recall  

---

## 📝 Results Snapshot

| Model                | Accuracy | Precision | Recall  | F1-Score |
|-----------------------|----------|-----------|---------|----------|
| Logistic Regression   | 0.6500   | 0.6493    | 0.6500  | 0.6455   |
| Random Forest         | **0.8212** | **0.8221** | **0.8212** | **0.8213** |
| XGBoost              | 0.7658   | 0.7679    | 0.7658  | 0.7664   |
| SVM                  | 0.7176   | 0.7264    | 0.7176  | 0.7187   |
| KNN                  | 0.7603   | 0.7619    | 0.7603  | 0.7608   |

---

## 🏆 Best Performing Model

From the results, **Random Forest** outperformed all other models with:

- **Accuracy:** 82.12%  
- **Precision:** 82.21%  
- **Recall:** 82.12%  
- **F1-Score:** 82.13%  

This makes Random Forest the most suitable model for predicting **Credit Score** in our dataset.

---

## 🔮 Future Work  
- Hyperparameter tuning to improve performance.  
- Feature engineering for deeper insights.  
- Model deployment as a **web application** for real-time credit score prediction.  

---

## Author  
Bhupesh Tayal

---