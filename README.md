# atidotTH
# **Customer Churn Classification**

## **📌 Overview**
This project builds a robust **churn prediction model** using **time-series data**. The model classifies whether a customer will **churn** between **Jan 1, 2024 – Feb 28, 2024**, based on their transaction history.

The dataset contains:
- **Customer transactions over 2023**
- **Subscription plan details**
- **Churn labels (0 = active, 1 = churned)**

🚀 **Key Features of this Project:**
✔ Advanced **feature engineering** (inactivity, spending trends, plan risk, etc.)  
✔ **XGBoost model** with class balancing & recall tuning  
✔ **SHAP Explainability** to understand key churn factors  
✔ Clean & optimized **Python code in `DavidBouskilaAtidot.py`**

---

## **🛠 Setup & Installation**
### **1️⃣ Install Dependencies**
Ensure you have Python 3.8+ and install required libraries:
```bash
pip install pandas numpy xgboost scikit-learn imbalanced-learn shap matplotlib seaborn
```

### **2️⃣ Run the Model**
Simply execute the `DavidBouskilaAtidot.py` script:
```bash
python DavidBouskilaAtidot.py
```
This will:
- **Load and preprocess the dataset** (`churn_data.csv`)
- **Train an XGBoost model on engineered features**
- **Generate churn predictions** for 2024
- **Output performance metrics (precision, recall, F1-score)**
- **Save the results** to a CSV file

---

## **📊 Feature Engineering**
To improve model accuracy, the following features were engineered:
1️⃣ **`normalized_inactivity`** → Time since last transaction, adjusted for tenure  
2️⃣ **`spending_change`** → Rolling spending trends (helps detect declining users)  
3️⃣ **`standard_plan_risk`** → Higher risk for Standard plan users based on inactivity  

💡 **Why?** These features were identified using **SHAP analysis** to **increase recall & precision**.

---

## **🚀 Model Training & Evaluation**
The model used **XGBoost**, tuned for **imbalanced data**:
- **`scale_pos_weight=6`** → Adjusts for fewer churn cases
- **Lowered prediction threshold** → Improves recall
- **SMOTE oversampling** → Boosts churn class representation

After training, the model achieved:
```bash
Precision: 51.2%
Recall: 68.8%
F1-score: 58.7%
```
(Exact values depend on dataset variability.)

---

## **📂 Output Files**
- **`churn_predictions.csv`** → Predictions for each customer (0 = active, 1 = churned)
- **`metrics.json`** → Model performance metrics (precision, recall, F1-score)
- **`churn_model.pkl`** → Trained XGBoost model (for deployment)

---

## **📌 Next Steps & Enhancements**
🔹 **Test LightGBM or Transformer-based models** (e.g., Temporal Fusion Transformer)  
🔹 **Use external data** (e.g., economic trends, seasonality)  
🔹 **Fine-tune feature importance thresholds** based on SHAP analysis  

📩 **Questions?** Feel free to reach out! 🚀

