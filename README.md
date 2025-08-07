# 🧠 User360 Customer Intelligence Engine

This repository contains a Jupyter notebook detailing the development of a **User360 Customer Intelligence Engine**. The project leverages **PySpark** to perform customer segmentation, churn prediction, lifetime value (LTV) prediction, and personalized recommendations using an e-commerce dataset from Kaggle.

---

## 📌 Project Overview

The core objective of this project is to create a **comprehensive understanding of user behavior and value**. By analyzing a dataset of user events, we build a multi-faceted engine to:

- **Segment Users**: Identify distinct user groups using K-Means clustering.
- **Predict Churn**: Use Logistic Regression to flag users likely to churn.
- **Estimate Lifetime Value (LTV)**: Predict user value using Gradient Boosted Trees.
- **Generate Recommendations**: Build a collaborative filtering engine using ALS.
- **Evaluate Campaigns**: Simulate uplift modeling to identify persuadable users.

---

## 📦 Dataset

We use the [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset):

- `events.csv`: User interaction events (e.g., `view`, `addtocart`, `transaction`)
- `item_properties_part1.csv` & `item_properties_part2.csv`: Item property details
- `category_tree.csv`: Hierarchy of product categories

The notebook includes code to automatically download and prepare the dataset.

---

## 🔍 Methodology

### 1. Data Ingestion & Preprocessing
- Authenticate with Kaggle and download data.
- Load into PySpark DataFrames.
- Clean and merge item properties, convert timestamps, handle nulls.

### 2. Feature Engineering
Key features include:
- `sessions`: Distinct user sessions  
- `total_events`: Total interactions  
- `unique_items`: Unique items viewed  
- `avg_time_between_events`: Avg. time gap between actions  
- `view_cart_ratio`: Ratio of `add to cart` vs `view`  
- `recency_days`: Days since last activity  

### 3. User Segmentation
- Standardize and vectorize features.
- Apply **K-Means Clustering**.
- Use **PCA** for 2D visualization of user clusters.

### 4. Churn Prediction
- Label users as churned if inactive >30 days.
- Train a **Logistic Regression** classifier.
- Evaluate using **AUC (Area Under Curve)**.

### 5. LTV Prediction
- Define LTV as number of purchases in last 60 days.
- Train **Gradient Boosted Trees (GBT) Regressor**.
- Evaluate with **Root Mean Squared Error (RMSE)**.

### 6. Recommendation System
- Use **ALS (Alternating Least Squares)** for collaborative filtering.
- Encode implicit feedback: `view = 1`, `transaction = 2`.
- Generate **top-5 item recommendations** per user.

### 7. Uplift Modeling (Campaign Targeting)
- Simulate uplift modeling using synthetic data.
- Train separate **Random Forest classifiers** for treatment and control.
- Identify “persuadable” users who benefit most from targeted marketing.

### 8. Streamlit Dashboard
- Basic interactive dashboard for:
  - User segments and behavior
  - Churn & LTV insights
  - Recommendation previews
  - Campaign effectiveness

---

## 🚀 How to Run

1. Open the Jupyter Notebook in **Google Colab**.
2. Follow instructions cell-by-cell to:
   - Install dependencies (`kaggle`, `pyspark`, `streamlit`, etc.)
   - Set up your Kaggle API key (`kaggle.json`)
   - Download and unzip dataset
   - Execute code blocks to build models
3. To run the **Streamlit Dashboard**, use `pyngrok` (requires an ngrok authtoken).

---

## 🛠 Tech Stack

- Python
- PySpark
- Scikit-learn
- Streamlit
- PCA, ALS, GBT, Logistic Regression, K-Means
- Kaggle API, pyngrok

---

## 📈 Use Cases

- Customer 360 analytics for e-commerce
- Marketing automation (churn prevention & campaign targeting)
- Recommendation systems
- Business intelligence dashboards

---

User360/
│
├── User360_Customer_Intelligence_Engine.ipynb # Main Jupyter notebook
├── requirements.txt # Python dependencies
├── kaggle.json # API key (user-specific, not included)
└── README.md # Project documentation (this file)
## 📁 Repository Structure

