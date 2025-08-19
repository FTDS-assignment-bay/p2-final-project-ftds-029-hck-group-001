# 🎯 E-Quinox: Contextual Bandit Optimization for Real-Time Marketing Decisions

## 📌 Overview
This repository contains the implementation and analysis of a **Contextual Bandit** model designed to optimize real-time marketing decisions for e-commerce. The project leverages reinforcement learning to balance exploration and exploitation, delivering personalized recommendations that maximize user conversion rates.

---

## 🚀 Problem Statement
**"How can we increase marketing conversion rates by providing real-time, context-aware recommendations while efficiently handling cold-start problems and delayed user feedback?"**

### Background:
- 62% of users expect personalized recommendations, but 78% leave due to irrelevant suggestions (*Salesforce, 2023*).
- Cold-start problems: New products receive 5x fewer clicks in the first 48 hours (*ACM Journal, 2022*).
- Delayed feedback: 61% of e-commerce transactions occur more than 24 hours after the click (*KDD Research, 2021*).

---

## 🎯 Objectives
1. **Increase Conversion Rate**  
   - Deliver highly relevant real-time product recommendations using CTR and Purchase Conversion Rate as KPIs.

2. **Overcome Cold-Start & Exploration Challenges**  
   - Implement a Multi-Armed Bandit (MAB) algorithm to balance exploration of new products and exploitation of popular ones.

3. **Reduce Churn & Improve Retention**  
   - Use delayed feedback to personalize recommendations per user session.

---

## 📂 Dataset
**Source:** [Retailrocket Recommender System Dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset)  
**Highlights:**
- 20.7 million real user events
- 1.4 million unique users
- Dynamic item properties (price/availability)
- Hierarchical category structure

---

## 🔍 Key Insights from EDA

### 📊 Event Type Distribution
- **Views:** 2.66M events
- **Add to Cart:** 69K events (2.6% of views)
- **Transactions:** 22K events (0.8% of views)

### ⏰ Peak Activity Hours
- Highest activity: 5 PM – 10 PM and 12 AM – 4 AM
- Lowest activity: 7 AM – 11 AM

### 📅 Weekly Activity
- Stable activity on weekdays (Mon–Fri)
- Significant drop on Saturdays, slight recovery on Sundays

### 🧭 Conversion Funnel
- Major drop from View → Add to Cart
- Only 32% of cart additions lead to purchase

---

## 🛠️ Methodology

### 🔧 Tech Stack
| Category           | Tools/Libraries               |
|--------------------|-------------------------------|
| Programming        | Python 3.9+                   |
| Data Processing    | pandas, numpy                 |
| Machine Learning   | scikit-learn, PyMC3 (optional)|
| Workflow           | Apache Airflow                |
| Database           | PostgreSQL                    |
| Visualization      | matplotlib, Streamlit         |

### 📦 ETL Pipeline
```
   start >> new_data >> loaded_data >>updated_data>>timestamp_change1>>features1 

    new_data >> equinox_bandit>> timestamp_change2>>features2>>update_bandit>>saved_bandit

```

# 🤖 Model: Logistic Regression Bandit

## ⚙️ Mechanism:
- Separate logistic regression model per action
- Input: Contextual features (user + event data)
- Output: Conversion probability (0–1)
- Action selection via **Thompson Sampling**
- Binary reward update (1 = conversion, 0 = no conversion)

## 📈 Prediction Formula:
<div align="center">

$P(Y=1|X) = \frac{1}{1 + e^{-(b_0 + b_1X)}}$

</div>

---

# 📊 Model Evaluation

## 📈 Action Selection Over Time
- **Actions tested:** `email_no_discount`, `email_10%_discount`, `banner_limited_time_offer`, `popup_abandoned_cart_reminder`
- **Model successfully converged to optimal action:** `email_no_discount`

## 🎯 Results:
- **Optimal Baseline:** `email_no_discount`
- **Conversion Rate:** 6.3%
- **Bandit Performance:** 6.3%
- **Regret:** 0.000

---

# 🧠 Prediction Example:
- **Recommended Action:** `email_no_discount`
- **Predicted Conversion Lift:** **97.8%**
- **Action Probabilities:**
  - `email_no_discount`: 0.9981
  - `email_10%_discount`: 0.0998
  - `banner_limited_time_offer`: 0.2144
  - `popup_abandoned_cart_reminder`: 0.0817

---

# 🧩 Challenges
- JSON serialization conflicts in Airflow DAGs
- Docker-PostgreSQL connectivity issues
- Complexity of Reinforcement Learning and marketing domain knowledge
- Defining effective marketing actions for the bandit model

---

# 🚀 Improvements & Future Work

## 🔧 Infrastructure:
- Improve Docker-PostgreSQL network configuration
- Use Parquet or custom serializers to avoid JSON issues in Airflow

## 🧠 Model Enhancements:
- Explore advanced RL methods (Random Forest, Bayesian Regression)
- Implement more complex marketing actions tailored to business context
- Integrate deep learning for better feature representation

---

# 👥 Team E-Quinox
| Role               | Name                 | Contact               |
|--------------------|----------------------|-----------------------|
| Data Engineer      | Nugroho Damar W.     | noreply@hacktiv8.com  |
| Data Scientist     | Rd. Ladityarsa I.    | noreply@hacktiv8.com  |
| Data Analyst       | Khalif Prabowo S.    | noreply@hacktiv8.com  |

---

# 🔗 Links
- **Deployment:** [Streamlit App](https://finalproject-equinox.streamlit.app)
- **Slide Deck:** [PPT](https://docs.google.com/presentation/d/1nzJ24EUKhpTNzzqIVvu6u9J3QWJKSTz5wdN4t7OtcZs/edit?slide=id.p5#slide=id.p5)

---

# 📚 References
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*
- Retailrocket Dataset: [Kaggle](https://www.kaggle.com/retailrocket/ecommerce-dataset)
