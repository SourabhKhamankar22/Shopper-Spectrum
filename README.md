#  Shopper Spectrum: E-Commerce Customer Segmentation & Recommendation Engine

## 📌 Problem Statement

In the highly competitive e-commerce landscape, businesses often struggle with customer retention and stagnant Average Order Values (AOV) due to "one-size-fits-all" marketing. Without understanding distinct purchasing behaviors, marketing budgets are wasted, and revenue is left on the table.

**Shopper Spectrum** solves this by analyzing raw e-commerce transactional data to build two Unsupervised Machine Learning systems:

1. **A Customer Segmentation Model** to group buyers into actionable personas (e.g., "High-Value" vs. "At-Risk") for targeted marketing.

2. **A Product Recommendation Engine** to automatically suggest "Frequently Bought Together" items, directly increasing AOV.

---

## ⚙️ Project Workflow & Methodology

### 1. Data Preprocessing & NLP

* **Data Cleaning:** Handled missing Customer IDs, removed negative quantities (cancellations), duplicate records, and engineered standard transactional features.

* **Outlier Treatment:** Applied 95th-percentile capping to extreme wholesale (B2B) buyers to prevent cluster distortion.

* **NLP on Catalog:** Utilized Natural Language Processing (Regex, Stopword Removal, and Lemmatization) to standardize thousands of noisy, manually entered product descriptions.

---

### 2. Customer Segmentation (RFM + K-Means)

* **Feature Engineering:** Calculated **R**ecency (days since last purchase), **F**requency (total purchases), and **M**onetary (total spend) for every customer.

* **Data Transformation:** Applied Log Transformation (`log1p`) and Standard Scaling to reduce skewness and improve clustering performance.

* **Algorithm:** Applied **K-Means Clustering** to the transformed RFM matrix.

* **Optimization:** Built a custom Grid Search loop evaluated via the **Silhouette Score** and tested multiple cluster configurations.

* **Segments Identified:**

  * 💎 High-Value Customers
  * 🛒 Regular Customers
  * 🤔 Occasional Customers
  * 📉 At-Risk Customers

---

### 3. Product Recommendation Engine (Collaborative Filtering)

* **Matrix Creation:** Built a sparse Item-User matrix mapping products to customer purchase quantities.

* **Algorithm:** Implemented **K-Nearest Neighbors (KNN)** using Item-Based Collaborative Filtering.

* **Optimization:** Evaluated multiple distance metrics including:

  * Cosine Similarity
  * Euclidean Distance
  * Manhattan Distance

* **Best Model:** Cosine Similarity produced the strongest recommendation quality for sparse e-commerce purchase data.

* **Output:** Returns the Top 5 most relevant products based on customer purchase behavior.

---

### 4. Deployment

* Serialized optimized models using Joblib:

  * `kmeans_model.joblib`
  * `rfm_scaler.joblib`
  * `knn_recommender.joblib`

* Saved recommendation assets:

  * `pivot_table.pkl`

* Built an interactive, real-time web application using **Streamlit**.

---

## 🌐 Live Demo

### Streamlit Application

**Live App:**
https://shopper-spectrum-grkfypjlrbfr7xw59bbxhp.streamlit.app/

---

## 🌟 Streamlit App Features

### 🛒 Product Recommendation Module

* Features a user-friendly product search interface.
* Utilizes the trained KNN recommendation engine.
* Returns Top 5 similar products instantly.
* Supports "Frequently Bought Together" recommendations.

#### Business Use Case

Can be integrated directly into an e-commerce storefront to increase Average Order Value (AOV).

---

### 👥 Customer Segmentation Module

Accepts user inputs:

* Recency
* Frequency
* Monetary

The application:

* Applies the saved Standard Scaler.
* Predicts the customer segment using the trained K-Means model.
* Displays actionable marketing recommendations based on customer behavior.

Examples:

| Segment       | Suggested Action             |
| ------------- | ---------------------------- |
| 💎 High-Value | Loyalty Rewards & VIP Offers |
| 🛒 Regular    | Upselling & Cross-Selling    |
| 🤔 Occasional | Personalized Promotions      |
| 📉 At-Risk    | Win-Back Campaigns           |

---

## 📊 Exploratory Data Analysis

The project includes extensive EDA covering:

* Transaction Volume by Country
* Top Selling Products
* Revenue by Country
* Monthly Revenue Trends
* Daily Transaction Trends
* Hourly Purchase Activity
* Quantity Distribution
* Unit Price Distribution
* Customer Spending Analysis
* RFM Distributions
* Correlation Heatmap
* Pair Plots
* Elbow Curve
* Silhouette Analysis
* Cluster Visualization
* Product Similarity Heatmap

---

## 🛠️ Tech Stack & Libraries

### Programming Language

* Python

### Data Manipulation

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-Learn
* SciPy

### Natural Language Processing

* NLTK
* Regex

### Model Serialization

* Joblib
* Pickle

### Deployment

* Streamlit

---

## 📂 Project Structure

```text
ShopperSpectrum/
│
├── app.py
├── requirements.txt
├── README.md
│
├── kmeans_model.joblib
├── rfm_scaler.joblib
├── knn_recommender.joblib
├── pivot_table.pkl
│
├── OnlineRetail.csv
│
└── notebooks/
    └── ML_Submission_Template.ipynb
```

---

## 🚀 How to Run the Application

### Prerequisites

Ensure Python is installed on your system.

---

### 1. Clone the Repository

```bash
git clone https://github.com/SourabhKhamankar22/Shopper-Spectrum

cd ShopperSpectrum
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure Required Files are Present

Make sure the following files are available:

* app.py
* kmeans_model.joblib
* rfm_scaler.joblib
* knn_recommender.joblib
* pivot_table.pkl

---

### 4. Launch the Application

```bash
python -m streamlit run app.py
```

or

```bash
streamlit run app.py
```

The application will automatically launch at:

```text
http://localhost:8501
```

---

## 📈 Business Impact

This solution enables businesses to:

* Improve customer retention.
* Increase customer lifetime value.
* Deliver personalized product recommendations.
* Identify churn-risk customers.
* Optimize marketing campaigns.
* Improve inventory planning and demand forecasting.

---

## 🔮 Future Enhancements

* Hybrid Recommendation Systems
* Deep Learning-Based Recommendations
* Customer Lifetime Value Prediction
* Real-Time Recommendation APIs
* Automated Marketing Integrations

---
