# 🏠 Airbnb Price Prediction – CIS 512 Final Project

**Author:** Anand Vekariya
**Course:** CIS 512 — Data Science
**Instructor:** Prof. Sumanlata Ghosh

---

## 📘 Overview

This project focuses on building a **predictive model to estimate Airbnb property prices** across multiple countries using advanced data cleaning, feature engineering, and regression modeling techniques.

The analysis aims to identify **key factors influencing Airbnb prices** and recommend **profitable investment locations and property types**.

---

## 🧹 1. Data Cleaning

### Dataset

* **Source:** Airbnb property listings dataset
* **Records:** ~123,000
* **Countries:** Belgium, Canada, Netherlands, Germany, etc.

### Cleaning Steps

1. **Dropped Irrelevant Columns**
   Removed URL, host images, text descriptions, coordinates, and other non-numeric/uninformative columns.
2. **Standardized City Names**
   Harmonized city name variations (e.g., *Bruxelles* → *Brussels*).
3. **Imputed Missing Values**

   * Replaced empty numeric fields with `0` or estimated values.
   * Derived `Host Age` and `Last Rented (Months)` from date columns.
4. **Processed Amenities**
   Categorized amenities (e.g., *AC/Heating*, *Kitchen*, *Safety*) into binary Yes/No columns.
5. **Feature Adjustments**

   * Linked `Host Location` with property address.
   * Calculated price where missing using weekly/monthly averages.
   * Cleaned and standardized all currency and numeric data.

---

## 🧠 2. Feature Engineering

### Transformations

* **Categorical Encoding:**
  Converted categorical columns to numerical values using **One-Hot Encoding**.
  Encoded:

  * Host Response Time
  * Country
  * Property Type
  * Room Type
  * Bed Type
  * Cancellation Policy

* **Binary Features Created:**
  Presence/absence indicators for amenities, property features, and availability attributes.

### Final Shape

* **Rows:** 123,061
* **Columns:** 91

---

## 📊 3. Data Preparation for Modeling

| Step                         | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| **Correlation Filtering**    | Removed features with correlation < 0.05 with Price        |
| **Train-Test Split**         | 75% training (~92,000 rows), 25% validation (~31,000 rows) |
| **Standardization**          | Applied `StandardScaler` to normalize numerical features   |
| **Dimensionality Reduction** | Reduced feature space to 57 columns                        |

---

## 🤖 4. Predictive Modeling

### Models Implemented

1. **Linear Regression**
2. **Lasso Regression**
3. **Ridge Regression**
4. **Random Forest Regression**

### Evaluation Metrics

* **R² (Coefficient of Determination)**
* **Adjusted R²**
* **Mean Squared Error (MSE)**
* **Root Mean Square Error (RMSE)**
* **Mean Absolute Percentage Error (MAPE)**

---

## 🏆 5. Model Comparison

| Model                        | Train R² | Validation R² | MSE        | RMSE      |
| ---------------------------- | -------- | ------------- | ---------- | --------- |
| Linear Regression            | 0.55     | 0.56          | 21,416     | 146.3     |
| Ridge Regression             | 0.55     | 0.56          | 21,416     | 146.3     |
| Lasso Regression             | 0.55     | 0.56          | 21,416     | 146.3     |
| **Random Forest Regression** | **0.61** | **0.60**      | **18,809** | **137.1** |

### 🥇 **Best Model:** Random Forest Regression

**Key Highlights:**

* Best generalization with consistent training/validation results.
* ~12% lower MSE and ~6% lower RMSE compared to linear models.
* Handles complex relationships and non-linear price patterns effectively.

---

## 💡 6. Model Insights

* Model predicts Airbnb prices **ranging from $30 to $550**.
* Highest accuracy observed between **$150–$330 price range**.
* Random Forest model demonstrates stable and realistic price estimation.

---

## 🌍 7. Key Findings

| Country         | Insights                                                                        |
| --------------- | ------------------------------------------------------------------------------- |
| **Canada**      | Toronto & Vancouver are reliable for investment with consistent price patterns. |
| **Netherlands** | Amsterdam & Rotterdam show stable growth potential.                             |
| **Germany**     | Berlin & Munich indicate high demand and accurate model predictions.            |

**Investment Recommendation:**
Focus on **apartments and townhouses** in major urban centers for optimal returns.

---

## ⚙️ 8. Tech Stack

| Category                | Tools                                            |
| ----------------------- | ------------------------------------------------ |
| **Languages**           | Python 3.11                                      |
| **Libraries**           | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| **Environment**         | Jupyter Notebook                                 |
| **Visualization**       | Matplotlib, Seaborn                              |
| **Modeling Techniques** | Linear, Ridge, Lasso, Random Forest              |
| **Data Storage**        | Excel (.xlsx)                                    |

---

## 📈 9. Outputs

* **`Raw_Final_Project_Data.xlsx`** → Original dataset
* **`Cleaning_Final_Clean_Project_Data.xlsx`** → Cleaned and processed dataset
* **`Modeling_Final_Clean_Project_Data.xlsx`** → Feature-engineered dataset
* **`predicted_price_Final_Project_Data.xlsx`** → Final predictions with actual vs predicted price

---

## 📚 10. Conclusion

✅ Random Forest Regression provides the most reliable Airbnb price predictions.
✅ The model successfully generalizes across multiple countries.
✅ Investment opportunities are strongest in **Canada, Netherlands, and Germany**.
✅ Apartments and townhouses yield the best return-to-price consistency.

---


### 👤 Author

**Anand Vekariya**

📧 **Email:** [[anand.d.vekariya@gmail.com](mailto:anand.d.vekariya@gmail.com)]

💼 **LinkedIn:** [Anand Vekariya](https://www.linkedin.com/in/anand-vekariya/)

<!-- 🌐 **Portfolio:** [https://anandvekariya.netlify.app](https://anandvekariya.netlify.app) -->
---