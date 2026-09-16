# 🏠 House Price Prediction

An end-to-end **Machine Learning project** that predicts house prices based on property characteristics such as area, bedrooms, bathrooms, location, and other relevant features.

The project covers the complete machine learning workflow:

**Data → Exploration → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction → Streamlit Deployment**

---

## 📌 Project Overview

House prices depend on multiple factors including property size, number of rooms, location, amenities, and other property characteristics.

The objective of this project is to build a machine learning model that can learn relationships between these features and historical house prices, then use the trained model to predict prices for new properties.

### Business Problem

> How can historical housing data be used to estimate the expected selling price of a property?

### Solution

Build an end-to-end regression pipeline that:

1. Loads the housing dataset
2. Performs data quality checks
3. Performs Exploratory Data Analysis
4. Handles missing values and categorical variables
5. Performs feature engineering
6. Trains multiple regression models
7. Evaluates model performance
8. Selects the appropriate model
9. Saves the trained model
10. Provides predictions through a Streamlit web application

---

# 🎯 Objectives

* Understand the factors affecting house prices
* Perform data cleaning and preprocessing
* Analyze relationships between features and house prices
* Build regression models
* Compare model performance
* Optimize the selected model
* Create a reusable prediction pipeline
* Deploy the model using Streamlit
* Provide an interactive interface for house price prediction

---

# 🛠️ Technology Stack

| Category             | Tools                      |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Data Manipulation    | Pandas, NumPy              |
| Data Visualization   | Matplotlib, Seaborn        |
| Machine Learning     | Scikit-learn               |
| Model Persistence    | Joblib / Pickle            |
| Web Application      | Streamlit                  |
| Development          | Jupyter Notebook / VS Code |
| Version Control      | Git & GitHub               |

---

# 📂 Project Structure

```text
House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── model.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> Update the folder names if your actual project structure is different.

---

# 🔄 Machine Learning Workflow

```text
                ┌─────────────────┐
                │   Raw Dataset   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Data Validation  │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Data Cleaning    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │      EDA         │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │Feature Engineering│
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Preprocessing    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Train/Test Split │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Model Training   │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Model Evaluation │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Best Model       │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Model Persistence│
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Streamlit App    │
                └────────┬────────┘
                         ↓
                ┌─────────────────┐
                │ Price Prediction │
                └─────────────────┘
```

---

# 📊 Dataset

The dataset contains historical information about residential properties.

Typical features may include:

* Property area
* Number of bedrooms
* Number of bathrooms
* Number of floors
* Location
* Property type
* Parking availability
* Furnishing status
* Age of property
* Other property characteristics

### Target Variable

```text
Price
```

The target variable represents the estimated/selling price of the property.

> The exact features and target column depend on the dataset used in this project.

---

# 🔍 Exploratory Data Analysis

The EDA phase focuses on understanding the dataset and identifying patterns.

### Data Quality Analysis

* Number of rows and columns
* Data types
* Missing values
* Duplicate records
* Unique values
* Outliers
* Numerical distributions
* Categorical distributions

### Statistical Analysis

```python
df.describe()
```

The analysis helps understand:

* Mean
* Median
* Standard deviation
* Minimum
* Maximum
* Quartiles

---

# 📈 Data Visualization

Important visualizations include:

### Distribution of House Prices

```text
House Price
    │
    │       █
    │     █ █
    │   █ █ █
    │ █ █ █ █
    └──────────────
       Price Range
```

### Correlation Analysis

A correlation matrix can be used to identify relationships between numerical variables.

### Feature vs Price Analysis

Examples:

* Area vs Price
* Bedrooms vs Price
* Bathrooms vs Price
* Location vs Price
* Property Age vs Price

---

# 🧹 Data Preprocessing

The preprocessing pipeline may include:

### Missing Value Treatment

Numerical features:

```python
df[column].fillna(df[column].median())
```

Categorical features:

```python
df[column].fillna(df[column].mode()[0])
```

### Duplicate Removal

```python
df.drop_duplicates()
```

### Categorical Encoding

Categorical variables can be converted into numerical representations using techniques such as:

* One-Hot Encoding
* Ordinal Encoding

### Feature Scaling

Depending on the selected algorithm, numerical features can be standardized using:

```python
StandardScaler()
```

---

# 🧠 Machine Learning Models

This project can compare multiple regression algorithms.

### 1. Linear Regression

```text
Simple baseline regression model
```

### 2. Decision Tree Regressor

Captures nonlinear relationships between features.

### 3. Random Forest Regressor

An ensemble of multiple decision trees designed to improve predictive performance and robustness.

### 4. Gradient Boosting

Builds models sequentially to reduce prediction errors.

### 5. Other Models

Depending on the dataset, additional algorithms can be evaluated, such as:

* XGBoost
* HistGradientBoostingRegressor
* Random Forest
* Extra Trees Regressor

---

# 📏 Model Evaluation

Regression models can be evaluated using:

### MAE — Mean Absolute Error

```text
MAE = Average(|Actual - Predicted|)
```

Lower MAE indicates smaller average prediction errors.

### MSE — Mean Squared Error

```text
MSE = Average((Actual - Predicted)²)
```

### RMSE — Root Mean Squared Error

```text
RMSE = √MSE
```

RMSE gives greater weight to larger errors.

### R² Score

```text
R² = 1 - SSres / SStotal
```

R² measures how much of the variation in the target variable is explained by the model.

---

# 📊 Model Comparison

Example evaluation table:

| Model             | MAE | RMSE | R² |
| ----------------- | --: | ---: | -: |
| Linear Regression |   — |    — |  — |
| Decision Tree     |   — |    — |  — |
| Random Forest     |   — |    — |  — |
| Gradient Boosting |   — |    — |  — |
| XGBoost           |   — |    — |  — |

The final model should be selected based on the evaluation results and project requirements rather than assuming a particular algorithm will always perform best.

---

# 💾 Model Saving

After selecting and training the final model, it can be saved using Joblib:

```python
import joblib

joblib.dump(model, "models/model.pkl")
```

The saved model can then be loaded by the prediction application:

```python
model = joblib.load("models/model.pkl")
```

---

# 🌐 Streamlit Application

The project includes a Streamlit application that allows users to enter property information and receive a predicted house price.

### Application Workflow

```text
User Input
    ↓
Input Validation
    ↓
Preprocessing
    ↓
Trained ML Model
    ↓
Prediction
    ↓
Estimated House Price
```

Run the application with:

```bash
streamlit run app.py
```

The application provides an interactive interface for:

* Entering property features
* Submitting prediction requests
* Viewing the predicted house price

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Lalit-Shishodia/House-Price-Prediction.git
```

## 2. Navigate to the project

```bash
cd House-Price-Prediction
```

## 3. Create a virtual environment

```bash
python -m venv .venv
```

## 4. Activate the environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
.venv\Scripts\activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created yet:

```bash
pip freeze > requirements.txt
```

---

# ▶️ Run the Project

### Run the Streamlit application

```bash
streamlit run app.py
```

### Run training

If your project has a training script:

```bash
python src/train.py
```

---

# 🔮 Example Prediction

Example input:

```text
Area          : 1500 sq ft
Bedrooms      : 3
Bathrooms     : 2
Location      : Hapur
Property Type : Apartment
```

The application processes the input and returns an estimated property price.

---

# 📌 Key Learnings

Through this project, the following Data Science concepts are demonstrated:

* Python programming
* NumPy
* Pandas
* Data cleaning
* Exploratory Data Analysis
* Statistical analysis
* Feature engineering
* Categorical encoding
* Feature scaling
* Regression algorithms
* Model evaluation
* Hyperparameter tuning
* Model serialization
* Prediction pipelines
* Streamlit deployment
* Git and GitHub

---

# 🚀 Future Improvements

Potential improvements include:

* Hyperparameter optimization
* Cross-validation
* Feature selection
* Advanced ensemble models
* XGBoost/LightGBM experimentation
* Model explainability using SHAP
* MLflow experiment tracking
* Docker deployment
* Cloud deployment
* Automated CI/CD
* Real-time property data integration
* Location-based feature engineering

---

# 💼 Project Use Case

This project demonstrates how Machine Learning can be used to support:

* Real-estate price estimation
* Property valuation
* Buyer decision support
* Real-estate analytics
* Market analysis
* Property investment analysis

The prediction should be treated as a model-based estimate and not as a guaranteed market valuation.

---

# 👨‍💻 Author

**Lalit Shishodia**

Aspiring Data Scientist / ML Engineer

**Skills:** Python • SQL • Power BI • Pandas • NumPy • Machine Learning • Data Analysis • Streamlit

---

# ⭐ If You Find This Project Useful

Feel free to explore the repository, experiment with the model, and improve the project with additional features and algorithms.

**Learn • Build • Analyze • Innovate**
