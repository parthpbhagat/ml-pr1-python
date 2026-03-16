# House Price Prediction using Machine Learning 🏡📊

## 📌 Project Overview
This project aims to build, evaluate, and compare various machine learning regression models to accurately predict real estate prices. By transitioning from a raw dataset to a highly optimized machine learning pipeline, this project explores foundational linear models, polynomial transformations, and advanced optimization algorithms like Batch and Stochastic Gradient Descent.

## 🗂️ Dataset
The dataset contains real estate property details. The objective is to predict the house price based on key features.
* **Target Variable:** `house_price_inr`
* **Key Features:** `area_sqft`, `bedrooms`, `bathrooms`, `location_score`
* **Raw Data:** `RealEstate_HousePrice_Dataset.csv` (Contains 4,200 records with multiple features including age, distance to city, etc.)
* **Processed Data:** `after_houseprice_dataset.csv` (Cleaned data with the most highly correlated features selected).

## 📂 Project Structure

The project is divided into several Jupyter Notebooks, each focusing on a specific step in the data science lifecycle:

1. **`data_preprocessing.ipynb`** - Handles Exploratory Data Analysis (EDA).
   - Identifies correlations, removes multicollinearity, and selects key features (`area_sqft`, `location_score`, etc.).
   - Exports the cleaned dataset for modeling.

2. **`simple_linear_regraction.ipynb`**
   - Implements a baseline Multiple Linear Regression model using Scikit-Learn.
   - Evaluates the initial linear relationship between features and house prices.

3. **`polinomial_linear_regraction.ipynb`**
   - Applies Polynomial Regression to capture complex, non-linear market trends.
   - Utilizes 3D visualizations to plot the multi-dimensional prediction plane.

4. **`batch_grediant_decent.ipynb`**
   - Manually implements the Batch Gradient Descent algorithm.
   - Demonstrates how model weights are iteratively updated across the *entire* dataset to minimize Mean Squared Error (MSE).

5. **`stochastic_grideant_decent.ipynb`**
   - Implements Stochastic Gradient Descent (SGD) for computational efficiency.
   - Shows how updating weights using single/small batches helps the model converge faster while maintaining accuracy.

