# 🏠 House Price Prediction Model

A machine learning project that predicts California median house values using supervised learning techniques.

This project is part of my practical journey into Machine Learning, alongside the **Machine Learning Specialization by Andrew Ng**. The goal is to move beyond learning algorithms theoretically and apply them to a complete real-world machine learning problem.

## 🎯 Project Objective

The objective of this project is to build a model that can predict the median house value of a California housing district based on characteristics such as:

* Median income
* House age
* Average number of rooms
* Average number of bedrooms
* Population
* Average occupancy
* Latitude
* Longitude

The project begins with a **Linear Regression baseline** and progressively evaluates more advanced machine learning approaches.

## 📊 Dataset

The project uses the **California Housing dataset** provided through `scikit-learn`.

The dataset contains **20,640 observations** and 8 input features.

### Features

| Feature      | Description                 |
| ------------ | --------------------------- |
| `MedInc`     | Median income               |
| `HouseAge`   | Median house age            |
| `AveRooms`   | Average number of rooms     |
| `AveBedrms`  | Average number of bedrooms  |
| `Population` | Population of the district  |
| `AveOccup`   | Average number of occupants |
| `Latitude`   | Geographic latitude         |
| `Longitude`  | Geographic longitude        |

### Target

`MedHouseVal` — median house value, measured in units of $100,000.

## 🧠 Machine Learning Approach

The project follows a progressive model-development approach:

1. Explore and understand the dataset
2. Split the data into training and testing sets
3. Establish a Linear Regression baseline
4. Evaluate model performance
5. Experiment with Ridge Regression
6. Compare different regularization strengths
7. Experiment with nonlinear models
8. Compare model performance
9. Select the strongest approach
10. Eventually deploy the model as an interactive application

## 📈 Initial Results

### Linear Regression Baseline

| Metric  | Result |
| ------- | -----: |
| MAE     | 0.5332 |
| RMSE    | 0.7456 |
| Test R² | 0.5758 |

Because the target is measured in units of $100,000:

* MAE ≈ **$53,320**
* RMSE ≈ **$74,560**

The initial model explains approximately **57.6% of the variation** in the test data.

### Ridge Regression

Several regularization strengths were tested.

The best result from the initial experiment was obtained with:

**α = 100**

with a test R² of approximately **0.5805**.

This represents a modest improvement over the Linear Regression baseline.

## 📉 Model Evaluation

Model performance is evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between the predicted and actual values.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

### R² Score

Measures how much of the variation in the target variable is explained by the model.

Actual versus predicted values are also visualized to understand how closely predictions follow the ideal prediction relationship.

## 🛠️ Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* GitHub

## 📁 Project Structure

```text
House-Price-Prediction-Model/
│
├── House_Price_Prediction_ML.ipynb
└── README.md
```

## 🚀 Future Improvements

This project is intentionally being developed progressively.

Planned improvements include:

* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting
* Hyperparameter tuning
* Cross-validation
* Feature engineering
* Improved model evaluation
* Model comparison
* Interactive prediction interface
* Model deployment

## 📚 Learning Context

This project is being developed while studying the **Machine Learning Specialization by Andrew Ng**.

Rather than simply completing course exercises, the project is intended to reinforce machine learning concepts by applying them to a practical problem.

The development process follows:

**Learn → Experiment → Evaluate → Improve → Build**

## 👨‍💻 Author

**Eddward82**

This repository documents my practical journey from learning machine learning fundamentals to building and deploying real-world ML applications.
