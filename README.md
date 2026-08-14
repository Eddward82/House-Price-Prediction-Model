# 🏠 California House Price Prediction

A supervised machine learning project that predicts median house values in California using demographic, housing, and geographic features.

This project was developed as part of my practical journey through machine learning, alongside the Machine Learning Specialization by Andrew Ng. Rather than using a single algorithm, I progressively trained, evaluated, and compared multiple models to understand how different machine learning approaches perform on the same problem.

---

## 🎯 Project Objective

The objective is to predict the median house value of a California housing district based on characteristics such as:

- Median income
- House age
- Average rooms
- Average bedrooms
- Population
- Average occupancy
- Latitude
- Longitude

The project follows a complete supervised machine learning workflow:

**Data → Exploration → Preprocessing → Training → Evaluation → Model Comparison → Hyperparameter Tuning → Model Selection**

---

## 📊 Dataset

The project uses the **California Housing dataset** provided through `scikit-learn`.

The dataset contains:

- **20,640 observations**
- **8 input features**
- **1 target variable**

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Population of the district |
| `AveOccup` | Average number of occupants |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

### Target

`MedHouseVal`

The target represents the median house value in units of **$100,000**.

For example:

`2.50 = $250,000`

---

# 🧠 Machine Learning Models

Five different supervised learning approaches were investigated.

### 1. Linear Regression

Used as the initial baseline model.

### 2. Ridge Regression

Used to investigate the effect of L2 regularization and different regularization strengths.

### 3. Decision Tree Regression

Introduced nonlinear relationships and provided an opportunity to study model complexity and overfitting.

### 4. Random Forest Regression

Used ensemble learning to combine predictions from multiple decision trees.

### 5. Gradient Boosting Regression

Used sequential decision trees where each new tree attempts to improve upon the errors of previous trees.

---

# 📈 Model Evaluation

The models were evaluated using three primary metrics.

### Mean Absolute Error (MAE)

Measures the average absolute difference between predicted and actual values.

Lower is better.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

Lower is better.

### R² Score

Measures the proportion of variation in the target variable explained by the model.

Higher is better.

---

# 🧪 Model Comparison

The following results were obtained during experimentation:

| Model | MAE | RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 0.5332 | 0.7456 | 0.5758 |
| Ridge Regression | ~0.5337 | ~0.7414 | ~0.5805 |
| Decision Tree | 0.4332 | 0.6446 | 0.6829 |
| Random Forest | ~0.3270 | ~0.5042 | ~0.8060 |
| **Gradient Boosting** | **0.3033** | **0.4625** | **0.8368** |

Lower MAE and RMSE are better, while higher R² is better.

---

# 🏆 Final Model

The best-performing model in this project was:

## Gradient Boosting Regressor

The final configuration selected was:

```
python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.2,
    max_depth=5,
    random_state=42
)
```

## 🏆 Final Performance

- **R²:** 0.8368
- **MAE:** 0.3033
- **RMSE:** 0.4625

The model explains approximately **83.7% of the variation** in the test data.

The MAE of 0.3033 corresponds to approximately **$30,330 average absolute prediction error**, because the target variable is expressed in $100,000 units.

---

## 🔍 Feature Importance

Feature importance was examined using the Random Forest model to understand which variables were most influential in its predictive decisions.

| Feature | Importance |
|---|---:|
| `MedInc` | 0.5267 |
| `AveOccup` | 0.1381 |
| `Latitude` | 0.0885 |
| `Longitude` | 0.0884 |
| `HouseAge` | 0.0540 |
| `AveRooms` | 0.0445 |
| `Population` | 0.0303 |
| `AveBedrms` | 0.0294 |

---

## 🔑 Key Findings

### 1. Nonlinear models performed better

The Decision Tree, Random Forest, and Gradient Boosting models substantially outperformed the linear models.

This suggests that the relationship between the available housing features and house values is not adequately captured by a simple linear relationship.

### 2. Ensemble methods were particularly effective

Random Forest significantly improved upon a single Decision Tree.

Gradient Boosting performed even better in the final experiments.

### 3. Median income was highly influential

`MedInc` was by far the most influential feature according to the Random Forest feature-importance measure.

### 4. Location matters

`Latitude` and `Longitude` together represented a substantial portion of the Random Forest's feature importance.

### 5. Model complexity needs to be controlled

Increasing model complexity can improve training performance while reducing performance on unseen data.

The experiments provided practical evidence of overfitting.

### 6. More complexity does not always mean better performance

The experiments showed diminishing returns from increasing tree depth and the number of estimators.

The goal is not to build the most complicated model possible, but to build a model that **generalizes well**.

> **Note:** Feature importance indicates how useful a feature was to the model's predictive decisions. It should not be interpreted as proof that a feature directly causes changes in house prices.

---

## 🌲 Random Forest and Cross-Validation

The Random Forest model achieved approximately:

- **R²:** 0.806
- **MAE:** 0.327
- **RMSE:** 0.504

Five-fold cross-validation was also performed on the Random Forest configuration.

The validation R² scores were approximately:

```text
0.8079
0.7949
0.8102
0.8062
0.8061
```

**Mean cross-validation R²:** 0.8051

**Standard deviation:** 0.0053

The relatively small standard deviation indicated consistent performance across the validation folds.

> **Note:** Cross-validation was performed on the Random Forest during experimentation. The final selected model was Gradient Boosting.

---

## 🔬 Hyperparameter Tuning

The models were not simply trained using default settings. Several hyperparameters were systematically investigated.

### Ridge Regression

Different values of `alpha` were tested to investigate the effect of regularization.

### Decision Tree

Different values of `max_depth` were tested to investigate the relationship between model complexity and generalization.

The best Decision Tree configuration used:

```text
max_depth = 10
```

### Random Forest

The following parameters were investigated:

- `max_depth`
- `n_estimators`

Increasing the number of trees improved performance initially, but the improvement became smaller as more trees were added.

### Gradient Boosting

The following parameters were investigated:

- `n_estimators`
- `learning_rate`
- `max_depth`

The selected configuration was:

```text
n_estimators = 300
learning_rate = 0.2
max_depth = 5
```

---

## 📉 Overfitting Analysis

One of the most important lessons from this project was understanding overfitting through experimentation.

During the Decision Tree experiments, increasing `max_depth` caused training performance to increase substantially while test performance eventually decreased.

A similar pattern was observed during Gradient Boosting experiments.

For example, deeper Gradient Boosting configurations eventually produced training R² values approaching **1.0**, while test performance deteriorated.

This demonstrated that:

> **A model that performs extremely well on training data is not necessarily a good model.**

The objective is to find a model that generalizes well to unseen data.

---

## 🛠️ Technologies

The project was built using:

- **Python**
- **Google Colab**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Joblib**
- **GitHub**

---

## 📁 Project Structure

```text
House-Price-Prediction-Model/
│
├── House_Price_Prediction_ML.ipynb
├── california_house_price_model.pkl
├── README.md
└── requirements.txt
```

### Files

| File | Description |
|---|---|
| `House_Price_Prediction_ML.ipynb` | Complete ML experimentation notebook |
| `california_house_price_model.pkl` | Trained final Gradient Boosting model |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Eddward82/House-Price-Prediction-Model.git
```

### 2. Navigate into the project

```bash
cd House-Price-Prediction-Model
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

The notebook can be opened using:

- Google Colab
- Jupyter Notebook
- JupyterLab
- VS Code

### 5. Run the notebook

Execute the notebook cells sequentially to reproduce the analysis and model experiments.

---

## 💾 Using the Trained Model

The trained model is saved as:

```text
california_house_price_model.pkl
```

It can be loaded using Joblib:

```python
import joblib

model = joblib.load("california_house_price_model.pkl")
```

The loaded model can then be used to generate predictions from appropriately formatted input features.

---

## 🚀 Future Improvements

The core machine learning modeling phase of this project is complete.

Potential future improvements include:

- Building a user-friendly house price prediction interface
- Creating a prediction API
- Deploying the model as a web application
- Adding input validation
- Improving feature engineering
- Investigating additional machine learning algorithms
- Using more advanced model interpretation techniques
- Adding automated model testing
- Monitoring model performance after deployment
- Evaluating the model on new external data

---

## 📚 Learning Context

This project was developed while studying machine learning concepts from the **Machine Learning Specialization by Andrew Ng**.

The project was intentionally developed progressively to reinforce concepts including:

- Supervised learning
- Regression
- Linear regression
- Regularization
- Decision trees
- Overfitting
- Ensemble learning
- Random forests
- Gradient boosting
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Feature importance

Rather than simply following course exercises, the project was used to apply the concepts to a complete machine learning workflow.

### Development Philosophy

**Learn → Build → Experiment → Evaluate → Improve → Compare**

---

## 👨‍💻 Author

**Eddward82**

This project represents a practical step in my journey from learning machine learning fundamentals to building real-world machine learning applications.

More machine learning projects will be added as I continue developing my skills in:

- Supervised learning
- Unsupervised learning
- Deep learning
- Model deployment
- Applied AI

---

## ⭐ Project Status

**Modeling phase: Complete ✅**

**Final model: Gradient Boosting Regressor**

**Best Test R²: 0.8368**

**Next stage: Model deployment and application development 🚀**







