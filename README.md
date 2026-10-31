# EV-Range-Prediction-Using-Machine-Learning
Machine learning-based Electric Vehicle Range Prediction using Linear Regression, Decision Tree, Random Forest, KNN, and MLP algorithms to predict EV driving range and compare model performance.

##  About the Project

EV Range Intelligence is a Machine Learning project designed to predict the electric driving range of Electric Vehicles (EVs). It uses vehicle-related information to train regression models and estimate the electric range.

The project implements and compares five Machine Learning algorithms to identify the best-performing prediction model.

##  Project Objectives

- Predict the electric range of electric vehicles.
- Analyze electric vehicle population data.
- Apply different Machine Learning regression algorithms.
- Compare model performance using evaluation metrics.
- Identify the most accurate model among the implemented algorithms.

##  Dataset

**Dataset Name:** Electric Vehicle Population Data

**Source:** Data.gov

**Dataset Link:** https://catalog.data.gov/dataset/electric-vehicle-population-data

### Features Used

- Model Year
- Make
- Model
- Electric Vehicle Type
- CAFV Eligibility

**Target Variable:** Electric Range

##  Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

##  Machine Learning Algorithms

Five regression algorithms are implemented:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors (KNN) Regressor
5. Multi-Layer Perceptron (MLP) Regressor

##  Project Workflow

1. Dataset Collection
2. Data Cleaning and Preprocessing
3. Handling Missing Values
4. Label Encoding
5. Training and Testing Split
6. Model Training
7. Prediction
8. Model Evaluation
9. Data Visualization

##  Model Performance

The following R² scores were obtained:

| Machine Learning Model | R² Score |
|---|---:|
| Random Forest | 0.9968 |
| Decision Tree | 0.9967 |
| KNN | 0.9955 |
| Linear Regression | 0.7680 |
| MLP Neural Network | 0.7623 |

**Best Performing Model:** Random Forest Regressor

##  Evaluation Metrics

The models are evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

##  Visualizations

The project includes:

- Actual vs Predicted Range Graphs
- KNN Residual Plot
- MLP Training Loss Curve
- R² Score Comparison
- Electric Range Distribution
- Model Year vs Electric Range
- Average Electric Range by Vehicle Type

##  Future Scope

- Improve prediction accuracy using advanced algorithms.
- Perform hyperparameter tuning.
- Include additional vehicle features.
- Develop a web application for EV range prediction.
- Explore real-time electric vehicle range estimation.

##  Conclusion

EV Range Intelligence demonstrates the application of Machine Learning in predicting electric vehicle range. Five regression algorithms are implemented and compared. Based on the reported R² scores, Random Forest Regressor achieved the best performance with an R² score of 0.9968.
