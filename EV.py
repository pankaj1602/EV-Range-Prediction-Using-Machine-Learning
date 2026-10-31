import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Load Dataset
df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/CA2 pre/export.csv")
print("\nDataset Shape:")
print(df.shape)
df = df.drop_duplicates()
features = [
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type",
    "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
]
target = "Electric Range"
df = df[features + [target]].copy()


# Clean Data and Handle Missing Values
df["Model Year"] = pd.to_numeric(df["Model Year"], errors="coerce")
df[target] = pd.to_numeric(df[target], errors="coerce")
df.loc[df["Model Year"] <= 0, "Model Year"] = np.nan
df.loc[df[target] <= 0, target] = np.nan
df = df.dropna(subset=[target])
df["Model Year"] = df["Model Year"].fillna(df["Model Year"].median())

for col in features:
    if col != "Model Year":
        df[col] = df[col].fillna("Unknown").astype(str)

for col in features:
    if col != "Model Year":
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

df = df.sample(n=min(20000, len(df)), random_state=42)
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


# Decision Tree Regressor Model
dt = DecisionTreeRegressor(max_depth=15, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)


# Random Forest Regressor Model
rf = RandomForestRegressor(n_estimators=100,max_depth=15,random_state=42,n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# KNN Regressor Model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)


# MLP Neural Network Model
mlp = MLPRegressor(hidden_layer_sizes=(50, 25),max_iter=300,random_state=42)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)


# Compare All Models
models_results = {
    "Linear Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred,
    "KNN": knn_pred,
    "MLP Neural Network": mlp_pred
}
results = []

for name, pred in models_results.items():
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)

    results.append({
        "Model": name,
        "R2 Score": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE").reset_index(drop=True)
print("\nModel Comparison:")
print(results_df.to_string(index=False))


# Best Model
best_model = results_df.iloc[0]
print("\nBest Model:", best_model["Model"])
print("R2 Score:", best_model["R2 Score"])
print("RMSE:", best_model["RMSE"])
print("MAE:", best_model["MAE"])


#G1
plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_pred, alpha=0.5, color="#1f77b4", label="Predictions")
x_line = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(x_line, x_line, "r--", label="Ideal Prediction")
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Electric Range")
plt.ylabel("Predicted Electric Range")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#G2
n = 100
plt.figure(figsize=(10, 5))
plt.plot(y_test.iloc[:n].to_numpy(),label="Actual Values",marker="o", color="gray")
plt.plot(dt_pred[:n],label="Predicted Values",marker="x", color="#ff7f0e")
plt.title("Decision Tree: Actual vs Predicted")
plt.xlabel("Test Sample")
plt.ylabel("Electric Range")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#G3
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],"r--")
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Electric Range")
plt.ylabel("Predicted Electric Range")
plt.grid(True)
plt.tight_layout()
plt.show()


#G4
residuals = y_test.to_numpy() - knn_pred
plt.figure(figsize=(8, 5))
plt.scatter(knn_pred, residuals, alpha=0.5, color="#9467bd")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("KNN: Residual Plot")
plt.xlabel("Predicted Electric Range")
plt.ylabel("Residual Error")
plt.grid(True)
plt.tight_layout()
plt.show()


#G5
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, color="#d62728", linewidth=2)
plt.title("MLP Neural Network: Loss Curve")
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


#G6
plt.figure(figsize=(9, 5))
bars = plt.bar(
    results_df["Model"],
    results_df["R2 Score"],
    color=[{"Linear Regression": "#00A6A6", "Decision Tree": "#E83E8C", "Random Forest": "#F2B705", "KNN": "#8B4513", "MLP Neural Network": "#2E4A9E"}[m] for m in results_df["Model"]]
)
plt.bar_label(bars, fmt="%.4f")
plt.title("R2 Score Comparison of All Models")
plt.xlabel("Regression Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# Obj 1: Electric Range Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Electric Range"], bins=20, color="skyblue", edgecolor="black")
plt.title("Objective 1: Distribution of Electric Range")
plt.xlabel("Electric Range (km)")
plt.ylabel("Number of Vehicles")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# Obj 2: Model Year vs Electric Range
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Model Year"],
    df["Electric Range"],
    alpha=0.5,
    color="blue"
)
plt.title("Objective 2: Model Year vs Electric Range")
plt.xlabel("Model Year")
plt.ylabel("Electric Range")
plt.grid(True)
plt.tight_layout()
plt.show()


# Obj 3: Average Electric Range by Vehicle Type
range_by_type = df.groupby("Electric Vehicle Type")["Electric Range"].mean()
plt.figure(figsize=(8, 5))
bars = plt.bar(
    range_by_type.index,
    range_by_type.values,
    color="orange"
)
plt.bar_label(bars, fmt="%.2f")
plt.title("Objective 3: Average Electric Range by Vehicle Type")
plt.xlabel("Electric Vehicle Type")
plt.ylabel("Average Electric Range")
plt.xticks(
    range_by_type.index,
    ["Battery Electric Vehicle (BEV)", "Plug-in Hybrid Electric Vehicle (PHEV)"]
)
plt.tight_layout()
plt.show()

