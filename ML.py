# =====================================
# IMPORTS
# =====================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)

# MODELS
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR


# =====================================
# MAIN CLASS
# =====================================
class EndToEndRegressionPipeline:

    def __init__(self, data: pd.DataFrame, target: str, random_state=42):
        self.data = data
        self.target = target
        self.random_state = random_state

        self.X = data.drop(columns=[target])
        self.y = data[target]

        self.results_df = None
        self.best_models = {}

        self.preprocessor = self._build_preprocessor()
        self.models = self._build_models()

    # =====================================
    # 1️⃣ EDA
    # =====================================
    def eda(self):
        print("\n===== DATA OVERVIEW =====")
        print(self.data.head())
        print("\n===== INFO =====")
        self.data.info()

        print("\n===== DESCRIBE =====")
        print(self.data.describe())

        print("\n===== MISSING VALUES =====")
        print(self.data.isnull().sum())

        # Target distribution
        plt.figure()
        sns.histplot(self.y, kde=True)
        plt.title("Target Distribution")
        plt.show()

        # Correlation (numeric only)
        num_df = self.data.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:
            plt.figure(figsize=(8,6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.show()

    # =====================================
    # 2️⃣ FEATURE ENGINEERING & PREPROCESS
    # =====================================
    def _build_preprocessor(self):
        num_cols = self.X.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = self.X.select_dtypes(include=["object", "category"]).columns

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        return ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ])

    # =====================================
    # 3️⃣ MODELS
    # =====================================
    def _build_models(self):
        return {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
            "DecisionTree": DecisionTreeRegressor(random_state=self.random_state),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=self.random_state),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=self.random_state),
            "GradientBoosting": GradientBoostingRegressor(),
            "HistGradientBoosting": HistGradientBoostingRegressor(),
            "SVR": SVR(kernel="rbf")
        }

    # =====================================
    # 4️⃣ MODEL EVALUATION
    # =====================================
    def evaluate_models(self, cv=5):
        scoring = {
            "MAE": "neg_mean_absolute_error",
            "MSE": "neg_mean_squared_error",
            "R2": "r2",
            "MedianAE": "neg_median_absolute_error"
        }

        records = []

        for name, model in self.models.items():
            pipe = Pipeline([
                ("prep", self.preprocessor),
                ("model", model)
            ])

            scores = cross_validate(pipe, self.X, self.y, cv=cv, scoring=scoring)

            mae = -scores["test_MAE"].mean()
            mse = -scores["test_MSE"].mean()
            rmse = np.sqrt(mse)
            r2 = scores["test_R2"].mean()
            medae = -scores["test_MedianAE"].mean()

            records.append({
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "MedianAE": medae
            })

        self.results_df = pd.DataFrame(records).sort_values("RMSE")
        return self.results_df

    # =====================================
    # 5️⃣ SELECT BEST MODELS
    # =====================================
    def select_best_models(self, top_n=3):
        best_names = self.results_df.head(top_n)["Model"].tolist()
        for name in best_names:
            self.best_models[name] = self.models[name]

    # =====================================
    # 6️⃣ SENIOR-LEVEL DIAGNOSTICS
    # =====================================
    def diagnostics(self, model_name):
        model = self.best_models[model_name]

        pipe = Pipeline([
            ("prep", self.preprocessor),
            ("model", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        residuals = y_test - preds

        # Actual vs Predicted
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, preds, alpha=0.35)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{model_name} | Actual vs Predicted")
        plt.show()

        # Residual Distribution
        plt.figure()
        sns.histplot(residuals, kde=True)
        plt.title(f"{model_name} | Residual Distribution")
        plt.show()

        # Residual vs Predicted
        plt.figure()
        plt.scatter(preds, residuals, alpha=0.35)
        plt.axhline(0, linestyle="--", color="red")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(f"{model_name} | Residuals vs Predicted")
        plt.show()

    # =====================================
    # 7️⃣ SUMMARY
    # =====================================
    def summary(self):
        print("\n===== MODEL COMPARISON TABLE =====\n")
        print(self.results_df)

"""

• RMSE düşük ama MAE yüksek → büyük hatalar var
• R² tek başına karar kriteri değildir
• MAE ≈ MedianAE → stabil model
• Tree/Boosting → performans
• Ridge → production güvenliği
"""


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    data = pd.read_csv("duzenlenmis_cattle_milk_yield.csv")

    pipeline = EndToEndRegressionPipeline(
        data=data,
        target="Sut_Verimi_L_gun"
    )

    pipeline.eda()
    results = pipeline.evaluate_models(cv=5)
    pipeline.select_best_models(top_n=3)
    pipeline.summary()

    for model_name in pipeline.best_models:
        pipeline.diagnostics(model_name)
