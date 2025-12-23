import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


class MLpipeline:
    def __init__(
        self,
        model,
        data,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.model = model
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = None
        self.pipeline = None
        self.y_pred = None
        self.results = {}

        self._load_data(data)

    # ---------------- LOAD DATA ----------------
    def _load_data(self, data):
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a CSV path or a pandas DataFrame")

        self.X = df.drop(columns=[self.target_col])
        self.y = df[self.target_col]

    # ---------------- EDA ----------------
    def EDA(self):
        print("\nFirst rows:")
        print(self.X.head())

        print("\nInfo:")
        self.X.info()

        print("\nDescribe:")
        print(self.X.describe(include="all"))

        print("\nMissing values:")
        print(self.X.isnull().sum())

        num_df = self.X.select_dtypes(include=[np.number])
        if not num_df.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Matrix")
            plt.show()

        plt.figure()
        sns.histplot(self.y, kde=True)
        plt.title("Target Distribution")
        plt.show()

    # ---------------- FEATURE ENGINEERING ----------------
    def FeatureEngineering_and_DataCleaning(self):
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        categorical_features = self.X.select_dtypes(include=["object", "category"]).columns

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()) 
        ])

        # OneHotEncoder parameter name differs across sklearn versions
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe)
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    # ---------------- TRAIN & EVALUATE ----------------
    def ModelTraining_and_Evaluation(self, cv: int = 5):
        if self.preprocessor is None:
            self.FeatureEngineering_and_DataCleaning()

        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", self.model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        cv_scores = cross_val_score(
            self.pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error"
        )

        # convert negative MSE scores to positive MSE for reporting
        cv_mse_scores = -cv_scores

        self.pipeline.fit(X_train, y_train)
        self.y_pred = self.pipeline.predict(X_test)

        mse = mean_squared_error(y_test, self.y_pred)

        self.results = {
            "cv_mse_mean": float(np.mean(cv_mse_scores)),
            "cv_mse_std": float(np.std(cv_mse_scores)),
            "test_rmse": float(np.sqrt(mse)),
            "test_mae": float(mean_absolute_error(y_test, self.y_pred)),
            "test_r2": float(r2_score(y_test, self.y_pred))
        }

        print("\nResults:")
        for k, v in self.results.items():
            print(f"{k}: {v}")

        return self.results

    # ---------------- MODEL TUNING ----------------
    def Model_Tuning(self, param_grid: dict, cv: int = 5):
        # ensure pipeline exists
        if self.pipeline is None:
            if self.preprocessor is None:
                self.FeatureEngineering_and_DataCleaning()
            self.pipeline = Pipeline(steps=[("preprocessor", self.preprocessor), ("model", self.model)])

        if not hasattr(self, "X_train"):
            raise RuntimeError("Call ModelTraining_and_Evaluation before Model_Tuning to create train/test splits.")

        grid = GridSearchCV(
            self.pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        grid.fit(self.X_train, self.y_train)

        self.pipeline = grid.best_estimator_
        self.y_pred = self.pipeline.predict(self.X_test)

        mse = mean_squared_error(self.y_test, self.y_pred)

        self.results.update({
            "tuned_rmse": float(np.sqrt(mse)),
            "tuned_r2": float(r2_score(self.y_test, self.y_pred)),
            "best_params": grid.best_params_
        })

        print("\nTuned Results:")
        for k, v in self.results.items():
            print(f"{k}: {v}")

        return self.results

    # ---------------- PLOTS ----------------
    def plot_results(self):
        # convert to numpy arrays for safe numeric operations
        y_true = np.array(self.y_test)
        y_pred = np.array(self.y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError("Shape mismatch between y_test and y_pred for plotting.")

        residuals = y_true - y_pred

        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.6)
        x_min = float(np.min(y_true))
        x_max = float(np.max(y_true))
        x_line = np.array([x_min, x_max], dtype=float)
        plt.plot(x_line, x_line, "r--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()

        plt.figure()
        sns.histplot(residuals, kde=True)
        plt.title("Residuals Distribution")
        plt.show()


# ---------------- RUN ----------------
if __name__ == "__main__":
    data = pd.read_csv("duzenlenmis_cattle_milk_yield.csv")

    model = Ridge()

    ml = MLpipeline(
        model=model,
        data=data,
        target_col="Sut_Verimi_L_gun"
    )

    ml.EDA()
    ml.ModelTraining_and_Evaluation(cv=5)

    param_grid = {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    ml.Model_Tuning(param_grid=param_grid, cv=5)
    ml.plot_results()
