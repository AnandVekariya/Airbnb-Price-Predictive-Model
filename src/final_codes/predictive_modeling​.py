import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# ============================
# File Paths
# ============================
MODELING_PATH = r"\database\Modeling_Final_Clean_Project_Data.xlsx"
RAW_FILE_PATH = r"\database\Cleaning_Final_Clean_Project_Data.xlsx"
OUTPUT_PATH = r"\database\predicted_price_Final_Project_Data.xlsx"


# ============================
# Helper Functions
# ============================


def load_data(modeling_path, raw_file_path):
    """Load data from Excel files."""
    df_modeling = pd.read_excel(modeling_path)
    df_cleaning = pd.read_excel(raw_file_path)
    print(
        f"✅ Data Loaded: {df_modeling.shape[0]} rows, {df_modeling.shape[1]} columns"
    )
    return df_modeling, df_cleaning


def filter_numeric_columns(df, target_column="Price", correlation_threshold=0.05):
    """Filter numeric columns based on correlation with target column."""
    numeric_data = df.select_dtypes(include=["int64", "float64"])
    correlation_matrix = numeric_data.corr()
    if target_column not in correlation_matrix.columns:
        raise KeyError(f"❌ Target column '{target_column}' not found in dataset.")

    filtered_corr = (
        correlation_matrix[target_column]
        .loc[correlation_matrix[target_column].abs() > correlation_threshold]
        .sort_values(ascending=False)
    )
    print(
        f"✅ Filtered {len(filtered_corr)} features correlated with '{target_column}' (> {correlation_threshold})"
    )
    return numeric_data[filtered_corr.index], filtered_corr


def split_data(df_filtered, target_column="Price", train_ratio=0.75, random_seed=42):
    """Split dataset into train and validation sets."""
    np.random.seed(random_seed)
    shuffled_idx = np.random.permutation(len(df_filtered))
    split_point = int(train_ratio * len(df_filtered))

    X = df_filtered.drop(columns=[target_column])
    y = df_filtered[target_column]

    X_train, X_val = (
        X.iloc[shuffled_idx[:split_point]],
        X.iloc[shuffled_idx[split_point:]],
    )
    y_train, y_val = (
        y.iloc[shuffled_idx[:split_point]],
        y.iloc[shuffled_idx[split_point:]],
    )
    print(f"✅ Split: {len(X_train)} training samples, {len(X_val)} validation samples")
    return X_train, y_train, X_val, y_val


def scale_features(X_train, X_val):
    """Standardize features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled


def mean_absolute_percentage_error(y_true, y_pred):
    """Compute Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return np.round(mape, 2)


def calculate_metrics(y_true, y_pred, n_samples, n_features):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return r2, adjusted_r2, mse, rmse, mape


def print_metrics(metrics, data_type):
    """Nicely print model performance metrics."""
    r2, adj_r2, mse, rmse, mape = metrics
    print(f"\n📊 {data_type} Metrics:")
    print(f"R²: {r2:.3f}")
    print(f"Adjusted R²: {adj_r2:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.2f}%")


def plot_results(y_train, y_pred_train, y_val, y_pred_val, model_name):
    """Visualize prediction performance."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_pred_train, color="blue", label="Train", alpha=0.5)
    plt.scatter(y_val, y_pred_val, color="red", label="Validation", alpha=0.5)
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.show()


# ============================
# Modeling Functions
# ============================


def linear_regression(X_train, y_train, X_val, y_val):
    model = LinearRegression().fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    metrics_train = calculate_metrics(
        y_train, y_pred_train, len(y_train), X_train.shape[1]
    )
    metrics_val = calculate_metrics(y_val, y_pred_val, len(y_val), X_train.shape[1])

    print_metrics(metrics_train, "Training")
    print_metrics(metrics_val, "Validation")
    plot_results(y_train, y_pred_train, y_val, y_pred_val, "Linear Regression")
    return metrics_train, metrics_val


def lasso_regression(X_train, y_train, X_val, y_val):
    X_train_scaled, X_val_scaled = scale_features(X_train, X_val)
    param_grid = {"alpha": np.logspace(-3, 3, 14)}
    lasso = Lasso(max_iter=10000)
    model = RandomizedSearchCV(
        lasso, param_distributions=param_grid, cv=3, n_iter=10, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    print(f"🔍 Best Alpha: {model.best_params_}")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)

    metrics_train = calculate_metrics(
        y_train, y_pred_train, len(y_train), X_train.shape[1]
    )
    metrics_val = calculate_metrics(y_val, y_pred_val, len(y_val), X_train.shape[1])

    print_metrics(metrics_train, "Training")
    print_metrics(metrics_val, "Validation")
    plot_results(y_train, y_pred_train, y_val, y_pred_val, "Lasso Regression")
    return metrics_train, metrics_val


def ridge_regression(X_train, y_train, X_val, y_val):
    param_grid = {"alpha": np.logspace(-3, 3, 14)}
    model = RandomizedSearchCV(
        Ridge(), param_distributions=param_grid, cv=3, random_state=42
    )
    model.fit(X_train, y_train)

    print(f"🔍 Best Alpha: {model.best_params_}")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    metrics_train = calculate_metrics(
        y_train, y_pred_train, len(y_train), X_train.shape[1]
    )
    metrics_val = calculate_metrics(y_val, y_pred_val, len(y_val), X_train.shape[1])

    print_metrics(metrics_train, "Training")
    print_metrics(metrics_val, "Validation")
    plot_results(y_train, y_pred_train, y_val, y_pred_val, "Ridge Regression")
    return metrics_train, metrics_val


def random_forest_regression(X_train, y_train, X_val, y_val):
    param_dist = {
        "n_estimators": range(200, 801, 200),
        "max_depth": [4, 6, 8, 10],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    model = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        cv=3,
        n_iter=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print(f"🌲 Best Parameters: {model.best_params_}")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    metrics_train = calculate_metrics(
        y_train, y_pred_train, len(y_train), X_train.shape[1]
    )
    metrics_val = calculate_metrics(y_val, y_pred_val, len(y_val), X_train.shape[1])

    print_metrics(metrics_train, "Training")
    print_metrics(metrics_val, "Validation")
    plot_results(y_train, y_pred_train, y_val, y_pred_val, "Random Forest Regression")

    return metrics_train, metrics_val, y_pred_train, y_pred_val


# ============================
# Main Execution Pipeline
# ============================


def predictive_modeling_pipeline():
    df_modeling, df_raw = load_data(MODELING_PATH, RAW_FILE_PATH)

    filtered_data, filtered_corr = filter_numeric_columns(
        df_modeling, target_column="Price", correlation_threshold=0.05
    )
    X_train, y_train, X_val, y_val = split_data(
        filtered_data, "Price", train_ratio=0.75
    )

    print("\n=== Linear Regression ===")
    linear_regression(X_train, y_train, X_val, y_val)

    print("\n=== Lasso Regression ===")
    lasso_regression(X_train, y_train, X_val, y_val)

    print("\n=== Ridge Regression ===")
    ridge_regression(X_train, y_train, X_val, y_val)

    print("\n=== Random Forest Regression ===")
    _, _, y_train_pred, y_val_pred = random_forest_regression(
        X_train, y_train, X_val, y_val
    )

    # Save predictions
    predicted_prices = pd.concat(
        [pd.Series(y_train_pred), pd.Series(y_val_pred)], ignore_index=True
    )
    actual_prices = pd.concat([y_train, y_val], ignore_index=True)

    result_df = pd.DataFrame(
        {
            "Actual Price": actual_prices.round(2),
            "Predicted Price": predicted_prices.round(2),
        }
    )

    df_raw = pd.concat([df_raw.reset_index(drop=True), result_df], axis=1)
    df_raw.to_excel(OUTPUT_PATH, index=False)
    print(f"\n💾 Final dataset with predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    predictive_modeling_pipeline()
