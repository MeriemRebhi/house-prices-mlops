# src/train.py

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib


# === Chemins de base ===
DATA_PATH = "data/train.csv"          # on utilise un chemin RELATIF, pas le C:\... du notebook
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Charge les données d'entraînement depuis un fichier CSV.
    """
    print(f"📥 Chargement des données depuis {path} ...")
    df = pd.read_csv(path)
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    1. Garde uniquement les colonnes numériques.
    2. Supprime les lignes avec des valeurs manquantes.
    3. Sépare X (features) et y (cible).
    4. Crée des bins pour stratifier la cible.
    5. Fait le train/test split avec stratification.
    """
    # Étape 3 du notebook : colonnes numériques + dropna
    df_num = df.select_dtypes(include=["int64", "float64"]).dropna()
    print(f"🔢 Colonnes numériques retenues : {df_num.shape[1]}")
    print(f"📊 Données après nettoyage : {df_num.shape[0]} lignes")

    # Étape 4 : X / y
    if "SalePrice" not in df_num.columns:
        raise ValueError("La colonne 'SalePrice' n'existe pas dans les données numériques.")

    X = df_num.drop("SalePrice", axis=1)
    y = df_num["SalePrice"]

    # Création de bins pour la stratification (comme dans le notebook)
    y_bins = pd.cut(y, bins=10, labels=False)

    # Étape 5 : train/test split avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y_bins,
    )

    print(f"📦 Taille du train : {X_train.shape}")
    print(f"📦 Taille du test  : {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Entraîne un modèle de régression linéaire.
    """
    print("🚂 Entraînement du modèle LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Modèle entraîné.")
    return model


def evaluate_model(
    model: LinearRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Calcule et affiche les métriques RMSE et R² pour train et test.
    (reprend la logique du notebook)
    """
    print("📈 Évaluation du modèle...")

    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # R²
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("=== Évaluation du modèle ===")
    print(f"RMSE (train) : {train_rmse:.2f}")
    print(f"RMSE (test)  : {test_rmse:.2f}")
    print(f"R² (train)   : {train_r2:.3f}")
    print(f"R² (test)    : {test_r2:.3f}")


def save_model(model: LinearRegression, path: str = MODEL_PATH):
    """
    Sauvegarde le modèle entraîné dans un fichier .pkl (joblib), comme dans le notebook.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Modèle sauvegardé dans : {path}")


if __name__ == "__main__":
    # Pipeline complet = équivalent du notebook, mais en script

    # 1. Chargement des données
    df = load_data()

    # 2. Préparation / split des données
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3. Entraînement du modèle
    model = train_model(X_train, y_train)

    # 4. Évaluation
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # 5. Sauvegarde du modèle
    save_model(model)
