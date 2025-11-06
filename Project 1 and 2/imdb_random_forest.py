#!/usr/bin/env python3
"""Train a Random Forest on the IMDb merged CSV.

Assumption: we create a binary target `high_rating` where rating >= 7.

This script:
- loads `imdb_merged_results.csv` from the same directory
- does lightweight preprocessing (parse year/review_count, encode categories, TF-IDF on storyline)
- trains RandomForestClassifier
- prints evaluation metrics and saves the trained model to `imdb_rf_model.joblib`
"""
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    return df


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    # Target: high_rating if rating >= 7.0
    df = df.copy()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[~df['rating'].isna()].reset_index(drop=True)
    df['high_rating'] = (df['rating'] >= 7.0).astype(int)

    # review_count sometimes contains non-numeric artifacts; coerce to numeric
    df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
    # take absolute values if negatives appear (these look like data artifacts)
    df['review_count'] = df['review_count'].abs()

    # Extract year number if possible (some values like '2024–2025')
    df['year_parsed'] = df['year'].astype(str).str.extract(r"(\d{4})").astype(float)

    # Fill storyline NaNs with empty string
    df['storyline'] = df['storyline'].fillna('')

    return df


def build_pipeline():
    text_col = 'storyline'
    cat_cols = ['genre1', 'genre2', 'genre3', 'country_of_origin', 'platform1', 'platform2']
    num_cols = ['review_count', 'year_parsed']

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=2000), text_col),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
        ],
        remainder='drop',
        sparse_threshold=0.3,
    )

    clf = Pipeline([
        ('pre', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    return clf


def main():
    base = Path(__file__).resolve().parent
    csv_path = base / 'imdb_merged_results.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    print('Loading data from', csv_path)
    df = load_data(str(csv_path))
    df = preprocess_df(df)

    feature_cols = ['storyline', 'genre1', 'genre2', 'genre3', 'country_of_origin', 'platform1', 'platform2', 'review_count', 'year_parsed']
    X = df[feature_cols]
    y = df['high_rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_pipeline()
    print('Training Random Forest...')
    model.fit(X_train, y_train)

    print('Evaluating...')
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print('\nAccuracy:', accuracy_score(y_test, y_pred))
    if y_proba is not None:
        try:
            print('ROC AUC:', roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    print('\nClassification report:')
    print(classification_report(y_test, y_pred))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, y_pred))

    out_model = base / 'imdb_rf_model.joblib'
    joblib.dump(model, out_model)
    print(f'Written trained model to: {out_model}')


if __name__ == '__main__':
    main()
