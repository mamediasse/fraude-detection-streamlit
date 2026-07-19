# -*- coding: utf-8 -*-
"""
train_model.py
Entraîne un modèle Random Forest de détection de transactions
suspectes / frauduleuses et le sérialise dans model/fraud_model.pkl

Usage :
    python model/train_model.py
"""
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Bank_transaction_scenario1.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fraud_model.pkl')

FEATURES_NUM = ['Montant', 'heure', 'jour_semaine', 'weekend']
FEATURES_CAT = ['Type de transaction', 'Status operation', 'Localisation']


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')

    # Nettoyage des doublons d'encodage sur Localisation
    df['Localisation'] = df['Localisation'].replace({
        'Saint-Louis': 'Saint Louis',
        'RunÃ©': 'Rufisque'
    })

    # Feature engineering sur la date
    df['Date'] = pd.to_datetime(df['Date'])
    df['heure'] = df['Date'].dt.hour
    df['jour_semaine'] = df['Date'].dt.dayofweek
    df['weekend'] = df['jour_semaine'].isin([5, 6]).astype(int)

    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), FEATURES_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), FEATURES_CAT)
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    return Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])


def main():
    print("Chargement des données...")
    df = load_and_clean(DATA_PATH)

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df['Target']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    print("Entraînement du modèle Random Forest...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1-score (macro) : {f1_macro:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Sérialisation du pipeline complet + de l'encodeur de labels
    artifact = {
        'pipeline': pipe,
        'label_encoder': le,
        'features_num': FEATURES_NUM,
        'features_cat': FEATURES_CAT,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Modèle sauvegardé -> {MODEL_PATH}")


if __name__ == '__main__':
    main()
