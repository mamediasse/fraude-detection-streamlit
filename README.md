# Détection de fraude bancaire — Streamlit

Application de détection de transactions bancaires **normales**, **suspectes** ou **frauduleuses**, basée sur un modèle de Machine Learning (Random Forest) et une interface Streamlit.

## Structure du projet

```
fraude-detection-streamlit/
│
├── data/
│   └── transactions.csv          # Jeu de données
│
├── model/
│   ├── train_model.py            # Script d'entraînement
│   └── fraud_model.pkl           # Modèle sérialisé (généré)
│
├── app.py                        # Application Streamlit
├── requirements.txt              # Dépendances
├── .gitignore
└── README.md
```

## Installation

```bash
# 1. Cloner le projet / se placer dans le dossier
cd fraude-detection-streamlit

# 2. Créer un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Entraîner le modèle

Le modèle doit être entraîné avant de lancer l'application (le fichier `fraud_model.pkl` n'est pas fourni si vous partez d'un dépôt vierge) :

```bash
python model/train_model.py
```

Cela génère `model/fraud_model.pkl`, qui contient :
- le pipeline complet (prétraitement + modèle Random Forest)
- l'encodeur des classes cibles
- la liste des variables numériques/catégorielles utilisées

## Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre par défaut sur `http://localhost:8501`.

## Fonctionnalités

### 🔍 Page Prédiction
Formulaire permettant de saisir les caractéristiques d'une transaction (montant, type, statut, localisation, date, heure) et d'obtenir :
- la classe prédite (Normal / Suspect / Fraude)
- les probabilités associées à chaque classe

### 📊 Page Tableau de bord
Visualisation exploratoire du jeu de données : répartition des classes, relation entre type de transaction/statut et fraude, distribution des montants.

## Modèle

- **Algorithme** : Random Forest (`n_estimators=300`, `class_weight='balanced'`)
- **Variables utilisées** : Montant, heure, jour de la semaine, weekend, type de transaction, statut de l'opération, localisation
- **Variables exclues** : identifiants (ID Client, Numéro de compte, Identifiant opération) — non prédictifs et risque de fuite de données
- **Performance (jeu de test)** : Accuracy ≈ 0.88, F1-score macro ≈ 0.74

## Notes

- Les identifiants clients ne sont volontairement pas utilisés comme variables prédictives.
- Le jeu de données présente un fort déséquilibre de classes (≈76% Normal, 20% Suspect, 4% Fraude) ; le modèle utilise une pondération de classes (`class_weight='balanced'`) pour compenser ce déséquilibre.
- Pour ré-entraîner le modèle après modification des données, relancer `python model/train_model.py`.
