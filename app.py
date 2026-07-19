# -*- coding: utf-8 -*-
"""
app.py
Application Streamlit — Détection de transactions bancaires
suspectes / frauduleuses.

Usage :
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fraud_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Bank_transaction_scenario1.csv')

st.set_page_config(
    page_title="Détection de fraude bancaire",
    page_icon="🏦",
    layout="wide"
)

COLOR_MAP = {"Normal": "#2ca02c", "Suspect": "#ff9800", "Fraude": "#d62728"}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=';')
    df['Localisation'] = df['Localisation'].replace({
        'Saint-Louis': 'Saint Louis',
        'RunÃ©': 'Rufisque'
    })
    return df


def predict_transaction(artifact, montant, type_transac, status, localisation, date, heure):
    pipe = artifact['pipeline']
    le = artifact['label_encoder']

    jour_semaine = date.weekday()
    weekend = int(jour_semaine in [5, 6])

    X_new = pd.DataFrame([{
        'Montant': montant,
        'heure': heure,
        'jour_semaine': jour_semaine,
        'weekend': weekend,
        'Type de transaction': type_transac,
        'Status operation': status,
        'Localisation': localisation,
    }])

    pred = pipe.predict(X_new)[0]
    proba = pipe.predict_proba(X_new)[0]
    label = le.inverse_transform([pred])[0]
    proba_dict = dict(zip(le.classes_, proba))
    return label, proba_dict


# =========================================================
# Sidebar - navigation
# =========================================================
st.sidebar.title("🏦 Fraud Detection")
page = st.sidebar.radio("Navigation", ["Prédiction", "Tableau de bord"])

try:
    artifact = load_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# =========================================================
# Page 1 : Prédiction sur une nouvelle transaction
# =========================================================
if page == "Prédiction":
    st.title("🔍 Prédiction sur une transaction")
    st.write(
        "Renseignez les informations d'une transaction pour évaluer "
        "si elle est **normale**, **suspecte** ou **frauduleuse**."
    )

    if not model_ready:
        st.error(
            "Modèle introuvable. Lancez d'abord `python model/train_model.py` "
            "pour générer `model/fraud_model.pkl`."
        )
    else:
        df = load_data()
        localisations = sorted(df['Localisation'].unique())
        types_transac = sorted(df['Type de transaction'].unique())
        statuts = sorted(df['Status operation'].unique())

        col1, col2 = st.columns(2)
        with col1:
            montant = st.number_input(
                "Montant (FCFA)", min_value=0.0, value=50000.0, step=1000.0
            )
            col2.metric("Taux de fraude", f"{(df['Target'] == 'Fraude').mean() * 100:.1f}%")
            col3.metric("Taux suspect", f"{(df['Target'] == 'Suspect').mean() * 100:.1f}%")

        with col2:
            localisation = st.selectbox("Localisation", localisations)
            date = st.date_input("Date de la transaction")
            heure = st.slider("Heure de la transaction", 0, 23, 12)

        if st.button("Analyser la transaction", type="primary"):
            label, proba = predict_transaction(
                artifact, montant, type_transac, status, localisation, date, heure
            )

            color = COLOR_MAP.get(label, "#666666")
            st.markdown(
                f"### Résultat : <span style='color:{color}'>**{label}**</span>",
                unsafe_allow_html=True
            )

            proba_df = pd.DataFrame({
                'Classe': list(proba.keys()),
                'Probabilité': list(proba.values())
            }).sort_values('Probabilité', ascending=False)

            fig = px.bar(
                proba_df, x='Classe', y='Probabilité', color='Classe',
                color_discrete_map=COLOR_MAP, text_auto='.1%',
                title="Probabilités par classe"
            )
            fig.update_layout(yaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            if label == "Fraude":
                st.error("⚠️ Cette transaction présente un fort risque de fraude.")
            elif label == "Suspect":
                st.warning("⚠️ Cette transaction est suspecte, une vérification est recommandée.")
            else:
                st.success("✅ Cette transaction semble normale.")

# =========================================================
# Page 2 : Tableau de bord / exploration des données
# =========================================================
else:
    st.title("📊 Tableau de bord des transactions")

    df = load_data()

    # Remplacer 'Target' par le nom exact de votre colonne cible : 'Class'
    target_col = 'Class' 

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions totales", f"{len(df):,}")
    col2.metric("Taux de fraude", f"{(df[target_col] == 'Fraude').mean() * 100:.1f}%")
    col3.metric("Taux suspect", f"{(df[target_col] == 'Suspect').mean() * 100:.1f}%")

    st.subheader("Répartition des classes")
    fig1 = px.pie(
        df, names=target_col, color=target_col, color_discrete_map=COLOR_MAP, hole=0.4
    )
    st.plotly_chart(fig1, use_container_width=True)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Type de transaction vs statut")
        fig2 = px.histogram(
            df, x='Type de transaction', color=target_col, barnorm='percent',
            color_discrete_map=COLOR_MAP
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col5:
        st.subheader("Status opération vs statut")
        fig3 = px.histogram(
            df, x='Status operation', color=target_col, barnorm='percent',
            color_discrete_map=COLOR_MAP
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Distribution du montant (log) par classe")
    df_plot = df.copy()
    df_plot['log_montant'] = np.log1p(df_plot['Montant'])
    fig4 = px.box(
        df_plot, x=target_col, y='log_montant', color=target_col,
        color_discrete_map=COLOR_MAP
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Aperçu des données")
    st.dataframe(df.head(50), use_container_width=True)
