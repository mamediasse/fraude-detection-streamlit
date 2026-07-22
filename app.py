# -*- coding: utf-8 -*-
"""
app.py - Interface Optimisée (Design Moderne)
Application Streamlit — Détection de transactions bancaires suspectes / frauduleuses.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
 
# Configuration de base du répertoire
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fraud_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Bank_transaction_scenario1.csv')
 
# 1. Configuration de la page (Mode Large et Thème Professionnel)
st.set_page_config(
    page_title="RiskGuard | Détection de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Palette de couleurs professionnelles
COLOR_MAP = {"Normal": "#10B981", "Suspect": "#89745F", "Fraude": "#EF4444"}
 
# Chargement du modèle et des données avec mise en cache
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
 
# Vérification de la présence du modèle
try:
    artifact = load_model()
    model_ready = True
except FileNotFoundError:
    model_ready = False
 
# =========================================================
# BARRE LATÉRALE (SIDEBAR) - Informations & Statut du système
# =========================================================
with st.sidebar:
    st.image("https://icons8.com", width=80)
    st.title("RiskGuard ML")
    st.write("Système de surveillance des transactions en temps réel.")
    st.divider()
 
    # Indicateur de statut du modèle IA
    if model_ready:
        st.success("🟢 Modèle Random Forest : Actif")
    else:
        st.error("🔴 Modèle IA : Introuvable")
 
    st.divider()
    st.caption("Version 2.0 (Interface Optimisée) • © 2026")
 
# =========================================================
# CORPS PRINCIPAL DE L'APPLICATION
# =========================================================
st.title("🛡️ Plateforme Intégrée de Détection de Fraude")
st.write("Analyse prédictive des risques bancaires et exploration analytique des flux.")
 
# Navigation par Onglets (Plus moderne que les boutons radio)
tab1, tab2 = st.tabs(["🔍 Analyse en Temps Réel", "📊 Tableau de Bord Décisionnel"])
 
# ---------------------------------------------------------
# ONGLET 1 : Prédiction sur une nouvelle transaction
# ---------------------------------------------------------
with tab1:
    if not model_ready:
        st.error("⚠️ Impossible de charger l'interface d'analyse. Le fichier `model/fraud_model.pkl` est manquant.")
    else:
        df = load_data()
        localisations = sorted(df['Localisation'].unique())
        types_transac = sorted(df['Type de transaction'].unique())
        statuts = sorted(df['Status operation'].unique())
 
        st.subheader("📋 Formulaire d'évaluation du risque")
 
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                montant = st.number_input("Montant de l'opération (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
                type_transac = st.selectbox("Canal / Type de transaction", types_transac)
                status = st.selectbox("Statut technique de l'opération", statuts)
 
            with col2:
                localisation = st.selectbox("Zone géographique / Localisation", localisations)
                date = st.date_input("Date d'exécution")
                heure = st.slider("Heure de la transaction (Format 24h)", 0, 23, 12)
 
            st.write("")
            submit_button = st.button("Lancer l'audit de sécurité", type="primary", use_container_width=True)
 
        # Tout ce qui suit dépend du clic sur le bouton
        if submit_button:
            label, proba = predict_transaction(artifact, montant, type_transac, status, localisation, date, heure)
 
            st.write("")
            st.subheader("🎯 Diagnostic du Système")
 
            if label == "Fraude":
                st.error(f"🚨 **ALERTE CRITIQUE : TRANSACTION REJETÉE ({label.upper()})**\n\nCette opération présente toutes les caractéristiques d'une fraude cybernétique active. Blocage immédiat recommandé.")
            elif label == "Suspect":
                st.warning("⚠️ **VIGILANCE REQUISE : TRANSACTION SUSPECTE**\n\nComportement inhabituel détecté. Une double authentification ou une validation manuelle par un agent est nécessaire.")
            else:
                st.success("✅ **TRANSACTION AUTORISÉE (Normale)**\n\nAucun indicateur de risque anormal détecté. L'opération suit le protocole standard.")
 
            # Graphique des probabilités et export, intégrés dans une colonne dédiée
            col_graph, col_vide = st.columns([2, 1])
            with col_graph:
                proba_df = pd.DataFrame({
                    'Niveau de Risque': list(proba.keys()),
                    'Indice de Confiance': list(proba.values())
                }).sort_values('Indice de Confiance', ascending=False)
 
                fig = px.bar(
                    proba_df, x='Indice de Confiance', y='Niveau de Risque', orientation='h',
                    color='Niveau de Risque', color_discrete_map=COLOR_MAP, text_auto='.1%',
                    title="Décomposition du score de probabilité par le modèle"
                )
                fig.update_layout(xaxis_tickformat='.0%', showlegend=False, height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width='stretch')
 
                # =========================================================
                # FONCTIONNALITÉ D'EXPORTATION CSV
                # =========================================================
                st.write("")
                st.markdown("**💾 Exportation des résultats**")
 
                # Alignement sémantique string du label pour éviter l'affichage de listes
                clean_label = label[0] if isinstance(label, (list, np.ndarray)) else str(label)
 
                # Structuration sécurisée des lignes d'exportation
                export_df = pd.DataFrame([{
                    'Date': date.strftime('%Y-%m-%d'),
                    'Heure': f"{heure}:00",
                    'Montant_FCFA': montant,
                    'Type_Transaction': type_transac,
                    'Statut_Operation': status,
                    'Localisation': localisation,
                    'Statut_Predi_IA': clean_label,
                    'Probabilite_Normal': f"{proba.get('Normal', 0.0) * 100:.2f}%" if 'Normal' in proba else "0.00%",
                    'Probabilite_Suspect': f"{proba.get('Suspect', 0.0) * 100:.2f}%" if 'Suspect' in proba else "0.00%",
                    'Probabilite_Fraude': f"{proba.get('Fraude', 0.0) * 100:.2f}%" if 'Fraude' in proba else "0.00%"
                }])
 
                # Conversion binaire sécurisée (séparateur d'Excel en Europe/Afrique)
                csv_data = export_df.to_csv(index=False, sep=';').encode('utf-8')
 
                # Bouton de téléchargement synchronisé avec la mise en page
                st.download_button(
                    label="📥 Télécharger le rapport de cette prédiction (CSV)",
                    data=csv_data,
                    file_name=f"prediction_fraude_{date.strftime('%Y%m%d')}_{heure}h.csv",
                    mime="text/csv",
                    width='stretch'
                )
 
# ---------------------------------------------------------
# ONGLET 2 : Tableau de bord / exploration des données
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Indicateurs Clés de Performance (KPI)")
    df = load_data()
    target_col = 'Target'
 
    # Cartes de scores épurées
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Flux Global Analysé", value=f"{len(df):,}", delta="Volume Standard")
    with col2:
        taux_fraude = (df[target_col] == 'Fraude').mean() * 100
        st.metric(label="Taux d'Incidents (Fraude)", value=f"{taux_fraude:.2f}%", delta="-0.4% vs mois dernier", delta_color="inverse")
    with col3:
        taux_suspect = (df[target_col] == 'Suspect').mean() * 100
        st.metric(label="Alertes en Attente (Suspect)", value=f"{taux_suspect:.2f}%", delta="Niveau stable")
 
    st.write("")
 
    # Organisation en grille pour les graphiques
    col_chart1, col_chart2 = st.columns(2)
 
    with col_chart1:
        with st.container(border=True):
            st.markdown("**Structure globale du portefeuille de risques**")
            fig1 = px.pie(
                df, names=target_col, color=target_col, color_discrete_map=COLOR_MAP,
                hole=0.5, height=300
            )
            fig1.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig1, width='stretch')
 
    with col_chart2:
        with st.container(border=True):
            st.markdown("**Distribution de la sévérité des montants (log)**")
            df_plot = df.copy()
            df_plot['Log-Montant (FCFA)'] = np.log1p(df_plot['Montant'])
            fig4 = px.box(
                df_plot, x=target_col, y='Log-Montant (FCFA)', color=target_col,
                color_discrete_map=COLOR_MAP, height=300
            )
            fig4.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig4, width='stretch')
 
    with st.container(border=True):
        st.markdown("**Vulnérabilité par typologie et statut technique des opérations**")
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            fig2 = px.histogram(
                df, x='Type de transaction', color=target_col, barnorm='percent',
                color_discrete_map=COLOR_MAP, title="Impact par Canal", height=280
            )
            st.plotly_chart(fig2, width='stretch')
        with col_hist2:
            fig3 = px.histogram(
                df, x='Status operation', color=target_col, barnorm='percent',
                color_discrete_map=COLOR_MAP, title="Impact par Statut Réseau", height=280
            )
            st.plotly_chart(fig3, width='stretch')
 
    # Section d'audit des données brutes cachable
    with st.expander("🗂️ Accéder au registre complet des transactions auditées"):
        st.dataframe(df.head(100), width='stretch')
 