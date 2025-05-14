
import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="🎾 ATP Match Predictor", layout="centered")
st.markdown("## 🎾 Prédiction de Match ATP - Modèles sauvegardés")
st.markdown("Remplissez les informations ci-dessous pour prédire l’issue d’un match.")

# Sidebar - choix du modèle
model_choice = st.sidebar.selectbox("Choisir le modèle :", ["XGBoost", "Random Forest", "Logistic Regression"])

# Formulaire utilisateur
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.text_input("Nom du joueur 1", "Joueur 1")
        rank1 = st.number_input("Classement joueur 1", min_value=1, max_value=3000, value=20)
        age1 = st.number_input("Âge joueur 1", min_value=16.0, max_value=50.0, value=28.0)
    with col2:
        player2 = st.text_input("Nom du joueur 2", "Joueur 2")
        rank2 = st.number_input("Classement joueur 2", min_value=1, max_value=3000, value=30)
        age2 = st.number_input("Âge joueur 2", min_value=16.0, max_value=50.0, value=26.0)

    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    tourney_level = st.selectbox("Niveau du tournoi", ["G", "M", "A", "D"])
    round_ = st.selectbox("Tour du tournoi", ["R128", "R64", "R32", "R16", "QF", "SF", "F"])
    best_of = st.radio("Format du match", [3, 5])

    submitted = st.form_submit_button("Prédire")

# Chargement du modèle sauvegardé
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# Prédiction
if submitted:
    input_df = pd.DataFrame([{
        'player_1_rank': rank1,
        'player_2_rank': rank2,
        'player_1_age': age1,
        'player_2_age': age2,
        'surface': surface,
        'tourney_level': tourney_level,
        'round': round_,
        'best_of': best_of,
        'rank_diff': rank1 - rank2,
        'age_diff': age1 - age2,
        'rank_sum': rank1 + rank2,
        'is_top10_1': int(rank1 <= 10),
        'is_top10_2': int(rank2 <= 10),
        'is_grand_slam': int(tourney_level == 'G')
    }])

    try:
        model = joblib.load(model_files[model_choice])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        winner = player1 if pred == 1 else player2
        st.success(f"✅ Vainqueur prédit : **{winner}**")
        st.markdown(f"**Probabilités :** {player1} = {proba[1]*100:.2f}% | {player2} = {proba[0]*100:.2f}%")
    except FileNotFoundError:
        st.error("❌ Modèle non trouvé. Assurez-vous que les fichiers .pkl sont dans le même dossier que ce script.")
