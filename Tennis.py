import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(page_title="🎾 Prédiction ATP", layout="centered")
st.title("🎾 ATP Match Predictor")
st.markdown("Sélectionne deux joueurs pour prédire le gagnant d’un match ATP avec différents modèles ML.")

# 📥 Chargement des joueurs
@st.cache_data
def load_players():
    df = pd.read_csv("players_summary.csv")  
    return df

players_df = load_players()
player_names = sorted(players_df["name"].unique().tolist())

# 🧠 Choix du modèle
model_choice = st.sidebar.selectbox("🔍 Choisis ton modèle :", [
    "XGBoost", "Random Forest", "Logistic Regression"
])

# 📋 Formulaire
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        player1_name = st.selectbox("Joueur 1", player_names, index=player_names.index("Jannik Sinner") if "Jannik Sinner" in player_names else 0)
    with col2:
        player2_name = st.selectbox("Joueur 2", player_names, index=player_names.index("Carlos Alcaraz") if "Carlos Alcaraz" in player_names else 1)

    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
    tourney_level = st.selectbox("Niveau tournoi", ["G", "M", "A", "D"])
    round_ = st.selectbox("Tour", ["R128", "R64", "R32", "R16", "QF", "SF", "F"])
    best_of = st.radio("Format", [3, 5])

    submitted = st.form_submit_button("🎯 Prédire")

# 📦 Fichiers modèles
model_files = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
}

# 🧠 Prédiction
if submitted:
    if player1_name == player2_name:
        st.error("❌ Les deux joueurs doivent être différents.")
    else:
        try:
            p1 = players_df[players_df["name"] == player1_name].iloc[0]
            p2 = players_df[players_df["name"] == player2_name].iloc[0]

            # Données du match dans le bon format
            input_dict = {
                'player_1_rank': p1['rank'],
                'player_2_rank': p2['rank'],
                'player_1_age': p1['age'],
                'player_2_age': p2['age'],
                'surface': surface,
                'tourney_level': tourney_level,
                'round': round_,
                'best_of': best_of,
                'rank_diff': p1['rank'] - p2['rank'],
                'age_diff': p1['age'] - p2['age'],
                'rank_sum': p1['rank'] + p2['rank'],
                'is_top10_1': int(p1['rank'] <= 10),
                'is_top10_2': int(p2['rank'] <= 10),
                'is_grand_slam': int(tourney_level == 'G'),
                'age_mean': (p1['age'] + p2['age']) / 2
            }

            input_df = pd.DataFrame([input_dict])

            # Chargement du bon modèle
            model_path = model_files[model_choice]
            model = joblib.load(model_path)

            # Prédiction
            pred = model.predict(input_df)[0]
            probas = model.predict_proba(input_df)[0]

            predicted_winner = player1_name if pred == 1 else player2_name
            st.success(f"🏆 Gagnant prédit : **{predicted_winner}**")

            st.markdown(f"""
                **Probabilités :**  
                - {player1_name} : {probas[1]*100:.2f}%  
                - {player2_name} : {probas[0]*100:.2f}%
            """)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
