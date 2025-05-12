import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Simulated dataset structure for model training
sample_data = pd.DataFrame({
    'player_1_rank': [1, 2, 5, 10],
    'player_2_rank': [3, 4, 8, 12],
    'player_1_age': [35, 28, 30, 32],
    'player_2_age': [25, 26, 29, 31],
    'surface': ['Hard', 'Clay', 'Grass', 'Hard'],
    'tourney_level': ['G', 'M', 'A', 'D'],
    'round': ['QF', 'R16', 'SF', 'F'],
    'best_of': [5, 3, 5, 3],
    'label': [1, 0, 1, 0]
})

X = sample_data.drop(columns=['label'])
y = sample_data['label']

# Preprocessing
cat_features = ['surface', 'tourney_level', 'round']
num_features = ['player_1_rank', 'player_2_rank', 'player_1_age', 'player_2_age', 'best_of']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Models
rf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(max_depth=10, n_estimators=200, random_state=0, class_weight='balanced'))
])
rf_pipeline.fit(X, y)

logreg_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l2', C=1))
])
logreg_pipeline.fit(X, y)

xgb_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=7, n_estimators=300, learning_rate=0.1))
])
xgb_pipeline.fit(X, y)

MODELS = {
    "XGBoost": xgb_pipeline,
    "Random Forest": rf_pipeline,
    "Logistic Regression": logreg_pipeline,
}

# Streamlit UI
st.set_page_config(page_title="Prédiction Tennis ATP", layout="centered")
st.title("🎾 Prédiction de match ATP")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.text_input("Nom du joueur 1")
        rank1 = st.number_input("Classement joueur 1", min_value=1, max_value=3000, value=1)
        age1 = st.number_input("Âge joueur 1", min_value=16.0, max_value=50.0, value=30.0)
    with col2:
        player2 = st.text_input("Nom du joueur 2")
        rank2 = st.number_input("Classement joueur 2", min_value=1, max_value=3000, value=2)
        age2 = st.number_input("Âge joueur 2", min_value=16.0, max_value=50.0, value=25.0)

    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
    level = st.selectbox("Niveau du tournoi", ["G", "M", "A", "D", "O"])
    round_ = st.selectbox("Tour du match", ["R128", "R64", "R32", "R16", "QF", "SF", "F"])
    best_of = st.radio("Format du match", [3, 5])
    model_choice = st.selectbox("Modèle à utiliser", ["XGBoost", "Random Forest", "Logistic Regression"])

    submitted = st.form_submit_button("Prédire")

if submitted:
    if player1 == player2:
        st.error("Les deux joueurs doivent être différents.")
    else:
        input_df = pd.DataFrame([{
            'player_1_rank': rank1,
            'player_2_rank': rank2,
            'player_1_age': age1,
            'player_2_age': age2,
            'surface': surface,
            'tourney_level': level,
            'round': round_,
            'best_of': best_of
        }])

        st.markdown("---")
        st.subheader("Résultat de la prédiction")

        selected_model = MODELS[model_choice]
        prob = selected_model.predict_proba(input_df)[0]
        predicted_label = selected_model.predict(input_df)[0]
        predicted_winner = player1 if predicted_label == 1 else player2

        st.markdown(f"### 🧠 {model_choice}")
        st.write(f"**Vainqueur prédit :** {predicted_winner}")
        st.write(f"**{player1} :** {round(prob[1]*100, 2)}% de chances de gagner")
        st.write(f"**{player2} :** {round(prob[0]*100, 2)}% de chances de gagner")
