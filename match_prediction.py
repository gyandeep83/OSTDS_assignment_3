"""
This project uses a Random Forest Classifier to predict football match outcomes (Home Win, Draw, Away Win)
based on historical performance data. It focuses on Premier League clubs such as Chelsea, Man United,
Tottenham, and others, analyzing features like goals scored and team names.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df['MatchOutcome'] = df.apply(lambda row: 1 if row['FTHG'] > row['FTAG'] 
                                  else (-1 if row['FTHG'] < row['FTAG'] else 0), axis=1)
    return df

def encode_teams(df):
    le = LabelEncoder()
    df['HomeTeam'] = le.fit_transform(df['HomeTeam'])
    df['AwayTeam'] = le.transform(df['AwayTeam'])
    return df, le

def train_model(df):
    X = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    y = df['MatchOutcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                  display_labels=['Away Win (-1)', 'Draw (0)', 'Home Win (1)'])
    disp.plot(cmap='coolwarm')
    plt.title("Confusion Matrix")
    plt.show()

    return model

def predict_match(model, home_team, away_team, df, le):
    home_team, away_team = home_team.title(), away_team.title()
    if home_team not in le.classes_ or away_team not in le.classes_:
        print("Error: One or both team names are not in the dataset!")
        return None  

    home_encoded = le.transform([home_team])[0]
    away_encoded = le.transform([away_team])[0]
    home_stats = df[df['HomeTeam'] == home_encoded][['FTHG', 'FTAG']].mean()
    away_stats = df[df['AwayTeam'] == away_encoded][['FTHG', 'FTAG']].mean()

    if home_stats.isnull().any() or away_stats.isnull().any():
        print(f"Warning: No sufficient data for {home_team} or {away_team}.")
        return None

    input_features = np.array([[home_encoded, away_encoded, home_stats['FTHG'], away_stats['FTAG']]])

    if np.isnan(input_features).any():
        print("Error: Encountered NaN values in input features!")
        return None

    prediction = model.predict(input_features)[0]
    label = {1: "Home Win 🏠✅", 0: "Draw 🔄", -1: "Away Win 🚀✅"}

    print("\nPrediction Guide: 1 = Home Win, 0 = Draw, -1 = Away Win")
    print(f"Predicted Outcome: {prediction} → {label[prediction]}")
    return label[prediction]

# === RUN ===
file_path = '/Users/gyandeep/OSTDS_assign_3/Datasets/final_dataset.csv'
df = load_data(file_path)
df, le = encode_teams(df)
model = train_model(df)

home_team = input("Enter Home Team: ")
away_team = input("Enter Away Team: ")
prediction = predict_match(model, home_team, away_team, df, le)
