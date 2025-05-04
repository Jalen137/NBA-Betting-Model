from flask import Flask, render_template, request
import joblib
import numpy as np
import os

from nba_api.stats.endpoints import leaguestandings
import time

def get_team_record(team_name):
    try:
        standings = leaguestandings.LeagueStandings(season="2023-24").get_data_frames()[0]
        team_row = standings[standings['TeamName'] == team_name]

        if not team_row.empty:
            wins = int(team_row.iloc[0]['Win'])
            losses = int(team_row.iloc[0]['Loss'])
            return wins, losses
        else:
            return None, None
    except Exception as e:
        print("Error fetching record:", e)
        return None, None

app = Flask(__name__)

# Load the trained model
model = joblib.load('xgboost_betting_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    try:
        team = request.form['team']
        spread = float(request.form['spread'])
        moneyline = float(request.form['moneyline'])
        total = float(request.form['total'])
        opponent = request.form['opponent']
        team_wins, team_losses = get_team_record(team)
        opponent_wins, opponent_losses = get_team_record(opponent)

        if None in [team_wins, team_losses, opponent_wins, opponent_losses]:
            return render_template('index.html', prediction="Could not fetch team records. Try again.")


        features = np.array([[spread, moneyline, total, team_wins, team_losses, opponent_wins, opponent_losses]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        if prediction == 1:
            result = f"YES — {team} will cover the spread. ({round(proba * 100, 2)}% confidence)"
        else:
            result = f"NO — {team} will not cover the spread. ({round((1 - proba) * 100, 2)}% confidence)"

        return render_template('index.html', prediction=result)

    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please enter valid numbers.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

if not os.path.exists("xgboost_betting_model.pkl"):
    raise FileNotFoundError("Model file not found. Please train the model and save it as xgboost_betting_model.pkl.")
