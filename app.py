from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from flask_caching import Cache
import csv
from datetime import datetime

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Ensure model exists
if not os.path.exists("xgboost_betting_model.pkl"):
    raise FileNotFoundError("Model file not found. Please train and save it as xgboost_betting_model.pkl.")

model = joblib.load("xgboost_betting_model.pkl")

# Create CSV log if needed
if not os.path.exists("prediction_log.csv"):
    with open("prediction_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Team", "Opponent", "Spread", "Moneyline", "Total", "Prediction", "Confidence"])

nba_teams = sorted([
    "Boston Celtics", "Milwaukee Bucks", "Philadelphia 76ers", "Cleveland Cavaliers",
    "New York Knicks", "Miami Heat", "Atlanta Hawks", "Brooklyn Nets",
    "Chicago Bulls", "Toronto Raptors", "Indiana Pacers", "Charlotte Hornets",
    "Orlando Magic", "Washington Wizards", "Detroit Pistons", "Denver Nuggets",
    "Memphis Grizzlies", "Sacramento Kings", "Phoenix Suns", "Los Angeles Clippers",
    "Golden State Warriors", "Los Angeles Lakers", "Minnesota Timberwolves",
    "New Orleans Pelicans", "Oklahoma City Thunder", "Dallas Mavericks",
    "Portland Trail Blazers", "Utah Jazz", "San Antonio Spurs", "Houston Rockets"
])

team_records = {
    "Boston Celtics": (64, 18),
    "Milwaukee Bucks": (49, 33),
    "Philadelphia 76ers": (47, 35),
    "Cleveland Cavaliers": (48, 34),
    "New York Knicks": (50, 32),
    "Miami Heat": (46, 36),
    "Atlanta Hawks": (36, 46),
    "Brooklyn Nets": (32, 50),
    "Chicago Bulls": (39, 43),
    "Toronto Raptors": (25, 57),
    "Indiana Pacers": (47, 35),
    "Charlotte Hornets": (21, 61),
    "Orlando Magic": (47, 35),
    "Washington Wizards": (15, 67),
    "Detroit Pistons": (14, 68),
    "Denver Nuggets": (57, 25),
    "Memphis Grizzlies": (27, 55),
    "Sacramento Kings": (46, 36),
    "Phoenix Suns": (49, 33),
    "Los Angeles Clippers": (51, 31),
    "Golden State Warriors": (46, 36),
    "Los Angeles Lakers": (47, 35),
    "Minnesota Timberwolves": (56, 26),
    "New Orleans Pelicans": (49, 33),
    "Oklahoma City Thunder": (57, 25),
    "Dallas Mavericks": (50, 32),
    "Portland Trail Blazers": (21, 61),
    "Utah Jazz": (31, 51),
    "San Antonio Spurs": (22, 60),
    "Houston Rockets": (41, 41)
}

def get_team_record(team_name):
    return team_records.get(team_name, (None, None))

@app.route('/')
def home():
    return render_template("index.html", nba_teams=nba_teams)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        team = request.form['team']
        opponent = request.form['opponent']
        spread = float(request.form['spread'])
        moneyline = float(request.form['moneyline'])
        total = float(request.form['total'])

        team_wins, team_losses = get_team_record(team)
        opponent_wins, opponent_losses = get_team_record(opponent)

        if None in [team_wins, team_losses, opponent_wins, opponent_losses]:
            return render_template("index.html", prediction="Could not fetch team records. Try again.", nba_teams=nba_teams)

        features = np.array([[spread, moneyline, total, team_wins, team_losses, opponent_wins, opponent_losses]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]
        confidence = round(proba * 100, 2) if prediction == 1 else round((1 - proba) * 100, 2)

        result = f"{team} will {'cover' if prediction == 1 else 'not cover'} the spread. ({confidence}% confidence)"

        with open("prediction_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                team,
                opponent,
                spread,
                moneyline,
                total,
                prediction,
                round(proba, 4)
            ])

        return render_template("index.html", prediction=result, nba_teams=nba_teams)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", nba_teams=nba_teams)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
