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
    "Boston Celtics": (61, 21),
    "Milwaukee Bucks": (48, 34),
    "Philadelphia 76ers": (24, 58),
    "Cleveland Cavaliers": (64, 18),
    "New York Knicks": (51, 31),
    "Miami Heat": (37, 45),
    "Atlanta Hawks": (40, 42),
    "Brooklyn Nets": (26, 56),
    "Chicago Bulls": (39, 43),
    "Toronto Raptors": (30, 52),
    "Indiana Pacers": (50, 32),
    "Charlotte Hornets": (19, 63),
    "Orlando Magic": (41, 41),
    "Washington Wizards": (18, 64),
    "Detroit Pistons": (44, 38),
    "Denver Nuggets": (50, 32),
    "Memphis Grizzlies": (48, 34),
    "Sacramento Kings": (40, 42),
    "Phoenix Suns": (36, 46),
    "Los Angeles Clippers": (50, 32),
    "Golden State Warriors": (48, 34),
    "Los Angeles Lakers": (50, 32),
    "Minnesota Timberwolves": (49, 33),
    "New Orleans Pelicans": (21, 61),
    "Oklahoma City Thunder": (68, 14),
    "Dallas Mavericks": (39, 43),
    "Portland Trail Blazers": (36, 46),
    "Utah Jazz": (17, 65),
    "San Antonio Spurs": (34, 48),
    "Houston Rockets": (52, 30)
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
