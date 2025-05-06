from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from flask_caching import Cache
import csv
from datetime import datetime

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Load models
nba_model = joblib.load("xgboost_betting_model.pkl")
nfl_model = joblib.load("xgboost_betting_model_nfl.pkl")

# Team lists
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

nfl_teams = sorted([
    "New England Patriots", "Kansas City Chiefs", "Dallas Cowboys", "Green Bay Packers",
    "Buffalo Bills", "Philadelphia Eagles", "San Francisco 49ers", "Baltimore Ravens",
    "Cincinnati Bengals", "Miami Dolphins", "Pittsburgh Steelers", "Cleveland Browns",
    "New York Giants", "New York Jets", "Los Angeles Rams", "Los Angeles Chargers",
    "Minnesota Vikings", "Tampa Bay Buccaneers", "Detroit Lions", "Chicago Bears",
    "Tennessee Titans", "Indianapolis Colts", "Seattle Seahawks", "Arizona Cardinals",
    "Carolina Panthers", "Atlanta Falcons", "New Orleans Saints", "Houston Texans",
    "Jacksonville Jaguars", "Las Vegas Raiders", "Denver Broncos", "Washington Commanders"
])

# Dummy records
nba_records = {team: (50, 32) for team in nba_teams}
nfl_records = {team: (9, 8) for team in nfl_teams}

# Log file
if not os.path.exists("prediction_log.csv"):
    with open("prediction_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "League", "Team", "Opponent", "Spread", "Moneyline", "Total", "Prediction", "Confidence"])

def get_team_record(team_name):
    return nba_records.get(team_name) or nfl_records.get(team_name) or (None, None)

@app.route('/')
def home():
    return render_template("index.html", nba_teams=nba_teams, nfl_teams=nfl_teams)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        sport = request.form['sport']
        team = request.form['team']
        opponent = request.form['opponent']
        spread = float(request.form['spread'])
        moneyline = float(request.form['moneyline'])
        total = float(request.form['total'])

        team_wins, team_losses = get_team_record(team)
        opponent_wins, opponent_losses = get_team_record(opponent)

        if None in [team_wins, team_losses, opponent_wins, opponent_losses]:
            return render_template("index.html", prediction="Could not fetch team records. Try again.", nba_teams=nba_teams, nfl_teams=nfl_teams)

        features = np.array([[spread, moneyline, total, team_wins, team_losses, opponent_wins, opponent_losses]])
        model = nba_model if sport == "NBA" else nfl_model

        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]
        confidence = round(proba * 100, 2) if prediction == 1 else round((1 - proba) * 100, 2)

        result = f"{team} will {'cover' if prediction == 1 else 'not cover'} the spread. ({confidence}% confidence)"

        with open("prediction_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                sport,
                team,
                opponent,
                spread,
                moneyline,
                total,
                prediction,
                round(proba, 4)
            ])

        return render_template("index.html", prediction=result, confidence=confidence, nba_teams=nba_teams, nfl_teams=nfl_teams)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", nba_teams=nba_teams, nfl_teams=nfl_teams)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
