from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams
from flask_caching import Cache
import csv
from datetime import datetime

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Ensure model exists
if not os.path.exists("xgboost_betting_model.pkl"):
    raise FileNotFoundError("Model file not found. Please train and save it as xgboost_betting_model.pkl.")

# Load model
model = joblib.load("xgboost_betting_model.pkl")

# Create CSV log if it doesn't exist
if not os.path.exists("prediction_log.csv"):
    with open("prediction_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Team", "Opponent", "Spread", "Moneyline", "Total", "Prediction", "Confidence"])

# Load all NBA teams
nba_teams = sorted([team["full_name"] for team in teams.get_teams()])

@cache.memoize(timeout=3600)  # Cache for 1 hour
def get_team_record(team_name):
    from nba_api.stats.library.http import NBAStatsHTTP

    # Set custom headers
    NBAStatsHTTP.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/"
    })

    standings = leaguestandings.LeagueStandings(season="2023-24").get_data_frames()[0]
    team_row = standings[standings['TeamName'] == team_name]

    if not team_row.empty:
        wins = int(team_row.iloc[0]['Win'])
        losses = int(team_row.iloc[0]['Loss'])
        return wins, losses
    else:
        return None, None


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

        # Save prediction to CSV
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
