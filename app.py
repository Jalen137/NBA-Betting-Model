from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('xgboost_betting_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        spread = float(request.form['spread'])
        moneyline = float(request.form['moneyline'])
        total = float(request.form['total'])
        team_wins = float(request.form['team_wins'])
        team_losses = float(request.form['team_losses'])
        opponent_wins = float(request.form['opponent_wins'])
        opponent_losses = float(request.form['opponent_losses'])

        features = np.array([[spread, moneyline, total, team_wins, team_losses, opponent_wins, opponent_losses]])

        prediction = model.predict(features)[0]

        result = "YES — they will cover the spread." if prediction == 1 else "NO — they will not cover the spread."
        return render_template('index.html', prediction=result)
    
    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please enter valid numbers.")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)


if not os.path.exists("betting_model.pkl"):
    raise FileNotFoundError("Model file not found. Please train the model and save it as betting_model.pkl.")
