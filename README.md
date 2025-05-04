# NBA Spread Predictor

A Flask web app that uses machine learning to predict whether an NBA team will cover the spread based on betting lines and recent performance. Built with XGBoost, deployed on Render, and styled with a responsive, dark-mode-enabled UI.

## Live Demo

[Click here to try it live](https://your-app-name.onrender.com)  
Replace with your actual Render URL

---

## Features

- Predicts if a team will cover the spread
- Inputs: spread, moneyline, total points, team records
- Team dropdowns with auto-filled win/loss stats
- Displays model confidence score (percentage)
- Mobile-friendly layout with dark mode toggle
- Loading animation during prediction

---

## Model Information

- Algorithm: XGBoost Classifier
- Accuracy: approximately 77.5% on validation set
- Features used:
  - Spread
  - Moneyline
  - Over/Under total
  - Team wins and losses
  - Opponent wins and losses

---

## How It Works

1. User selects two teams and enters the betting lines
2. The app auto-fills win/loss records from a JSON object
3. The model predicts if the selected team will cover the spread
4. The result and confidence level are displayed

---

## Tech Stack

- Python
- Flask
- XGBoost
- scikit-learn
- HTML/CSS
- JavaScript
- Hosted on Render

---

## Folder Structure

nba_predictor/
├── app.py
├── xgboost_betting_model.pkl
├── team_records.json
├── requirements.txt
├── Procfile
├── templates/
│ └── index.html
├── static/
│ └── style.css


---

## Future Improvements

- Pull live records from NBA API
- Add rolling average stats (last 3 games)
- Save predictions and compare to actual outcomes
- Create login and user prediction history system

---

## Author

Jalen Broxie  
Data Science Student at Washington & Jefferson College  
[LinkedIn](https://www.linkedin.com/) • [Portfolio](#) • [Email](mailto:your@email.com)

---

## To Run Locally

```bash
git clone https://github.com/yourusername/nba-spread-predictor.git
cd nba-spread-predictor
pip install -r requirements.txt
python app.py
