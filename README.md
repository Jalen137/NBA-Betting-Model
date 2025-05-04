# NBA Spread Prediction App

This project is a machine learning web application that predicts whether an NBA team will cover the point spread in a given game. It uses a Random Forest classifier trained on historical NBA betting data and is served through a Flask web interface.

## Project Overview

Users can input basic game-level betting information such as:

- Point Spread
- Moneyline
- Over/Under Total

Based on the trained model, the app predicts whether the team is likely to cover the spread.

The web app is built with Flask and styled using custom HTML/CSS. It is designed to be simple, fast, and extendable.

## Features

- Predict spread coverage using a trained Random Forest model
- User-friendly web interface built with Flask
- Custom CSS styling for a clean layout
- Fully functional on localhost
- Easily deployable to platforms like Render or Railway

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- HTML/CSS
- Jupyter Notebook (for model development)

## File Structure

\```
nba_predictor/
├── app.py                  # Flask backend
├── betting_model.pkl       # Trained model (not tracked in GitHub)
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   └── style.css           # Custom styles
├── notebooks/
│   └── model_dev.ipynb     # Jupyter notebook for model training
\```

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Install dependencies

\```bash
pip install flask pandas scikit-learn joblib
\```

### Run the app

\```bash
python app.py
\```

Then open your browser and go to:

http://127.0.0.1:5000/


### Using the App

1. Enter the spread, moneyline, and total points for an NBA game
2. Submit the form
3. The app returns a prediction on whether the team will cover the spread

## Model Information

The model is trained on historical NBA game data with columns including:

- Spread
- Moneyline
- Over/Under total
- Final scores (used to derive the target)

The model type is a `RandomForestClassifier`. You can retrain or tune it using the included Jupyter notebook.

## Note

The trained model file (`betting_model.pkl`) is excluded from this repository due to file size constraints.  
To use the app locally:

1. Open the Jupyter notebook in the `notebooks/` folder  
2. Run all cells to train the model  
3. Save the model with:

\```python
joblib.dump(model, 'betting_model.pkl')
\```

Then move the file to the main `nba_predictor/` directory.

## Future Improvements

- Add more features: home/away, opponent strength, rest days
- Improve model accuracy with XGBoost or LightGBM
- Add prediction confidence levels
- Deploy the app publicly via Render or Railway

## License

This project is for educational purposes and is not intended for actual betting use.
