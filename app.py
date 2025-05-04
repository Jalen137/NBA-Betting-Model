from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('betting_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        spread = float(request.form['spread'])
        moneyline = float(request.form['moneyline'])
        total = float(request.form['total'])

        features = np.array([[spread, moneyline, total]])
        prediction = model.predict(features)[0]

        result = "YES — they will cover the spread." if prediction == 1 else "NO — they will not cover the spread."
        return render_template('index.html', prediction=result)
    
    except ValueError:
        return render_template('index.html', prediction="Invalid input. Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)
