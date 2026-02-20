from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':

        step = int(request.form['step'])
        transaction_type = label_encoder.transform([request.form['type']])[0]
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        balance_diff_org = oldbalanceOrg - newbalanceOrig
        balance_diff_dest = newbalanceDest - oldbalanceDest

        features = np.array([[step, transaction_type, amount,
                              oldbalanceOrg, newbalanceOrig,
                              oldbalanceDest, newbalanceDest,
                              balance_diff_org, balance_diff_dest]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "Fraudulent Transaction 🚨" if prediction == 1 else "Legitimate Transaction ✅"

        return render_template('submit.html', 
                               prediction=result, 
                               probability=round(probability * 100, 2)
                              )

if __name__ == "__main__":
    app.run(debug=True)