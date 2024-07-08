from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the Gaussian Mixture Model
with open('gmm_model.pkl', 'rb') as file:
    gmm_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cust_id = request.form['cust_id']
    balance = float(request.form['balance'])
    purchases = float(request.form['purchases'])

    input_data = np.array([[balance, purchases]])
    cluster = gmm_model.predict(input_data)

    return render_template('index.html', result=f'Customer ID: {cust_id} belongs to Cluster: {cluster[0]}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
