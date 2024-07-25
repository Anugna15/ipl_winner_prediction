import numpy as np
import pandas as pd
import pickle
import json
from flask import Flask, render_template, url_for, jsonify, request
import os

os.chdir('/home/Dharsha/codegnan')

# Create a Flask object
app = Flask(_name_)

# Loading the pickle files (model)
classifier = pickle.load(open('winner_prediction_catboost_classifier.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('a.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = []

    # Retrieve values from dropdowns
    dropdown_values = [request.form.get('dropdown1'), request.form.get('dropdown2')]
    data.extend(dropdown_values)

    # Retrieve other values from the form
    other_values = [value for name, value in request.form.items() if name not in ['dropdown1', 'dropdown2']]

    # Convert numeric values to float, keep non-numeric values as strings
    for value in other_values:
        try:
            data.append(float(value))
        except ValueError:
            data.append(value)

    final_input = np.array(data).reshape(1, -1)
    output = classifier.predict(final_input)[0]

    if output == 0:
        prediction_text = "The team might not be winning"
    elif output == 1:
        prediction_text = "The team might be winning"
    else:
        prediction_text = "Invalid prediction"

    return render_template('a.html', prediction_text=prediction_text)