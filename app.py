from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.preprocessing import StandardScaler
from src.logger import logging

from src.pipeline.predict_model import CustomData, PredictModel

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
             model=request.form.get('model'),
             duration=request.form.get('duration')
        )

        predict_model = PredictModel()
        results = predict_model.evaluate_model(data.model, data.duration)

        logging.info(f'Logging the results :{results}')
        return render_template('home.html', results = results)
    
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('predict.html')
    else:
        data = CustomData(
             model='LSTM',
             duration=request.form.get('duration')
        )

        predict_model = PredictModel()
        profit, filepath = predict_model.predict_from_model(data.model, data.duration)
        new_filepath = "http://127.0.0.1:5000/static/image/" + filepath.split('\\')[-1]
        logging.info(f"Filepath = {new_filepath}")

        logging.info(f'Logging the results :{profit}')
        return render_template('predict.html',profit=profit, image=new_filepath)

if __name__=="__main__":
    app.run(debug=True)