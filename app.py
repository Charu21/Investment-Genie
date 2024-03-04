from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.logger import logging

from src.pipeline.predict_model import CustomData, PredictModel

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        duration_in_str = request.form.get('duration')
        duration_in_int = 0
        if duration_in_str == "Daily":
            duration_in_int = 3
        elif duration_in_str == "Weekly":
            duration_in_int = 7
        else: 
            duration_in_int = 30
        
        data = CustomData(
             model=request.form.get('model'),
             duration=duration_in_int
        )

        # pred_df = data.get_data_as_dataframe()
        # print(pred_df)

        predict_model = PredictModel()
        results = predict_model.evaluate_model(data.model, data.duration)

        logging.info(f'Logging the results :{results}')
        return render_template('home.html', results = results)
    
if __name__=="__main__":
    app.run(debug=True)