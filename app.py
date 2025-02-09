import sys
import os
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add the project root directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT)

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    traceback.print_exc()  # Print traceback for debugging
    return jsonify({"error": str(e)}), 500

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                Ticker=request.form.get('ticker'),  # Match the form field names
                Date=request.form.get('date'),
                Open=float(request.form.get('open')),
                High=float(request.form.get('high')),
                Low=float(request.form.get('low')),
                Close=float(request.form.get('close')),
                Volume=int(request.form.get('volume'))
            )

            # Prepare the data for prediction
            pred_df = data.get_data_as_data_frame()
            print("Input Data for Prediction:", pred_df)

            # Initialize prediction pipeline
            predict_pipeline = PredictPipeline()
            print("Mid Prediction")

            # Make predictions
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")

            # Return results to the template
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error occurred during prediction. Please check your inputs.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
