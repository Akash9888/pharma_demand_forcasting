# deploy_pipeline.py

# Import necessary libraries
from flask import Flask, request, jsonify
# Flask: lightweight web framework for handling HTTP requests and responses

import joblib
# joblib: efficient serialization library, used to load the pre-trained pipeline

import pandas as pd
# pandas: data manipulation library, used here to build DataFrame from user input


# =========================================
# 1. Initialize the Flask application
# =========================================
app = Flask(__name__)
# ‘app’ is our WSGI application; __name__ helps Flask locate static assets if any


# =========================================
# 2. Load the trained pipeline once at startup
# =========================================
pipeline = joblib.load('final_xgb_pipeline.pkl')
# The pipeline bundles all preprocessing (cleaning, feature‐engineering)
# and the trained XGBoost model into one object

FEATURE_NAMES = list(pipeline.feature_names_in_)
# Extract the list of raw feature names the pipeline expects
# (i.e., the original 10 input columns)


# =========================================
# 3. Define the prediction endpoint
# =========================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload with exactly the raw input features.
    Returns the model’s predicted quantity.
    """

    # 3.1 Parse JSON payload from the request body
    payload = request.get_json(force=True)
    # force=True ensures we attempt to parse JSON even if header isn’t set

    # 3.2 Convert payload into a one-row DataFrame in correct column order
    try:
        df_raw = pd.DataFrame([payload])[FEATURE_NAMES]
        # Selecting FEATURE_NAMES guarantees missing or extra keys will raise KeyError
    except KeyError as e:
        # Handle missing or unexpected feature keys
        missing = e.args[0]
        return (
            jsonify({
                "error": f"Missing or unknown feature in payload: '{missing}'",
                "expected_features": FEATURE_NAMES
            }),
            400
        )

    # 3.3 Delegate all preprocessing and inference to the pipeline
    y_pred = pipeline.predict(df_raw)[0]
    # pipeline.predict applies cleaning, feature creation, encoding, and model

    # 3.4 Format and return the JSON response
    return jsonify({
        "prediction": float(y_pred),
        "model_version": getattr(pipeline.named_steps['model'], 'version', None)
    }), 200
    # Returns HTTP 200 on success with the numeric prediction


# =========================================
# 4. Run the app (development server)
# =========================================
if __name__ == '__main__':
    # host='0.0.0.0' makes the server accessible externally (e.g., Docker containers)
    # port=5000 is the default Flask port; change if needed by host environment
    app.run(host='0.0.0.0', port=5000, debug=True)
    # debug=True enables auto-reload and detailed error pages (disable in production)
