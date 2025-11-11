import os
import io
import boto3
import joblib
import pandas as pd
from datetime import datetime
from schemas import HousePredictionRequest, PredictionResponse

# --- Load Model and Preprocessor from S3 at Startup ---

# 1. Get S3 bucket and file locations from environment variables
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "labs-content")
MODEL_KEY = os.getenv("MODEL_KEY", "production/model.pkl")
PREPROCESSOR_KEY = os.getenv("PREPROCESSOR_KEY", "production/preprocessor.pkl")

# Check if the required environment variable is set
if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET_NAME environment variable is not set.")

try:
    print(f"Loading artifacts from s3://{S3_BUCKET}...")
    # 2. Initialize S3 client and download files into memory
    s3_client = boto3.client("s3")

    # Download and load the preprocessor
    preprocessor_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=PREPROCESSOR_KEY)
    preprocessor_bytes = io.BytesIO(preprocessor_obj['Body'].read())
    preprocessor = joblib.load(preprocessor_bytes)

    # Download and load the model
    model_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    model_bytes = io.BytesIO(model_obj['Body'].read())
    model = joblib.load(model_bytes)

    print(f"Successfully loaded model and preprocessor.")

except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor from S3: {str(e)}")


def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    # Prepare input data
    input_data = pd.DataFrame([request.dict()])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Dummy value for compatibility

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make prediction
    predicted_price = model.predict(processed_features)[0]

    # ... (rest of your prediction logic is the same) ...
    
    predicted_price = round(float(predicted_price), 2)
    confidence_interval = [round(float(value), 2) for value in [predicted_price * 0.9, predicted_price * 1.1]]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat()
    )

def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """
    Perform batch predictions.
    """
    input_data = pd.DataFrame([req.dict() for req in requests])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0

    processed_features = preprocessor.transform(input_data)
    predictions = model.predict(processed_features)
    return predictions.tolist()