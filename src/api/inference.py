import os
import io
import joblib
import pandas as pd
from datetime import datetime
from schemas import HousePredictionRequest, PredictionResponse

# --- Load Model and Preprocessor from S3 or Local Storage ---

# Get S3 bucket and file locations from environment variables
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
MODEL_KEY = os.getenv("MODEL_KEY", "production/model.pkl")
PREPROCESSOR_KEY = os.getenv("PREPROCESSOR_KEY", "production/preprocessor.pkl")

# Local fallback paths
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/trained/model.pkl")
LOCAL_PREPROCESSOR_PATH = os.getenv("LOCAL_PREPROCESSOR_PATH", "models/trained/preprocessor.pkl")

model = None
preprocessor = None

# Try loading from S3 first if credentials are available
if S3_BUCKET:
    try:
        import boto3
        print(f"Attempting to load artifacts from s3://{S3_BUCKET}...")
        
        # Initialize S3 client and download files into memory
        s3_client = boto3.client("s3")

        # Download and load the preprocessor
        preprocessor_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=PREPROCESSOR_KEY)
        preprocessor_bytes = io.BytesIO(preprocessor_obj['Body'].read())
        preprocessor = joblib.load(preprocessor_bytes)

        # Download and load the model
        model_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
        model_bytes = io.BytesIO(model_obj['Body'].read())
        model = joblib.load(model_bytes)

        print(f"Successfully loaded model and preprocessor from S3.")

    except Exception as e:
        print(f"Failed to load from S3: {str(e)}")
        print("Falling back to local model files...")
        model = None
        preprocessor = None

# Fallback to local files if S3 loading failed or credentials not available
if model is None or preprocessor is None:
    try:
        print(f"Loading artifacts from local storage...")
        print(f"Model path: {LOCAL_MODEL_PATH}")
        print(f"Preprocessor path: {LOCAL_PREPROCESSOR_PATH}")
        
        # Load preprocessor from local file
        if os.path.exists(LOCAL_PREPROCESSOR_PATH):
            preprocessor = joblib.load(LOCAL_PREPROCESSOR_PATH)
            print(f"Successfully loaded preprocessor from {LOCAL_PREPROCESSOR_PATH}")
        else:
            raise FileNotFoundError(f"Preprocessor not found at {LOCAL_PREPROCESSOR_PATH}")

        # Load model from local file
        if os.path.exists(LOCAL_MODEL_PATH):
            model = joblib.load(LOCAL_MODEL_PATH)
            print(f"Successfully loaded model from {LOCAL_MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Model not found at {LOCAL_MODEL_PATH}")

    except Exception as e:
        raise RuntimeError(f"Error loading model or preprocessor from local storage: {str(e)}")

# Final check to ensure both model and preprocessor are loaded
if model is None or preprocessor is None:
    raise RuntimeError("Failed to load model and preprocessor from both S3 and local storage.")


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
