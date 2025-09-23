import kfp
from kfp.dsl import pipeline, component, InputPath, OutputPath

BASE_IMAGE = "docker.io/deepaktyagi048/house-price-base"

@component(base_image=BASE_IMAGE)
def process_data(
    raw_data_path: str,
    cleaned_data_csv: OutputPath("csv"),
):
    """Cleans and preprocesses the raw housing data."""
    import subprocess
    subprocess.run([
        'python', 'src/data/run_processing.py',
        '--input', raw_data_path,
        '--output', cleaned_data_csv
    ], check=True)


@component(base_image=BASE_IMAGE)
def engineer_features(
    cleaned_data_csv: InputPath("csv"),
    featured_data_csv: OutputPath("csv"),
    preprocessor_pkl: OutputPath("pkl"),
):
    """Applies transformations and generates features."""
    import subprocess
    subprocess.run([
        'python', 'src/features/engineer.py',
        '--input', cleaned_data_csv,
        '--output', featured_data_csv,
        '--preprocessor', preprocessor_pkl
    ], check=True)


@component(base_image=BASE_IMAGE, packages_to_install=['boto3'])
def train_and_upload_model(
    config_path: str,
    featured_data_csv: InputPath("csv"),
    preprocessor_in: InputPath("pkl"),
    s3_bucket: str,
    aws_access_key_id: str,    
    aws_secret_access_key: str,  
    trained_model_pkl: OutputPath("pkl") 
):
    """Trains the model AND uploads the final artifacts to S3."""
    import subprocess
    import boto3
    import os

    aws_credentials_path = os.path.expanduser("~/.aws/credentials")
    os.makedirs(os.path.dirname(aws_credentials_path), exist_ok=True)
    with open(aws_credentials_path, "w") as f:
        f.write("[default]\n")
        f.write(f"aws_access_key_id = {aws_access_key_id}\n")
        f.write(f"aws_secret_access_key = {aws_secret_access_key}\n")
    # -----------------------------------------------------------

    # Part 1: Train the model
    subprocess.run(['python', 'src/models/train_model.py', '--config', config_path, '--data', featured_data_csv, '--output-model-path', trained_model_pkl], check=True)

    # Part 2: Upload artifacts using boto3
    s3_client = boto3.client('s3')
    preprocessor_key = 'production/preprocessor.pkl'
    model_key = 'production/model.pkl'

    s3_client.upload_file(Filename=preprocessor_in, Bucket=s3_bucket, Key=preprocessor_key)
    s3_client.upload_file(Filename=trained_model_pkl, Bucket=s3_bucket, Key=model_key)
    print(f"Successfully uploaded artifacts to s3://{s3_bucket}/production/")

# ----------------------------- Compile the pipeline -----------------------------

@pipeline(name='House Price Direct Credentials Pipeline')
def house_price_pipeline(
    raw_data_path: str,
    config_path: str,
    s3_bucket: str,
    aws_access_key_id: str,      
    aws_secret_access_key: str  
):
    # Step 1: Process Data
    process_data_task = process_data(raw_data_path=raw_data_path)
    process_data_task.set_caching_options(enable_caching=False)
    
    # Step 2: Engineer Features
    # This step uses the output from the previous step as its input
    engineer_features_task = engineer_features(
        cleaned_data_csv=process_data_task.outputs['cleaned_data_csv']
    )

    engineer_features_task.set_caching_options(enable_caching=False)       
    
    # Step 3
    train_upload_task = train_and_upload_model(
        config_path=config_path,
        featured_data_csv=engineer_features_task.outputs['featured_data_csv'],
        preprocessor_in=engineer_features_task.outputs['preprocessor_pkl'],
        s3_bucket=s3_bucket,
        aws_access_key_id=aws_access_key_id,          
        aws_secret_access_key=aws_secret_access_key    
    )
    train_upload_task.after(engineer_features_task)

if __name__ == '__main__':
    import os
    
    # --- Read credentials from your LOCAL environment variables ---
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not aws_key or not aws_secret:
        raise ValueError("AWS credentials not found in local environment variables.")
    # --------------------------------------------------------------

    kfp.compiler.Compiler().compile(
        pipeline_func=house_price_pipeline,
        package_path='house_price_pipeline_direct_creds.yaml',
        # Pass the credentials from your local machine into the pipeline
        pipeline_parameters={
            'raw_data_path': 'data/raw/house_data.csv',
            'config_path': 'configs/model_config.yaml',
            's3_bucket': 'labs-content', 
            'aws_access_key_id': aws_key,
            'aws_secret_access_key': aws_secret
        }
    )
