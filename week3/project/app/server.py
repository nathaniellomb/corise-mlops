import json
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from datetime import datetime

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    [TO BE IMPLEMENTED]
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    global classifier
    global log_file

    classifier = NewsCategoryClassifier(verbose=True)
    classifier.load(MODEL_PATH)
    log_file = open(LOGS_OUTPUT_PATH, 'a', buffering=1)

    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    [TO BE IMPLEMENTED]
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    log_file.flush()
    log_file.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    Run model inference and get model predictions for model inputs specified in request
    """
    start = datetime.now()
    scores = classifier.predict_proba(request.__dict__)
    label = classifier.predict_label(request.__dict__)
    response = PredictResponse(scores = scores, label = label)
    end = datetime.now()

    log_file.write(json.dumps({
        'timestamp': start.strftime("<%Y:%m:%d %H:%M:%S>"),
        'request': request.__dict__,
        'response': response.__dict__,
        'latency': (end - start).microseconds / 1000
    }, indent=2) + '\n')
    
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
