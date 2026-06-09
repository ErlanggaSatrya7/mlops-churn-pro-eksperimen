from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, generate_latest
import uvicorn
import random

app = FastAPI()

# Metrik
REQUEST_COUNT = Counter("http_requests_total", "Total Permintaan API", ["method", "endpoint"])
PREDICTION_COUNT = Counter("model_predictions_total", "Total Prediksi")

# Endpoint Metrics
@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics():
    return generate_latest()

@app.get("/")
def root():
    return {"message": "API Monitoring Aktif"}

@app.get("/predict")
def predict():
    REQUEST_COUNT.labels(method="GET", endpoint="/predict").inc()
    PREDICTION_COUNT.inc()
    return {"status": "sukses", "prediksi": random.choice(["Churn", "No Churn"])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)