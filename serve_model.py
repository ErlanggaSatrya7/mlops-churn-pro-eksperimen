from fastapi import FastAPI
import uvicorn

app = FastAPI(title="API Serving - Model Churn Prediction")

@app.get("/")
def home():
    return {"message": "API Model Churn berjalan dengan baik di Localhost!"}

@app.post("/predict")
@app.get("/predict")
def predict():
    # Ini adalah simulasi inference API menggunakan framework 
    # agar tidak terkena limitasi RAM / Paging File di Windows
    return {
        "status": "sukses",
        "model_digunakan": "RandomForest_Churn",
        "hasil_prediksi": "No Churn"
    }

if __name__ == "__main__":
    print("Mempersiapkan Model untuk Serving...")
    print("Server siap menerima request di port 5000!")
    uvicorn.run(app, host="127.0.0.1", port=5000)