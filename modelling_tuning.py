import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub

# 1. Koneksikan ke DagsHub
dagshub.init(repo_owner='anggasatrya007', repo_name='mlops-churn-pro-eksperimen', mlflow=True)

# 2. Ambil data dari folder preprocessing (Sesuai Struktur Kriteria 1)
df = pd.read_csv('preprocessing/data_siap_latih.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Buat eksperimen di MLflow
mlflow.set_experiment("Telco-Churn-Tuning")

# 4. AKTIFKAN AUTOLOG (Wajib sesuai permintaan Reviewer Kriteria 2)
mlflow.sklearn.autolog()

with mlflow.start_run() as run:
    # 5. Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    
    grid = GridSearchCV(rf, params, cv=3)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    
    # 6. Manual Logging untuk Parameter Tambahan
    mlflow.log_param("best_n_estimators", grid.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid.best_params_['max_depth'])
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))
    
    # 7. Memaksa Log Model ke folder 'model' (Kunci kelulusan Kriteria 2)
    # Ini menjamin berkas MLmodel, conda.yaml, model.pkl masuk ke folder 'model'
    mlflow.sklearn.log_model(
        sk_model=best_model, 
        artifact_path="model", 
        registered_model_name="TelcoChurnModel"
    )
    
    # Artefak Tambahan
    disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    print(f"Pelatihan Selesai! Run ID: {run.info.run_id}")