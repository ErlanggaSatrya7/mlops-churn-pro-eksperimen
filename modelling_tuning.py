import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import dagshub

# 1. Koneksi ke DagsHub Repo Baru
dagshub.init(repo_owner='anggasatrya007', repo_name='mlops-churn-pro-eksperimen', mlflow=True)

# 2. Aktifkan Autolog TAPI matikan fungsi log_models bawaannya agar tidak tabrakan
mlflow.sklearn.autolog(log_models=False)

# 3. Ambil data dari folder preprocessing
df = pd.read_csv('preprocessing/data_siap_latih.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Set Nama Eksperimen Baru yang Bersih
mlflow.set_experiment("Telco-Churn-Fix-Artifact")

with mlflow.start_run() as run:
    # 5. Gunakan model standar tanpa GridSearchCV agar prosesnya langsung dan lurus
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. KUNCI UTAMA: Log model secara manual langsung ke folder bernama 'model'
    # Perintah ini yang akan memaksa file MLmodel, conda.yaml, dan model.pkl muncul di root
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
    
    print(f"Pelatihan Selesai! Run ID: {run.info.run_id}")