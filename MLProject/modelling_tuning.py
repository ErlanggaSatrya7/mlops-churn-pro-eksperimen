import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import dagshub
import os

# 1. Inisialisasi DagsHub
dagshub.init(repo_owner='anggasatrya007', repo_name='mlops-churn-pro-eksperimen', mlflow=True)

# 2. LOAD DATA
df = pd.read_csv('preprocessing/data_siap_latih.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. TRAINING MANUAL
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. LOG MANUAL (Ini yang akan memaksa folder 'model' muncul)
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model", # Folder ini yang akan muncul di DagsHub!
        registered_model_name="TelcoChurnModel"
    )
    print("Model berhasil di-log secara manual ke folder 'model'")