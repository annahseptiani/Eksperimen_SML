import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

def main():
    # 1. Integrasi Otomatis dari Awal dengan Otentikasi Non-Interaktif untuk GitHub Actions
    # Kita mengambil token DagsHub secara aman dari environment variable yang disediakan ci.yml
    dagshub_token = os.getenv("DAGSHUB_TOKEN")
    
    if dagshub_token:
        # Jika berjalan di GitHub Actions (Token terdeteksi), gunakan otentikasi otomatis
        dagshub.init(
            repo_owner='annahseptiani14', 
            repo_name='Eksperimen_SML', 
            mlflow=True,
            authentication="token",
            token=dagshub_token
        )
    else:
        # Jika Anda menjalankannya secara lokal di komputer Anda sendiri
        dagshub.init(repo_owner='annahseptiani14', repo_name='Eksperimen_SML', mlflow=True)
    
    # 2. Membaca file dataset secara aman menggunakan os.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "wine_preprocessed.csv")
    
    print(f"Memuat data dari folder lokal: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Menetapkan nama eksperimen di DagsHub / MLflow Tracking
    mlflow.set_experiment("Wine_Quality_Baseline")
    
    with mlflow.start_run():
        # 3. Proses Training Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Evaluasi Model
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # 5. Logging ke MLflow Tracking Server (DagsHub)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        mlflow.sklearn.log_model(model, "baseline_model")
        print("Model baseline berhasil dilatih dan dicatat ke DagsHub!")

        # ====================================================
        # TAMBAHAN UNTUK KRITERIA 3 ADVANCE (ARTIFACTS GITHUB)
        # ====================================================
        # A. Membuat file info metrik dalam format JSON
        metrics_dict = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec
        }
        metrics_path = os.path.join(current_dir, "metric_info.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Artefak metrik berhasil disimpan lokal di: {metrics_path}")

        # B. Membuat gambar Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        
        matrix_path = os.path.join(current_dir, "training_confusion_matrix.png")
        plt.savefig(matrix_path)
        plt.close()
        print(f"Artefak gambar berhasil disimpan lokal di: {matrix_path}")
        # ====================================================

if __name__ == "__main__":
    main()