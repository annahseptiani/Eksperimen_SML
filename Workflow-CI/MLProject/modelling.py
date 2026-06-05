import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

def main():
    # Integrasi otomatis dengan DagsHub online
    dagshub.init(repo_owner='annahseptiani14', repo_name='Eksperimen_SML', mlflow=True)
    
    # Karena sudah sejajar, kita bisa langsung membaca file secara aman dengan os.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "wine_preprocessed.csv")
    
    print(f"Memuat data dari folder lokal: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Wine_Quality_Baseline")
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # Manual Logging
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        mlflow.sklearn.log_model(model, "baseline_model")
        print("Model baseline berhasil dilatih dan dicatat ke DagsHub!")

if __name__ == "__main__":
    main()