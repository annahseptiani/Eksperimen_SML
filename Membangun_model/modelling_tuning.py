import os
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

def main():
    dagshub.init(repo_owner='annahseptiani14', repo_name='Eksperimen_SML', mlflow=True)
    
    # Karena sudah sejajar, kita bisa langsung membaca file secara aman dengan os.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "wine_preprocessed.csv")
    
    print(f"Memuat data dari folder lokal: {data_path}")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Wine_Quality_Tuning")
    
    with mlflow.start_run():
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 5, 10]
        }
        
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        preds = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
        # Artefak 1: Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        
        plot_path = os.path.join(current_dir, "training_confusion_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        mlflow.log_artifact(plot_path)
        
        # Artefak 2: JSON Report
        report_data = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "total_test_samples": len(y_test)
        }
        report_path = os.path.join(current_dir, "metric_info.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=4)
        mlflow.log_artifact(report_path)
        
        mlflow.sklearn.log_model(best_model, "tuned_model")
        print("Model tuning berhasil di-track ke DagsHub bersama 2 artefak tambahan!")

if __name__ == "__main__":
    main()