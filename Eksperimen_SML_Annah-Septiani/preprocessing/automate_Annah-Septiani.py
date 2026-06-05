# Uji coba pemicu otomatis GitHub Actions untuk Kriteria 1
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing(input_path, output_path):
    print(f"Membaca file data mentah dari: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Menghapus Duplikat
    df = df.drop_duplicates()
    
    # 2. Imputasi Missing Values
    df['alcohol'] = df['alcohol'].fillna(df['alcohol'].median())
    df['pH'] = df['pH'].fillna(df['pH'].mean())
    
    # 3. Penanganan Outlier dengan metode IQR
    Q1 = df['alcohol'].quantile(0.25)
    Q3 = df['alcohol'].quantile(0.75)
    IQR = Q3 - Q1
    df['alcohol'] = np.where(df['alcohol'] > (Q3 + 1.5 * IQR), (Q3 + 1.5 * IQR), df['alcohol'])
    df['alcohol'] = np.where(df['alcohol'] < (Q1 - 1.5 * IQR), (Q1 - 1.5 * IQR), df['alcohol'])
    
    # 4. Feature Engineering (Penciptaan Fitur Baru)
    df['pH_group'] = np.where(df['pH'] < 3.3, 'Asam', 'Normal')
    
    # 5. Membuat Target Labelling
    y = np.where(df['quality'] >= 6, 1, 0)
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH_group', 'sulphates', 'alcohol']
    X = df[features].copy()
    
    # 6. Transformasi Fitur Kategorikal (Label Encoding)
    X['pH_group'] = LabelEncoder().fit_transform(X['pH_group'])
    
    # 7. Standardisasi Fitur Numerik
    numeric_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']
    X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
    
    # 8. Penggabungan Data Bersih dan Ekspor ke CSV
    df_clean = X.copy()
    df_clean['target'] = y
    
    print(f"Menyimpan file hasil preprocessing ke: {output_path}")
    df_clean.to_csv(output_path, index=False)
    print("Proses Preprocessing Selesai dan Sukses!")

if __name__ == "__main__":
    # --- PERBAIKAN JALUR FILE (PATH) DINAMIS DI SINI ---
    # Mendeteksi letak folder dari file automate_Annah-Septiani.py ini berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # wine_raw.csv berada di satu tingkat di luar folder preprocessing (folder induk)
    input_file = os.path.abspath(os.path.join(current_dir, "../wine_raw.csv"))
    
    # wine_preprocessed.csv disimpan di dalam folder preprocessing tempat script ini berada
    output_file = os.path.abspath(os.path.join(current_dir, "wine_preprocessed.csv"))
    
    run_preprocessing(input_file, output_file)