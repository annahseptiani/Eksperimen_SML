import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def run_preprocessing(input_path, output_path):
    df = pd.read_csv(input_path)
    
    df = df.drop_duplicates()
    
    df['alcohol'] = df['alcohol'].fillna(df['alcohol'].median())
    df['pH'] = df['pH'].fillna(df['pH'].mean())
    
    Q1 = df['alcohol'].quantile(0.25)
    Q3 = df['alcohol'].quantile(0.75)
    IQR = Q3 - Q1
    df['alcohol'] = np.where(df['alcohol'] > (Q3 + 1.5 * IQR), (Q3 + 1.5 * IQR), df['alcohol'])
    df['alcohol'] = np.where(df['alcohol'] < (Q1 - 1.5 * IQR), (Q1 - 1.5 * IQR), df['alcohol'])
    
    df['pH_group'] = np.where(df['pH'] < 3.3, 'Asam', 'Normal')
    
    y = np.where(df['quality'] >= 6, 1, 0)
    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH_group', 'sulphates', 'alcohol']
    X = df[features].copy()
    
    X['pH_group'] = LabelEncoder().fit_transform(X['pH_group'])
    
    numeric_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']
    X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
    
    df_clean = X.copy()
    df_clean['target'] = y
    df_clean.to_csv(output_path, index=False)

if __name__ == "__main__":
    run_preprocessing("wine_raw.csv", "wine_preprocessed.csv")