from ucimlrepo import fetch_ucirepo 
import pandas as pd

def load_heart_disease_data():
    #Funkcja pobiera dane z UCI i przygotowuje je do analizy.

    # Pobieranie repozytorium o ID 45 (Heart Disease)
    heart_disease = fetch_ucirepo(id=45) 
    
    # X to nasze cechy (wiek, cholesterol itp.), y to cel (czy chory)
    X = heart_disease.data.features 
    y = heart_disease.data.targets 
    
    return X, y

def check_missing_data(df):
    #Sprawdza i wypisuje liczbę brakujących danych w każdej kolumnie.
    #df - Data Frame

    missing = df.isnull().sum()
    print("Number of missing values in the columns:")
    print(missing[missing > 0]) # Pokaż tylko te, gdzie braki istnieją
    return missing

def preprocess_data(X, y):
    """
    Czyści dane: uzupełnia braki i koduje zmienne kategoryczne.
    """
    # 1. Kopiujemy dane, żeby nie zmieniać oryginału
    X = X.copy()
    y = y.copy()

    # 2. Imputacja (uzupełnianie) braków - wybieramy medianę
    # Dlaczego mediana? Bo jest odporna na wartości odstające.
    X = X.fillna(X.median())

    # 3. Kodowanie zmiennych kategorycznych (One-Hot Encoding)
    # Wybieramy kolumny kategoryczne wg dokumentacji UCI
    cat_cols = ['cp', 'restecg', 'slope', 'thal', 'ca']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 4. Mapowanie y na binarny (0 = zdrowy, 1 = chory)
    # W oryginale są wartości 0,1,2,3,4. Zamieniamy wszystko > 0 na 1.
    y_binary = (y > 0).astype(int)

    return X, y_binary

# Krótki test, czy działa
if __name__ == "__main__":
    X, y = load_heart_disease_data()
    print("Data loaded successfully!")
    print(f"Number of samples: {len(X)}")
    check_missing_data(X)