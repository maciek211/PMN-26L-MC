from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def run_experiment(X, y, depth):
    """
    Trenuje model drzewa dla zadanej głębokości i zwraca dane do krzywej ROC.
    """
    # 1. Podział na zbiór treningowy (80%) i testowy (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Inicjalizacja modelu z konkretnym parametrem max_depth
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
    # 3. Trenowanie
    model.fit(X_train, y_train)

    # 4. Pobieranie prawdopodobieństwa dla krzywej ROC
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 5. Obliczanie punktów do wykresu ROC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, model.score(X_test, y_test)