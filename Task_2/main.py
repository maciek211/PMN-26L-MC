from src.data_loader import load_heart_disease_data, check_missing_data, preprocess_data
from src.visualization import plot_distributions
from src.model import run_experiment 
import matplotlib.pyplot as plt

# 1. Pobieranie danych
X, y = load_heart_disease_data()

# 2. Analiza przed czyszczeniem
print("--- Pre-cleaning analysis ---")
check_missing_data(X)

# 3. Czyszczenie danych
X_clean, y_clean = preprocess_data(X, y)

# 4. Wizualizacja
print("--- Chart generation ---")
plot_distributions(X_clean, y_clean)

# --- EKSPERYMENTY ---

print("--- Running experiments ---")

# Definiujemy głębokości drzewa do przetestowania
# To są nasze "parametry poddawane eksperymentom"
depths = [2, 3, 5, 10]

plt.figure(figsize=(10, 8))

for d in depths:
    # Uruchamiamy funkcję z pliku src/model.py
    fpr, tpr, roc_auc, score = run_experiment(X_clean, y_clean, d)
    
    # Rysujemy linię dla każdej głębokości na jednym wykresie
    plt.plot(fpr, tpr, label=f'Deep {d} (AUC = {roc_auc:.2f}, Acc = {score:.2f})')
    print(f"Deep: {d} | Accuracy: {score:.2f} | AUC: {roc_auc:.2f}")

# Dodajemy elementy wykresu ROC
plt.plot([0, 1], [0, 1], color='gray', linestyle='--') # Linia referencyjna (model losowy)
plt.xlabel('False Positive Rate') #(Błędne alarmy)
plt.ylabel('True Positive Rate') #(Wykrywalność)
plt.title('The ROC curve for different depths of the decision tree')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()