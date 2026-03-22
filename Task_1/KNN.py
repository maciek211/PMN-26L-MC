from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
  

# 1. POBIERANIE DANYCH  
# fetch dataset 
#idzie do archiwum (UCI) i przynosi na zbiór o id 53 (Iris)
iris = fetch_ucirepo(id=53)  
# data (as pandas dataframes) 
X = iris.data.features #tabel z wymiarami (centymetry)
y = iris.data.targets  #lista nazw (nazwy gatunków)
  
# metadata 
#print(iris.metadata) 
  
# variable information 
# print(iris.variables) 
print(X.describe())

# 2. Podział na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Skalowanie danych (żeby każda cecha miała takie same znaczenie)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Tworzymy model - powiedzmy, że sprawdzamy 3 najbliższych sąsiadów
knn = KNeighborsClassifier(n_neighbors=3)

# 5. "Uczymy" model na danych treningowych
knn.fit(X_train_scaled, y_train.values.ravel()) # .values.ravel() tylko "prostuje" dane dla biblioteki

# 6. Prosimy model, żeby zgadł gatunki dla 30 kwiatów, których nie widział
y_pred = knn.predict(X_test_scaled)
# print(y_pred)

# Wyświetlamy raport
print("\n--- RAPORT KLASYFIKACJI ---")
print(classification_report(y_test, y_pred))
                            
# Macierz pomyłek - pokaże nam dokładnie, co z czym się pomyliło
print("\n--- MACIERZ POMYŁEK ---")
print(confusion_matrix(y_test, y_pred))

#Tworzymy "maszynkę" t-SNE, która zrobi nam 2 wymiary (komponenty)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)

#Przerabiamy nasze 4-wymiarowe dane testowe na 2-wymiarowe
X_tsne = tsne.fit_transform(X_test_scaled)


# --- SEKCJA RYSOWANIA WYKRESU ---
# Przygotowujemy "tłumacza" nazw na liczby dla legendy
le = LabelEncoder()
y_encoded = le.fit_transform(y_pred) 

plt.figure(figsize=(10, 7))

# Rysujemy punkty (używamy y_encoded do kolorów)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded, cmap='viridis')

# Dodajemy legendę (pobiera nazwy gatunków z LabelEncoder)
plt.legend(handles=scatter.legend_elements()[0], labels=list(le.classes_))

# Dodajemy podpisy osi i tytuł
plt.title("Wizualizacja t-SNE (Wyniki klasyfikacji KNN)")
plt.xlabel("Wymiar t-SNE 1")
plt.ylabel("Wymiar t-SNE 2")

plt.show()