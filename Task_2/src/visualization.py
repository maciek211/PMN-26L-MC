import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(X, y):
    # Wykres 1: Rozkład wieku
    plt.figure(figsize=(10, 5))
    sns.histplot(X['age'], kde=True, color='skyblue')
    plt.title('Age distribution of patients')
    plt.show()

    # Wykres 2: Liczebność klas (Zdrowi vs Chorzy)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y.iloc[:, 0], palette='magma')
    plt.title('Number of patients: Healthy (0) versus Sick (1)')
    plt.show()