import numpy as np
from sklearn.datasets import load_iris
"""
1.1 Zadanie nr 1
W języku programowania wybranym przez prowadzącego (np. Matlab lub Python) zaimplemen-
tuj sztuczną sieć neuronową z warstwą ukrytą, z N wejściami i M wyjściami.

1.2 Zadanie nr 2
Zaimplementuj algorytm uczenia z propagacją wsteczną dla kryterium zadanego przez prowa-
dzącego.
Implementację poprzyj wyprowadzeniami na kartce.
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class SiecNeuronowa:
    def __init__(self, n_IN, n_H, n_OUT):
        self.n_IN = n_IN
        self.n_H = n_H
        self.n_OUT = n_OUT
        self.w1 = np.random.randn(n_IN, n_H) * 0.5
        self.b1 = np.random.randn(n_H) * 0.5
        self.w2 = np.random.randn(n_H, n_OUT) * 0.5
        self.b2 = np.random.randn(n_OUT) * 0.5
    
    def forward(self, X):
        self.suma1 = np.dot(X, self.w1) + self.b1
        self.h = sigmoid(self.suma1) # aktywacja wartw ukrytych
        self.suma2 = np.dot(self.h, self.w2) + self.b2
        self.out = sigmoid(self.suma2) # predykcja
        return self.out
    
    # https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
    def backward(self, X, y, out, eta):
        E_local = out - y
        # gradient lokalny, wag i biasów out
        delta2 = E_local * out * (1 - out)
        dw2 = np.dot(self.h.T, delta2)
        db2 = np.sum(delta2, axis=0)
        
        # ukryta
        delta1 = np.dot(delta2, self.w2.T) * self.h * (1 - self.h)
        dw1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)
        
        # zmiana wag i biasów
        self.w2 -= eta * dw2
        self.b2 -= eta * db2
        self.w1 -= eta * dw1
        self.b1 -= eta * db1
    
    def train(self, X, y, epochs, eta):
        for epoch in range(epochs):
            out = self.forward(X)
            self.backward(X, y, out, eta)
            if epoch % 1000 == 0:
                mse = np.mean((out - y) ** 2)
                print(f"Epoka {epoch}, Blad: {mse:.6f}")
        
"""
1.3 Zadanie nr 3
Naucz sieć podstawowych operacji logicznych.
Naucz, czyli: wybierz postać ciągu uczącego, wybierz wagi początkowe, zaprezentuj wyniki cząst-
kowe (zbiór uczący, wagi, wartości kryterium, sumy cząstkowe, wyjścia z warstw sieci).
"""

def zad3():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    y_and = np.array([[0], [0], [0], [1]])
    y_or = np.array([[0], [1], [1], [1]])
    op = [
        ("XOR", y_xor),
        ("AND", y_and),
        ("OR", y_or)
    ]
    
    for nazwa, y in op:
        print(f"\n--- {nazwa} ---")
        siec = SiecNeuronowa(2, 4, 1)
        print(f"Wagi pocz w1:\n{siec.w1}")
        print(f"Wagi pocz w2:\n{siec.w2}")
        
        siec.train(X, y, epochs=5000, eta=0.5)
        
        print(f"Wagi fin w1:\n{siec.w1}")
        print(f"Wagi fin w2:\n{siec.w2}")
        
        wyniki = siec.forward(X)
        for i in range(len(X)):
            print(f"IN: {X[i]} -> OUT: {wyniki[i][0]:.4f} (Oczekiwalam: {y[i][0]})")

"""
1.4 Zadanie nr 4
Przygotuj (znajdź) inny zbiór danych. Przeprowadź uczenie sieci na przygotowanym zbiorze
danych. Zaprezentuj wyniki.
"""
def zad4():
    iris = load_iris()
    X = iris.data
    y_labels = iris.target
    y = np.zeros((len(y_labels), 3))
    # one hot encoding - zamiana etykiet na wektor binarny dla wyjsc
    for i, label in enumerate(y_labels):
        y[i, label] = 1
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0) # 0 1
    
    siec = SiecNeuronowa(4, 5, 3) # 4 cechy 3 klasy
    print(f"Wagi pocz w1:\n{siec.w1}")
    print(f"Wagi pocz w2:\n{siec.w2}")
    
    siec.train(X_norm, y, epochs=10000, eta=0.3)
    
    print(f"Wagi fin w1:\n{siec.w1}")
    print(f"Wagi fin w2:\n{siec.w2}")
    
    wyniki = siec.forward(X_norm)
    przewidziane = np.argmax(wyniki, axis=1)
    prawdziwe = np.argmax(y, axis=1)
    dokladnosc = np.mean(przewidziane == prawdziwe)
    print(f"Dokladnosc klasyfikacji: {dokladnosc * 100:.2f}%\n")
    
    nazwy = ["Setosa", "Versicolor", "Virginica"]
    for i in range(0, len(X), 30):
        print(f"Probka {i}: Przewidziana: {nazwy[przewidziane[i]]}, "
              f"Prawdziwa: {nazwy[prawdziwe[i]]}")

if __name__ == "__main__":
    # print(f"Zadanie 3:\n")
    # zad3()
    print(f"\n{'-'*80}\n")
    print(f"Zadanie 4:\n")
    zad4()