import numpy as np
import matplotlib.pyplot as plt

class SiecNeuronowa:
    def __init__(self, n_wejsc, n_ukrytych, n_wyjsc):
        self.n_wejsc = n_wejsc
        self.n_ukrytych = n_ukrytych
        self.n_wyjsc = n_wyjsc
        
        # inicjalizacja wag małymi losowymi wartosciami
        self.w1 = np.random.randn(n_wejsc, n_ukrytych) * 0.5
        self.b1 = np.random.randn(n_ukrytych) * 0.5
        self.w2 = np.random.randn(n_ukrytych, n_wyjsc) * 0.5
        self.b2 = np.random.randn(n_wyjsc) * 0.5
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_pochodna(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        # warstwa ukryta
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # warstwa wyjsciowa
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, wyjscie, lr):
        m = X.shape[0]
        
        # blad warstwy wyjsciowej - kryterium MSE
        delta2 = (wyjscie - y) * self.sigmoid_pochodna(self.z2)
        dw2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        # blad warstwy ukrytej
        delta1 = np.dot(delta2, self.w2.T) * self.sigmoid_pochodna(self.z1)
        dw1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        # aktualizacja wag
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
    
    def trenuj(self, X, y, epoki, lr):
        historia_bledu = []
        
        for epoka in range(epoki):
            wyjscie = self.forward(X)
            self.backward(X, y, wyjscie, lr)
            
            blad = np.mean((wyjscie - y) ** 2)
            historia_bledu.append(blad)
            
            if epoka % 1000 == 0:
                print(f"Epoka {epoka}, Blad: {blad:.6f}")
        
        return historia_bledu
    
    def przewiduj(self, X):
        return self.forward(X)


def zadanie_operacje_logiczne():
    print("\n=== ZADANIE 3: OPERACJE LOGICZNE ===\n")
    
    # zbior uczacy dla XOR
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y_xor = np.array([[0], [1], [1], [0]])
    y_and = np.array([[0], [0], [0], [1]])
    y_or = np.array([[0], [1], [1], [1]])
    
    operacje = [
        ("XOR", y_xor),
        ("AND", y_and),
        ("OR", y_or)
    ]
    
    for nazwa, y in operacje:
        print(f"\n--- Uczenie operacji {nazwa} ---")
        print(f"Zbior uczacy:\nX:\n{X}\ny:\n{y}")
        
        siec = SiecNeuronowa(2, 4, 1)
        print(f"\nWagi poczatkowe w1:\n{siec.w1}")
        print(f"Wagi poczatkowe w2:\n{siec.w2}")
        
        historia = siec.trenuj(X, y, epoki=5000, lr=0.5)
        
        print(f"\nWagi koncowe w1:\n{siec.w1}")
        print(f"Wagi koncowe w2:\n{siec.w2}")
        
        wyniki = siec.przewiduj(X)
        print(f"\nWyniki predykcji:")
        for i in range(len(X)):
            print(f"Wejscie: {X[i]} -> Wyjscie: {wyniki[i][0]:.4f} (Oczekiwane: {y[i][0]})")
        
        plt.figure(figsize=(8, 5))
        plt.plot(historia)
        plt.title(f'Blad uczenia - operacja {nazwa}')
        plt.xlabel('Epoka')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.savefig(f'/mnt/user-data/outputs/blad_{nazwa.lower()}.png')
        plt.close()


def zadanie_inny_zbior():
    print("\n=== ZADANIE 4: KLASYFIKACJA IRIS ===\n")
    
    # zbior Iris - uproszczony, 2 cechy, 3 klasy
    # sepal length, sepal width dla kazdego gatunku
    X = np.array([
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
        [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [5.5, 2.3], [6.5, 2.8],
        [6.3, 3.3], [5.8, 2.7], [7.1, 3.0], [6.3, 2.9], [6.5, 3.0],
        [4.9, 2.5], [4.8, 2.6], [5.0, 2.0], [5.2, 2.7], [5.4, 3.0]
    ])
    
    # kodowanie one-hot: setosa=[1,0,0], versicolor=[0,1,0], virginica=[0,0,1]
    y = np.array([
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]
    ])
    
    print("Zbior danych Iris (wybrane cechy):")
    print(f"Ksztalt X: {X.shape}, Ksztalt y: {y.shape}")
    print(f"\nPrzykladowe dane:\nX[:3]:\n{X[:3]}\ny[:3]:\n{y[:3]}")
    
    # normalizacja danych
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    
    siec = SiecNeuronowa(2, 5, 3)
    print(f"\nWagi poczatkowe w1:\n{siec.w1}")
    print(f"Wagi poczatkowe w2:\n{siec.w2}")
    
    historia = siec.trenuj(X_norm, y, epoki=10000, lr=0.3)
    
    print(f"\nWagi koncowe w1:\n{siec.w1}")
    print(f"Wagi koncowe w2:\n{siec.w2}")
    
    wyniki = siec.przewiduj(X_norm)
    przewidziane_klasy = np.argmax(wyniki, axis=1)
    prawdziwe_klasy = np.argmax(y, axis=1)
    
    dokladnosc = np.mean(przewidziane_klasy == prawdziwe_klasy)
    print(f"\nDokladnosc klasyfikacji: {dokladnosc * 100:.2f}%")
    
    print("\nPrzykladowe predykcje:")
    nazwy_klas = ["Setosa", "Versicolor", "Virginica"]
    for i in range(0, len(X), 5):
        print(f"Probka {i}: Przewidziana klasa: {nazwy_klas[przewidziane_klasy[i]]}, "
              f"Prawdziwa klasa: {nazwy_klas[prawdziwe_klasy[i]]}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(historia)
    plt.title('Blad uczenia - klasyfikacja Iris')
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig('/mnt/user-data/outputs/blad_iris.png')
    plt.close()


if __name__ == "__main__":
    zadanie_operacje_logiczne()
    zadanie_inny_zbior()
    print("\nWykresy zapisane w /mnt/user-data/outputs/")