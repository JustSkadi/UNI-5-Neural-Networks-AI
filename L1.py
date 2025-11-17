import numpy as np
"""
Zadanie 1
W języku programowania wybranym przez prowadzącego (np. Matlab lub Python) zaimplemen-
tuj model perceptronu (prostego): unipolarnego i bipolarnego z jednym wejściem i z dwoma
wejściami (razem 4 perceptrony).
        x - dane wejściowe
        w - wagi danych x
        t - próg
        n - suma ważona
        f - funkcja aktywacji ( to co w return )
"""
def unip1(x, w, t):
    n = w * x
    return 1 if n >= t else 0

def unip2(x1, x2, w1, w2, t):
    n = w1 * x1 + w2 * x2
    return 1 if n >= t else 0

def bip1(x, w, t):
    n = w * x
    return 1 if n >= t else -1

def bip2(x1, x2, w1, w2, t):
    n = w1 * x1 + w2 * x2
    return 1 if n >= t else -1

"""
Zadanie 2
Naucz perceptron tak, aby realizował (osobno, czyli przeprowadź cztery niezależne uczenia)
operacje logiczne: LUB, I, NIE, ALBO. Ewentualnie stwierdź (i uzasadnij), że nauczenie jest
niewykonalne.
Naucz, czyli: wybierz postać ciągu uczącego, wybierz wagi początkowe, zaprezentuj wyniki cząst-
kowe (zbiór uczący, wagi, wartości kryterium, sumy cząstkowe, wyjścia z perceptronu).
        X - macierz wejść
        y - wektor wyjść oczekiwanych
        eta - współczynnik uczenia ( jak szybko się uczy )
        max_e - maksymalna liczba epok ( przejść przez loopy uczące )
        b - bias ( stała, przesuwająca funkcje aktywacji )
"""
def uczenie(X, y):
    eta=0.1
    max_epochs=100
    przyklady, cechy = X.shape
    w = np.zeros(cechy)
    b = 0
    pom = []
    pom2 = []
    
    for epoch in range(max_epochs):
        errors = 0
        for i in range(przyklady):
            n = np.dot(w, X[i]) + b # dot - iloczyn skalarny
            y_pred = 1 if n >= 0 else 0
            if y_pred != y[i]:
                errors += 1
                error = y[i] - y_pred
                w += eta * error * X[i]
                b += eta * error
            if i == 1 and epoch == 0:
                pom2.append({
                    "suma wag": n,
                    'wagi': w.copy(),
                    'bias': b,
                    'errors': errors
                })
                
        pom.append({
            'epoka': epoch + 1,
            'wagi': w.copy(),
            'bias': b,
            'errors': errors
        })
    return w, b, pom, pom2

# OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])
w_or, b_or, hist_or, pom2 = uczenie(X_or, y_or)
# print(pom2)
# print(f"Wagi końcowe: {w_or}, Bias: {b_or}")
# print(f"OR: {h_or}")

# AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
w_and, b_and, hist_and, pom2 = uczenie(X_and, y_and)
# print(pom2)
# print(f"Wagi końcowe: {w_and}, Bias: {b_and}")
# for i in range (len(hist_and)):
#     print(f"{hist_and[i]}\n")

# NOT
X_not = np.array([[0], [1]])
y_not = np.array([1, 0])
w_not, b_not, hist_not, pom2 = uczenie(X_not, y_not)
# print(pom2)
# print(f"Wagi końcowe: {w_not}, Bias: {b_not}")
# print(f"NOT: {h_not}")

# XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
w_xor, b_xor, hist_xor, pom2 = uczenie(X_xor, y_xor)
# print(pom2)
# print(f"Wagi końcowe: {w_xor}, Bias: {b_xor}")
# print(f"XOR: {h_xor}")
# for i in range (len(hist_xor)):
#     print(f"{hist_xor[i]}\n")
"""
XOR ( Albo ) się nie da, bo nie da się podzielić wyjść jedną linią na 2 kategorie ( JAK SIĘ ROZRYSUJE NA WYKRESIE TO WIDAĆ )
"""

"""
Zadanie 3
Uzasadnij wyniki uzyskane w zadaniu nr 2 odwołując się do modelu perceptronu (wzorów) i do reguły perceptronu.
Pierwsze dwie iteracje każdego uczenia (z zadania nr 2) wykonaj samodzielnie, na kartce papieru.
"""
# Rozpisane na tablecie, wzory są w głównej pętli funkcji "uczenie"

"""
Zadanie 4
Zaprojektuj i zaimplementuj sztuczną sieć neuronową umożliwiającą poprawną realizację
tych formuł logicznych, których nie udało się zrealizować za pomocą pojedynczego perceptronu.
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class XOR:
    def __init__(self):
        np.random.seed(42)
        self.w1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.random.randn(2) * 0.5
        self.w2 = np.random.randn(2) * 0.5
        self.b2 = np.random.randn() * 0.5
    
    def f(self, x):
        suma1 = np.dot(x, self.w1) + self.b1
        self.h = sigmoid(suma1)
        
        suma2 = np.dot(self.h, self.w2) + self.b2
        y = sigmoid(suma2)
        return y
    
    def train(self, X, y, epochs=10000, eta=1.0):
        for epoch in range(epochs):
            for i in range(len(X)):
                out = self.f(X[i])
                blad = y[i] - out
                
                # popraw wagi
                delta2 = blad * out * (1 - out)
                self.w2 += eta * delta2 * self.h
                self.b2 += eta * delta2
                
                delta1 = delta2 * self.w2 * self.h * (1 - self.h)
                self.w1 += eta * np.outer(X[i], delta1)
                self.b1 += eta * delta1
siec = XOR()
siec.train(X_xor, y_xor, epochs=10000, eta=1.0)
print("\nWYNIKI:")
for i in range(len(X_xor)):
    wynik = siec.f(X_xor[i])
    predykcja = 1 if wynik >= 0.5 else 0
    print(f"x = {X_xor[i]}, oczekiwane = {y_xor[i]}, wyjście = {wynik:.4f}, predykcja = {predykcja}")

"""
Zadanie 5
Zaprojektuj inne zbiory wartości, których kształtna klasyfikacja nie jest możliwa
z wykorzystaniem pojedynczego perceptronu. Wyciąg uzasadnij.
"""
def inny():
    X_szachownica = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0, 2], [0, 3], [1, 2], [1, 3],
        [2, 0], [2, 1], [3, 0], [3, 1],
        [2, 2], [2, 3], [3, 2], [3, 3]
    ])
    y_szachownica = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
# To samo co z XOR tylko większe, problem wielowymiarowości