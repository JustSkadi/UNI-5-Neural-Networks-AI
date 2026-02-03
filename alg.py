import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np

def fitness(x): # MAX
    return x * math.sin(10 * math.pi * x) + 2.0

def plot_function():
    x = np.linspace(-1, 2, 1000)
    y = [fitness(xi) for xi in x]
    opt_x = 1.8504
    opt_y = fitness(opt_x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.axhline(y=opt_y, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=opt_x, color='r', linestyle='--', alpha=0.5)
    plt.plot(opt_x, opt_y, 'ro', markersize=10, label=f'Optimum: x={opt_x:.4f}, f(x)={opt_y:.4f}')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.savefig('wykres.png', dpi=300, bbox_inches='tight')
    plt.close()

class GA:
    def __init__(self, pop_size, chrom_len, p_cross, p_mut, max_gen, elitism=True):
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.max_gen = max_gen
        self.elitism = elitism
        self.x_min = -1
        self.x_max = 2
    
    def populacja(self):
        pop = []
        for _ in range(self.pop_size):
            chrom = [random.randint(0, 1) for _ in range(self.chrom_len)]
            pop.append(chrom)
        return pop
    
    def decode(self, chrom):
        val = 0
        for i, bit in enumerate(chrom):
            val += bit * (2 ** (self.chrom_len - 1 - i)) # zmiana na 10
        max_val = 2**self.chrom_len - 1
        x = self.x_min + (self.x_max - self.x_min) * (val / max_val) # do przedziału
        return x
    
    def ocen_osobnika(self, chrom):
        x = self.decode(chrom)
        return fitness(x)
    
    def ocen_pop(self, pop):
        scores = []
        for chrom in pop:
            scores.append(self.ocen_osobnika(chrom))
        return scores
    
    def ruletka(self, pop, scores):
        sum_fitness = sum(scores)
        if sum_fitness <= 0:
            min_score = min(scores)
            scores_adj = [s - min_score + 1 for s in scores]
            sum_fitness = sum(scores_adj)
        else:
            scores_adj = scores
        
        probs = [s / sum_fitness for s in scores_adj] # normalizacja sumy prawdopodobieństw do 1
        idx = random.choices(range(len(pop)), weights=probs, k=1)[0] # losowanie osobnika
        return pop[idx], probs[idx]
    
    def krzyz(self, p1, p2):
        if random.random() < self.p_cross: # robimy?
            point = random.randint(1, self.chrom_len - 1)
            c1 = p1[:point] + p2[point:]
            c2 = p2[:point] + p1[point:]
            return c1, c2
        else:
            return p1[:], p2[:]
    
    def mutate(self, chrom):
        mut = chrom[:]
        for i in range(len(mut)):
            if random.random() < self.p_mut:
                mut[i] = 1 - mut[i]
        return mut
    
    def run(self):
        pop = self.populacja()
        best_hist = []
        avg_hist = []
        start = time.time()
        
        for gen in range(self.max_gen):
            scores = self.ocen_pop(pop)
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            best_hist.append(best_score)
            avg_hist.append(avg_score)
            
            new_pop = []
            
            if self.elitism:
                best_idx = scores.index(max(scores))
                new_pop.append(pop[best_idx][:])
            
            while len(new_pop) < self.pop_size:
                p1, w1 = self.ruletka(pop, scores)
                p2, w2 = self.ruletka(pop, scores)
                c1, c2 = self.krzyz(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)
            
            print(f"Rodzice: {p1} , {p2}, {w1} , {w2}")
            pop = new_pop
        
        elapsed = time.time() - start
        
        final_scores = self.ocen_pop(pop)
        best_idx = final_scores.index(max(final_scores))
        best_chrom = pop[best_idx]
        best_x = self.decode(best_chrom)
        best_val = max(final_scores)
        
        return {
            'best_x': best_x,
            'best_val': best_val,
            'time': elapsed,
            'best_hist': best_hist,
            'avg_hist': avg_hist
        }


def test_params():
    for size in [3]: # 10, 30, 70
        ga = GA(
            pop_size=size,
            chrom_len=10,
            p_cross=0.5,
            p_mut=0.04,
            max_gen=20,
            elitism=True
        )
        result = ga.run()
        print(f"Rozmiar populacji: {size}")
        print(f"Najlepsze x: {result['best_x']:.4f}")
        print(f"Wartosc funkcji: {result['best_val']:.4f}")
        print(f"Czas: {result['time']:.4f}s")
    
    print("\n")
    for pc in [0.05, 0.3, 0.65, 0.95]:
        ga = GA(
            pop_size=20,
            chrom_len=10,
            p_cross=pc,
            p_mut=0.04,
            max_gen=20,
            elitism=True
        )
        result = ga.run()
        print(f"Pr. krzyzowania: {pc}")
        print(f"Najlepsze x: {result['best_x']:.4f}")
        print(f"Wartosc funkcji: {result['best_val']:.4f}")
    
    print("\n")
    for pm in [0.0005, 0.008, 0.06, 0.3]:
        ga = GA(
            pop_size=20,
            chrom_len=10,
            p_cross=0.5,
            p_mut=pm,
            max_gen=20,
            elitism=True
        )
        result = ga.run()
        print(f"Pr. mutacji: {pm}")
        print(f"Najlepsze x: {result['best_x']:.4f}")
        print(f"Wartosc funkcji: {result['best_val']:.4f}")
    
    print("\n")
    for elit in [False, True]:
        ga = GA(
            pop_size=20,
            chrom_len=10,
            p_cross=0.5,
            p_mut=0.04,
            max_gen=20,
            elitism=elit
        )
        result = ga.run()
        print(f"Elitaryzm: {elit}")
        print(f"Najlepsze x: {result['best_x']:.4f}")
        print(f"Wartosc funkcji: {result['best_val']:.4f}")
    
    print("\n")
    for chlen in [5, 8, 12, 16]:
        ga = GA(
            pop_size=20,
            chrom_len=chlen,
            p_cross=0.5,
            p_mut=0.04,
            max_gen=20,
            elitism=True
        )
        result = ga.run()
        print(f"Dlugosc chromosomu: {chlen}")
        print(f"Najlepsze x: {result['best_x']:.4f}")
        print(f"Wartosc funkcji: {result['best_val']:.4f}")
    
    print("\n")
    for gen in [5]: #  15, 40, 80
        ga = GA(
            pop_size=20,
            chrom_len=10,
            p_cross=0.5,
            p_mut=0.04,
            max_gen=gen,
            elitism=True
        )
        result = ga.run()
        print(f"Liczba generacji: {gen}")
        print(f"Najlepsze x: {result['best_x']:.4f}")
        print(f"Wartosc funkcji: {result['best_val']:.4f}")

if __name__ == "__main__":
    # plot_function()
    # print("\n")
    test_params()