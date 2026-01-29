# Hasan Köstek-22370031028 İşlemsel Zekâ Ödevi
# Parçacık Sürü Optimizasyonu (PSO) ile Rastrigin Fonksiyonu Optimizasyonu

import numpy as np
import matplotlib.pyplot as plt

# ==== 1) Problem: Rastrigin (2D) ====
def rastrigin(x):
    # x: (..., 2)
    A = 10
    return A*2 + (x[...,0]**2 - A*np.cos(2*np.pi*x[...,0])) + (x[...,1]**2 - A*np.cos(2*np.pi*x[...,1]))

bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])  # each row: [min, max]
dim = 2

# ==== 2) PSO Parametreleri ====
n_particles = 40
n_iters = 150
w = 0.72
c1 = 1.49
c2 = 1.49

rng = np.random.default_rng(42)
lower, upper = bounds[:,0], bounds[:,1]
span = upper - lower
vmax = 0.2 * span

# ==== 3) Başlat ====
X = rng.uniform(lower, upper, size=(n_particles, dim))
V = rng.uniform(-vmax, vmax, size=(n_particles, dim))

pbest_pos = X.copy()
pbest_fit = rastrigin(X)
gbest_idx = np.argmin(pbest_fit)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_fit = pbest_fit[gbest_idx]

history = [gbest_fit]

# ==== 4) Döngü ====
for t in range(1, n_iters+1):
    r1 = rng.random((n_particles, dim))
    r2 = rng.random((n_particles, dim))
    # Hız güncelle
    V = (w*V
         + c1*r1*(pbest_pos - X)
         + c2*r2*(gbest_pos - X))
    # Hız sınırı
    V = np.clip(V, -vmax, vmax)

    # Konum güncelle
    X = X + V
    # Sınırlar
    for d in range(dim):
        out_low = X[:,d] < lower[d]
        out_high = X[:,d] > upper[d]
        X[out_low, d] = lower[d]
        X[out_high, d] = upper[d]
        # Sınırı aşanların hızını sıfırla (daha stabil)
        V[out_low | out_high, d] = 0.0

    # Değerlendir
    fit = rastrigin(X)

    # pbest güncelle
    improved = fit < pbest_fit
    pbest_pos[improved] = X[improved]
    pbest_fit[improved] = fit[improved]

    # gbest güncelle
    if pbest_fit.min() < gbest_fit:
        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

    history.append(gbest_fit)

print(f"Best fitness: {gbest_fit:.6f} at position {gbest_pos}")

# ==== 5) Görseller ====
# 5a) Yakınsama eğrisi
plt.figure(figsize=(6,4))
plt.plot(history, lw=2)
plt.xlabel("İterasyon")
plt.ylabel("En iyi uygunluk (Rastrigin)")
plt.title("PSO Yakınsama Eğrisi")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5b) Kontur + son sürü konumu
# Kontur için 2B grid
grid_n = 300
gx = np.linspace(lower[0], upper[0], grid_n)
gy = np.linspace(lower[1], upper[1], grid_n)
GX, GY = np.meshgrid(gx, gy)
ZZ = rastrigin(np.stack([GX, GY], axis=-1))

plt.figure(figsize=(6,6))
cs = plt.contour(GX, GY, ZZ, levels=30)
plt.clabel(cs, inline=True, fontsize=7)
plt.scatter(X[:,0], X[:,1], s=25, label="Parçacıklar")
plt.scatter([gbest_pos[0]], [gbest_pos[1]], s=80, marker="*", label="gbest", zorder=5)
plt.title("PSO – Rastrigin Kontur Üzerinde Sürü")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

# Görsellerin Anlattıkları

# Yakınsama Grafiği:
# Bu grafikte, her iterasyonda sürünün en iyi uygunluk (fitness) değeri çizilmiştir. Eğri aşağıya doğru indikçe algoritmanın daha iyi çözümler bulduğu görülür.
# Grafiğin sonunda eğrinin sabitlenmesi, PSO’nun optimal sonuca ulaştığını veya yerel bir minimumda dengeye geldiğini gösterir.

# Kontur Grafiği (Rastrigin Fonksiyonu Üzerinde Sürü Hareketi):
# Bu görselde, Rastrigin fonksiyonunun yüzey haritası (kontur) üzerinde parçacıkların son iterasyondaki konumları gösterilmektedir.
# Her nokta bir parçacığı, yıldız (*) işareti ise sürünün bulduğu en iyi çözümü (gbest) temsil eder.
# Parçacıkların çoğunun aynı bölgeye toplanması, sürü zekâsının etkili bir şekilde küresel en iyiye yakınsadığını kanıtlar.
