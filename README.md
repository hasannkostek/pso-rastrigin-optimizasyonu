# Rastrigin Fonksiyonu Üzerinde Parçacık Sürü Optimizasyonu (PSO)

Bu proje, **İşlemsel Zeka** dersi kapsamında geliştirilmiştir. **Rastrigin Fonksiyonunun** global minimum noktasını bulmak için **Parçacık Sürü Optimizasyonu (PSO)** algoritması kullanılmıştır.

## 🎯 Proje Özeti
Rastrigin fonksiyonu, optimizasyon algoritmalarının performansını test etmek için kullanılan zorlu (birçok yerel minimumu olan) bir problemdir.

* **Amaç:** Rastrigin fonksiyonunu minimize etmek.
* **Algoritma:** Parçacık Sürü Optimizasyonu (PSO).
* **Parçacık Sayısı:** 40.
* **İterasyon Sayısı:** 150.

## 📊 Grafikler ve Analiz

Kod çalıştırıldığında iki temel grafik üretir:

### 1. Yakınsama Eğrisi (Convergence Curve)
Her iterasyonda sürünün bulduğu "en iyi" değeri gösterir. Eğrinin aşağı doğru inmesi, algoritmanın daha iyi çözümler bulduğunu kanıtlar.

### 2. Kontur Grafiği ve Sürü Konumu
Parçacıkların (mavi noktalar) fonksiyon üzerindeki hareketini ve en iyi çözüm noktasına (yıldız) nasıl toplandığını görselleştirir.

## 🛠️ Kurulum ve Çalıştırma

1. Projeyi indirin veya kopyalayın.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install numpy matplotlib
3. Uygulamayı çalıştırın:
   python pso_app.py

👨‍💻 Hazırlayan
Hasan Köstek - Bilgisayar Mühendisliği Öğrencisi
