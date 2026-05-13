# İstanbul Trafik Yoğunluğu Model Performans Raporu

## Veri Özeti
- **Train Satır Sayısı:** 400,000
- **Validation Satır Sayısı:** 198,827
- **Test Satır Sayısı:** 181,569
- **Özellik (Feature) Sayısı:** 33

---

## 1. Regresyon Sonuçları
Hedef değişken: `congestion_score` (Trafik Yoğunluk Skoru)

| Model | Val RMSE | Test RMSE | Test MAE | Test R² | Test MAPE (%) |
|-------|----------|-----------|----------|---------|---------------|
| **XGBoost (Seçilen)** | **0.0841** | **0.1017** | **0.0700** | **0.8233** | **19.15** |
| LightGBM | 0.0854 | 0.1018 | 0.0700 | 0.8230 | 19.26 |
| RandomForest | 0.0883 | 0.1033 | 0.0692 | 0.8178 | 19.14 |

---

## 2. Sınıflandırma Sonuçları
Hedef değişken: `traffic_status` (Akıcı, Yoğun, Kilit)

| Model | Val F1 (Ağırlıklı) | Test Doğruluk (Acc) | Test F1 (Ağırlıklı) | Test F1 (Makro) | Sınıf Bazlı F1 (Akıcı / Kilit / Yoğun) |
|-------|--------------------|---------------------|---------------------|-----------------|----------------------------------------|
| **XGBoost_clf (Seçilen)** | **0.8563** | **0.8460** | **0.8460** | **0.8459** | **0.8492 / 0.8411 / 0.8475** |
| LightGBM_clf | 0.8536 | 0.8431 | 0.8431 | 0.8430 | 0.8460 / 0.8381 / 0.8448 |
| RandomForest_clf | 0.8227 | 0.8116 | 0.8110 | 0.8160 | 0.8363 / 0.8118 / 0.7998 |
| LogisticRegression_baseline | 0.7429 | 0.7333 | 0.7306 | 0.7380 | 0.7484 / 0.7585 / 0.7070 |

---

## 3. Confusion Matrix (XGBoost_clf Test Seti Üzerinde)

| Gerçek \ Tahmin | Akıcı | Kilit | Yoğun |
|-----------------|-------|-------|-------|
| **Akıcı**       | 32545 | 66    | 5930  |
| **Kilit**       | 133   | 43925 | 8310  |
| **Yoğun**       | 5432  | 8090  | 77138 |

---

## 4. En İyi Model Seçim Gerekçesi
Model performans değerlendirmeleri sonucunda regresyon görevinde ve sınıflandırma görevinde en yüksek performansı sergileyen modeller **XGBoost** ve **XGBoost_clf** olarak belirlenmiştir. 
- Regresyon modelinde XGBoost, Test R² değeri `0.8233` ile varyansın büyük kısmını başarılı şekilde açıklayarak en düşük Test RMSE (`0.1017`) oranını elde etmiştir.
- Sınıflandırma görevinde XGBoost Classifier, `0.8460` Test Ağırlıklı F1 skoruyla sınıf dengesizliğine en iyi uyum sağlayan ve tüm sınıflarda istikrarlı çalışan (per-class f1 > 0.84) model olmuştur.

---

## 5. En Önemli Özellikler (Feature Importance) - İlk 15

| Sıra | Özellik (Feature) | Önem Skoru |
|------|-------------------|------------|
| 1 | `cs_lag_1h` | 0.5451 |
| 2 | `is_night` | 0.1444 |
| 3 | `cs_lag_24h` | 0.0461 |
| 4 | `is_weekend` | 0.0398 |
| 5 | `hour` | 0.0342 |
| 6 | `hour_sin` | 0.0313 |
| 7 | `hour_cos` | 0.0200 |
| 8 | `day_of_week` | 0.0133 |
| 9 | `month` | 0.0108 |
| 10 | `dow_sin` | 0.0103 |
| 11 | `dow_cos` | 0.0099 |
| 12 | `is_holiday` | 0.0089 |
| 13 | `week_of_year` | 0.0076 |
| 14 | `is_rush_hour` | 0.0071 |
| 15 | `rolling_mean_3h` | 0.0060 |

Model ağırlıklı olarak son 1 saatteki (`cs_lag_1h`) ve son 1 gün aynı saatteki (`cs_lag_24h`) geçmiş yoğunluk durumuna dayanmaktadır. Ek olarak günün saati ve haftasonu gibi zamansal/döngüsel özellikler de yüksek önem taşımaktadır.
