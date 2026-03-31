# HemaCheck: Kan Hücresi Anomali Tespiti
## AI-Driven Blood Cell Analysis Project - Implementation Plan

---

## Proje Özeti

**HemaCheck**, kan hücrelerinin morfolojik özelliklerini kullanarak **anormal hücre tespiti** yapan bir makine öğrenmesi projesidir. CytoDiffusion benchmark verileri temel alınarak, tıbbi görüntülemede AI uygulamalarını gösteren end-to-end bir veri bilimi projesidir.

### Hedefler
- Kan hücrelerinde anomali tespiti (binary: Normal/Anormal)
- Morfolojik özelliklerden sınıflandırma modeli
- Explainable AI ile klinik yorumlanabilirlik
- Gerçek dünya veri seti benchmark karşılaştırması

### Veri Seti Özellikleri
- **5882 hücre örneği**
- **38 morfolojik özellik**: çap, alan, yoğunluk, şekil, renk değerleri
- **18 farklı hücre tipi**: Normal (7) + Anormal (11)
- **Hastalık kategorileri**: Lösemi, Anemi, Enfeksiyon, Artefakt

---

## Implementation Plan - 4 Aşama

### Aşama 1: EDA ve Veri Hazırlama (2-3 saat)

| Görev | Açıklama | Çıktı |
|-------|----------|-------|
| 1.1 | Veri yükleme ve temizlik | `notebooks/01_eda.ipynb` |
| 1.2 | Anomali dağılımı analizi | İmbalance kontrolü (Normal:~60%, Anormal:~40%) |
| 1.3 | Hücre tipi görselleştirme | Pie/bar chart'lar |
| 1.4 | Özellik korelasyon matrisi | Heatmap analizi |
| 1.5 | PCA/t-SNE boyut indirgeme | 2D/3D scatter plot'lar |

**Key Findings Dokümantasyonu:**
- Blast_Cell ve Prolymphocyte → Lösemi belirteçleri
- Hypersegmented_Neutrophil → B12 eksikliği göstergesi
- Schistocyte/Spherocyte → Anemi indikatörleri

### Aşama 2: Feature Engineering (2 saat)

| Görev | Açıklama |
|-------|----------|
| 2.1 | Kategorik encoding (cell_type, disease_category) |
| 2.2 | Sayısal scaling (StandardScaler) |
| 2.3 | Yeni özellik türetimi: `shape_index = circularity × (1 - eccentricity)` |
| 2.4 | Feature selection: SelectKBest veya RFE |
| 2.5 | Train/validation/test split (70/15/15) |

### Aşama 3: Model Geliştirme (3-4 saat)

| Model | Kullanım Amacı | Metrik |
|-------|---------------|--------|
| **Random Forest** | Baseline + feature importance | F1-score, AUC |
| **XGBoost/LightGBM** | Ana model | Precision, Recall |
| **Logistic Regression** | Interpretability | ROC-AUC |

**Model Optimizasyonu:**
- Cross-validation (5-fold)
- Hyperparameter tuning (Optuna/GridSearch)
- SMOTE/ADASYN ile imbalance handling

**Hedef Metrikler:**
- ROC-AUC > 0.95 (CytoDiffusion benchmark: 0.99)
- F1-score > 0.90
- Recall (Anomaly) > 0.85

### Aşama 4: Explainable AI ve Raporlama (3 saat)

| Görev | Açıklama | Çıktı |
|-------|----------|-------|
| 4.1 | SHAP değer analizi | `shap.summary_plot()` |
| 4.2 | Feature importance görselleştirme | Horizontal bar chart |
| 4.3 | Confusion matrix | Normalized heatmap |
| 4.4 | ROC/PR curve'leri | Comparison plot |
| 4.5 | GitHub README + LinkedIn post içeriği | Markdown dosyaları |

---

## Proje Yapısı

```
HemaCheck/
├── data/
│   ├── raw/                    # Orijinal CSV'ler
│   └── processed/              # Temizlenmiş veriler
├── notebooks/
│   ├── 01_eda.ipynb           # Keşifsel analiz
│   ├── 02_feature_eng.ipynb   # Özellik mühendisliği
│   ├── 03_modeling.ipynb      # Model eğitimi
│   └── 04_interpretation.ipynb # SHAP analizi
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
├── models/                     # Kaydedilmiş modeller (.pkl)
├── reports/
│   ├── figures/               # Tüm görselleştirmeler
│   ├── linkedin_post.md       # LinkedIn içeriği
│   └── github_summary.md      # README içeriği
└── requirements.txt
```

---

## Zaman Çizelgesi (Toplam: ~10-12 saat)

| Gün | Aşama | Süre |
|-----|-------|------|
| 1 | EDA + Feature Engineering | 4-5 saat |
| 2 | Model geliştirme + tuning | 4 saat |
| 3 | Interpretation + Dokümantasyon | 2-3 saat |

---

## LinkedIn Post Başlıkları (Öneriler)

1. **"AI in Hematology: Building a Blood Cell Anomaly Detection System"**
   - ROC-AUC: 0.97 achievement
   - Feature importance insights
   - Clinical impact

2. **"From Raw Data to Clinical Insights: My End-to-End ML Pipeline"**
   - Technical architecture
   - Key challenges & solutions

3. **"Explainable AI in Medical Diagnosis"**
   - SHAP analysis results
   - Top predictive features
   - Trust in AI systems

---

## GitHub README Bölümleri

1. **Project Overview** - 2-3 cümle özet
2. **Dataset Description** - 18 hücre tipi tablosu
3. **Key Results** - Metrikler + ROC curve
4. **Technical Approach** - Pipeline diagram
5. **Feature Importance** - SHAP plot
6. **Installation & Usage** - Quick start
7. **Future Work** - Next steps

---

## Riskler ve Mitigasyon

| Risk | Mitigasyon |
|------|------------|
| Imbalanced data | SMOTE + class weights |
| Overfitting | Cross-validation + regularization |
| Feature leakage | Train/test strict separation |
| Low interpretability | SHAP + simple models as baseline |

---

## Success Criteria

- [ ] ROC-AUC ≥ 0.95
- [ ] 5+ görselleştirme (publication-ready)
- [ ] SHAP interpretasyonu
- [ ] GitHub repo + README
- [ ] 1 LinkedIn post (technical)

---

## Teknik Stack

- **Python**: pandas, numpy, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn, plotly
- **Interpretability**: shap
- **Environment**: Jupyter notebooks

---

*Plan hazır - başlamaya hazır mısın? İlk olarak `notebooks/01_eda.ipynb` ile başlayabiliriz.*
