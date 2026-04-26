# 🌾 AgricolApp — TP INF232 EC2
### Application Streamlit · Analyse de données agricoles

---

## 🚀 Lancer en local

```bash
# 1. Installer les dépendances
pip install streamlit pandas numpy scikit-learn matplotlib

# 2. Lancer l'application
streamlit run KAMLA_FOTSO_MALCOM_21T2898.py

# 3. Ouvrir http://localhost:8501
---

## 📁 Structure

```
agricolapp_streamlit/
├── KAMLA_FOTSO_MALCOM_21T2898.py            ← Application principale Streamlit
├── requirements.txt  ← Dépendances Python
└── README.md
```

---

## 🔬 Modules & bibliothèques

| Module | Bibliothèque |
|--------|-------------|
| Collecte données | Streamlit session_state |
| Statistiques descriptives | pandas (describe, quantile…) |
| Régression simple | numpy (MCO manuel) |
| Régression multiple | numpy.linalg.lstsq |
| ACP | sklearn.decomposition.PCA |
| k-NN | sklearn.neighbors.KNeighborsClassifier |
| K-Means | sklearn.cluster.KMeans |
| Visualisations | matplotlib |

---

*TP INF232 EC2 — Analyse de données agricoles*
