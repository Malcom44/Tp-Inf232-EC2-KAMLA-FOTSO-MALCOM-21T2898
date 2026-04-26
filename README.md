# 🌾 AgricolApp — TP INF232 EC2
### Application Streamlit · Analyse de données agricoles

---

## 🚀 Lancer en local

```bash
# 1. Installer les dépendances
pip install streamlit pandas numpy scikit-learn matplotlib

# 2. Lancer l'application
streamlit run app.py

# 3. Ouvrir http://localhost:8501
```

---

## 🌐 Déployer sur Streamlit Cloud (gratuit)

1. **Créer un compte** sur [share.streamlit.io](https://share.streamlit.io)
2. **Pousser le projet sur GitHub** :
   ```bash
   git init
   git add .
   git commit -m "AgricolApp INF232"
   git remote add origin https://github.com/TON_USERNAME/agricolapp.git
   git push -u origin main
   ```
3. Sur Streamlit Cloud → **New app** → choisir le dépôt → `app.py`
4. Cliquer **Deploy** → lien public en 2 minutes ✓

---

## 📁 Structure

```
agricolapp_streamlit/
├── app.py            ← Application principale Streamlit
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
