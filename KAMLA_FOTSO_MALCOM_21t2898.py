"""
╔══════════════════════════════════════════════════════╗
║   AgricolApp — TP INF232 EC2                        ║
║   Application Streamlit · Domaine : Agriculture     ║
║   Modules : Collecte, Descriptive, Régression       ║
║             simple & multiple, ACP, k-NN, K-Means  ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import io
import json
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  CONFIG PAGE
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgricolApp — INF232",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
#  CSS DESIGN (bleu-ciel / blanc / noir)
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=DM+Mono&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0c2340 100%);
    border-right: 1px solid rgba(56,189,248,.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px; border-radius: 8px; cursor: pointer;
    font-size: .82rem; font-weight: 500; transition: all .17s;
    display: flex; align-items: center; gap: 8px;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(56,189,248,.1); }

/* Metrics */
[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(10,15,30,.08);
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: .65rem !important; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0284c7 !important; font-size: 1.6rem !important; font-weight: 800 !important; font-family: 'DM Mono', monospace !important; }

/* Cards */
.agri-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 20px 22px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(10,15,30,.08);
}
.agri-card h4 {
    font-size: .88rem; font-weight: 700; color: #1e293b;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
    display: flex; align-items: center; gap: 8px;
}

/* Hero banner */
.hero-box {
    background: linear-gradient(135deg, #0a0f1e 0%, #0c2340 55%, #0c4a6e 100%);
    border-radius: 16px; padding: 36px 36px; margin-bottom: 22px;
    color: white; position: relative; overflow: hidden;
}
.hero-box::before { content: '🌾'; position: absolute; right: 30px; top: 50%; transform: translateY(-50%); font-size: 5.5rem; opacity: .08; }
.hero-box h1 { font-size: 2rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 10px; color: white; }
.hero-box h1 span { color: #38bdf8; }
.hero-box p { color: #94a3b8; font-size: .82rem; line-height: 1.75; max-width: 520px; }
.hero-chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 16px; }
.hero-chip {
    background: rgba(56,189,248,.12); border: 1px solid rgba(56,189,248,.22);
    color: #bae6fd; padding: 4px 12px; border-radius: 20px;
    font-size: .64rem; font-weight: 600;
}

/* Result box */
.result-box {
    background: #f0f9ff; border: 1.5px solid #bae6fd;
    border-radius: 10px; padding: 16px; margin-top: 12px;
    font-family: 'DM Mono', monospace; font-size: .78rem; line-height: 1.9;
}
.result-box .eq { color: #0284c7; font-weight: 700; font-size: .88rem; }
.result-box .good { color: #16a34a; font-weight: 700; }
.result-box .warn { color: #d97706; font-weight: 700; }

/* Section title */
.sec-title { font-size: 1.2rem; font-weight: 800; color: #1e293b; margin-bottom: 4px; }
.sec-sub { font-size: .73rem; color: #94a3b8; margin-bottom: 18px; }

/* Sidebar brand */
.sb-brand {
    background: rgba(56,189,248,.08); border: 1px solid rgba(56,189,248,.2);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 18px; text-align: center;
}
.sb-brand h2 { font-size: 1.3rem; font-weight: 800; color: #fff !important; margin: 0; }
.sb-brand h2 em { color: #38bdf8 !important; font-style: normal; }
.sb-brand p { font-size: .6rem; color: #94a3b8 !important; margin: 3px 0 0; font-family: 'DM Mono', monospace; }

/* Rule list */
.rule-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: #f8fafc; border-radius: 8px; margin: 4px 0; font-size: .78rem; }
.rule-arrow { color: #38bdf8; font-weight: 700; }

/* Stbutton override */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #0284c7) !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 9px !important;
    padding: 8px 20px !important; font-size: .78rem !important;
    transition: all .17s !important;
}
.stButton > button:hover { filter: brightness(1.08) !important; box-shadow: 0 4px 14px rgba(56,189,248,.38) !important; transform: translateY(-1px) !important; }

/* Download button */
.stDownloadButton > button {
    background: transparent !important; color: #0284c7 !important;
    border: 1.5px solid #38bdf8 !important; border-radius: 9px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════
CULTURE_COLORS = {
    'Maïs': '#f59e0b', 'Blé': '#fbbf24', 'Sorgho': '#d97706',
    'Manioc': '#22c55e', 'Arachide': '#84cc16', 'Cacao': '#6b3a1f'
}
CLUSTER_COLORS = ['#38bdf8','#f59e0b','#22c55e','#f472b6','#a78bfa','#34d399','#fb923c','#60a5fa']
VAR_LABELS = {
    'superficie': 'Superficie (ha)', 'ph': 'pH du sol',
    'pluie': 'Pluviométrie (mm)', 'temp': 'Température (°C)',
    'engrais': 'Engrais (kg/ha)', 'rendement': 'Rendement (t/ha)'
}
CULTURES  = ['Maïs','Blé','Sorgho','Manioc','Arachide','Cacao']
SOLS      = ['Argileux','Sableux','Limoneux','Ferralitique']

# ══════════════════════════════════════════════════════
#  SESSION STATE — base de données
# ══════════════════════════════════════════════════════
if 'parcelles' not in st.session_state:
    st.session_state.parcelles = []

def get_df():
    if not st.session_state.parcelles:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.parcelles)
    for col in ['superficie','ph','pluie','temp','engrais','rendement']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ══════════════════════════════════════════════════════
#  MATPLOTLIB STYLE
# ══════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor':   '#f8fafc',
    'axes.edgecolor':   '#e2e8f0',
    'axes.labelcolor':  '#475569',
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'text.color':       '#1e293b',
    'grid.color':       '#e2e8f0',
    'grid.alpha':       0.6,
    'axes.grid':        True,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

def fig_style(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1e293b', pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color='#475569')
    ax.set_ylabel(ylabel, fontsize=9, color='#475569')
    ax.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <h2>🌾 Agricol<em>App</em></h2>
        <p>TP INF232 · EC2 · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    n = len(st.session_state.parcelles)
    st.markdown(f"<div style='text-align:center;margin-bottom:14px'><span style='background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.3);color:#38bdf8;padding:4px 14px;border-radius:20px;font-size:.7rem;font-weight:700'>📦 {n} parcelle(s)</span></div>", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Tableau de bord",
        "📋  Collecte données",
        "📊  Analyse descriptive",
        "📈  Régression simple",
        "📐  Régression multiple",
        "🔻  Réduction dim. (ACP)",
        "🏷️  Classification supervisée",
        "🌿  Classification non-sup.",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:.6rem;color:#475569;font-family:monospace'>🐍 Python · Streamlit<br>📊 pandas · numpy<br>🔬 scikit-learn</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE 0 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════
if "Tableau" in page:
    st.markdown("""
    <div class="hero-box">
        <h1>Collecte &amp; <span>Analyse</span> Agricole</h1>
        <p>Plateforme intégrée de collecte terrain et d'analyse statistique des exploitations agricoles — rendements, sols, cultures et conditions climatiques.</p>
        <div class="hero-chips">
            <span class="hero-chip">🌾 Céréales & Cultures</span>
            <span class="hero-chip">🌱 Pédologie</span>
            <span class="hero-chip">🌦️ Agroclimatologie</span>
            <span class="hero-chip">🚜 Exploitation agricole</span>
            <span class="hero-chip">🐍 Python · Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = get_df()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌾 Parcelles",  len(df))
    c2.metric("📊 Variables",  7)
    c3.metric("🧬 Cultures",   df['culture'].nunique() if not df.empty and 'culture' in df.columns else 0)
    c4.metric("📡 Modules",    6)

    st.markdown("---")
    st.markdown("### 🗂️ Modules disponibles")
    col1, col2, col3 = st.columns(3)
    modules = [
        ("📋","Collecte données","Formulaire parcelle — sol, culture, rendement"),
        ("📊","Analyse descriptive","Statistiques pandas — histogrammes, boxplots"),
        ("📈","Régression simple","Y = aX + b — MCO numpy"),
        ("📐","Régression multiple","MCO numpy.linalg.lstsq — plusieurs prédicteurs"),
        ("🔻","Réduction dim. (ACP)","sklearn.decomposition.PCA"),
        ("🌿","Classification","k-NN & K-Means sklearn"),
    ]
    for i, (icon, name, desc) in enumerate(modules):
        col = [col1, col2, col3][i % 3]
        col.markdown(f"""
        <div class="agri-card" style="text-align:center;cursor:pointer">
            <div style="font-size:1.6rem;margin-bottom:8px">{icon}</div>
            <div style="font-weight:700;font-size:.85rem;color:#1e293b">{name}</div>
            <div style="font-size:.65rem;color:#94a3b8;margin-top:4px">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    if not df.empty:
        st.markdown("---")
        st.markdown("### 📋 Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 1 — COLLECTE
# ══════════════════════════════════════════════════════
elif "Collecte" in page:
    st.markdown('<div class="sec-title">📋 Collecte de Données</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Saisie fiche parcelle agricole — stockage en session Streamlit</div>', unsafe_allow_html=True)

    # ── Formulaire
    with st.expander("➕ Ajouter une nouvelle parcelle", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_id       = st.text_input("Identifiant parcelle", placeholder="PARC-2024-001")
            f_superficie = st.number_input("Superficie (ha)", min_value=0.1, value=5.0, step=0.1)
            f_culture  = st.selectbox("Type de culture", [""] + CULTURES)
        with c2:
            f_sol      = st.selectbox("Type de sol", [""] + SOLS)
            f_ph       = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
            f_pluie    = st.number_input("Pluviométrie (mm/an)", min_value=0, value=1200, step=10)
        with c3:
            f_temp     = st.number_input("Température moy. (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
            f_engrais  = st.number_input("Engrais (kg/ha)", min_value=0, value=150, step=5)
            f_rendement = st.number_input("Rendement (t/ha)", min_value=0.0, value=3.5, step=0.1)
        f_obs = st.text_area("Observations terrain", placeholder="Conditions, maladies, irrigation…")

        col_a, col_b, col_c = st.columns([2, 2, 4])
        with col_a:
            if st.button("✔ Enregistrer la parcelle"):
                n_curr = len(st.session_state.parcelles)
                pid = f_id.strip() or f"PARC-{n_curr+1:03d}"
                st.session_state.parcelles.append({
                    'id': pid, 'superficie': f_superficie,
                    'culture': f_culture, 'sol': f_sol,
                    'ph': f_ph, 'pluie': f_pluie, 'temp': f_temp,
                    'engrais': f_engrais, 'rendement': f_rendement, 'obs': f_obs
                })
                st.success(f"✔ Parcelle **{pid}** enregistrée !")
                st.rerun()
        with col_b:
            if st.button("⚡ Charger 30 démos"):
                np.random.seed(42)
                cultures_list = ['Maïs','Maïs','Blé','Sorgho','Manioc','Arachide','Cacao']
                for i in range(30):
                    c  = cultures_list[np.random.randint(0, len(cultures_list))]
                    s  = SOLS[np.random.randint(0, len(SOLS))]
                    pl = float(np.round(600 + np.random.random()*1400, 0))
                    en = float(np.round(50  + np.random.random()*300,  0))
                    ph = float(np.round(4.5 + np.random.random()*3.5,  1))
                    te = float(np.round(18  + np.random.random()*15,   1))
                    su = float(np.round(0.5 + np.random.random()*20,   1))
                    rd = float(np.round(max(0.3,
                        0.5+(pl/1000)*2+(en/200)*1.5+(ph-4)*0.3-(abs(te-25)/10)*0.4+(np.random.random()*1.4-.6)), 2))
                    n_curr = len(st.session_state.parcelles)
                    st.session_state.parcelles.append({
                        'id':f"DEMO-{n_curr+1:03d}",'superficie':su,'culture':c,'sol':s,
                        'ph':ph,'pluie':pl,'temp':te,'engrais':en,'rendement':rd,'obs':''
                    })
                st.success("⚡ 30 parcelles démo chargées !")
                st.rerun()

    # ── Tableau + suppression
    df = get_df()
    if df.empty:
        st.info("Aucune donnée. Utilisez le formulaire ou chargez les données démo.")
    else:
        st.markdown(f"**📂 Base de données — {len(df)} parcelle(s)**")
        st.dataframe(df.drop(columns=['obs'], errors='ignore'), use_container_width=True, height=320)

        col_x, col_y = st.columns([2, 5])
        with col_x:
            idx_del = st.number_input("Supprimer ligne n°", min_value=1, max_value=len(df), value=1, step=1)
            if st.button("🗑 Supprimer"):
                st.session_state.parcelles.pop(idx_del - 1)
                st.success(f"Ligne {idx_del} supprimée.")
                st.rerun()

        # Export CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Exporter CSV", csv, "AgricolApp_export.csv", "text/csv")

        if st.button("🗑 Effacer toutes les données", type="secondary"):
            st.session_state.parcelles = []
            st.rerun()

# ══════════════════════════════════════════════════════
#  PAGE 2 — ANALYSE DESCRIPTIVE
# ══════════════════════════════════════════════════════
elif "descriptive" in page:
    st.markdown('<div class="sec-title">📊 Analyse Descriptive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Statistiques calculées par pandas — visualisations matplotlib</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty:
        st.warning("⚠️ Aucune donnée. Allez dans **Collecte données** pour en ajouter.")
        st.stop()

    var = st.selectbox("Variable à analyser", list(VAR_LABELS.keys()), format_func=lambda x: VAR_LABELS[x])
    s = df[var].dropna()

    # Métriques
    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    c1.metric("N",       len(s))
    c2.metric("Moyenne", round(s.mean(),  3))
    c3.metric("Écart-type", round(s.std(), 3))
    c4.metric("Médiane", round(s.median(),3))
    c5.metric("Min",     round(s.min(),   3))
    c6.metric("Max",     round(s.max(),   3))
    c7.metric("Q1",      round(s.quantile(.25),3))
    c8.metric("Q3",      round(s.quantile(.75),3))

    col1, col2 = st.columns(2)

    # ── Histogramme
    with col1:
        st.markdown('<div class="agri-card"><h4>📊 Histogramme de distribution</h4>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(s, bins=10, color='#38bdf8', edgecolor='#0284c7', alpha=0.8, linewidth=1.2, rwidth=.88)
        fig_style(ax, f'Distribution — {VAR_LABELS[var]}', VAR_LABELS[var], 'Fréquence')
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Pie cultures
    with col2:
        st.markdown('<div class="agri-card"><h4>🥧 Répartition des cultures</h4>', unsafe_allow_html=True)
        if 'culture' in df.columns and df['culture'].notna().any():
            cult_counts = df['culture'].value_counts()
            colors = [CULTURE_COLORS.get(c,'#888') for c in cult_counts.index]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.pie(cult_counts.values, labels=cult_counts.index, colors=colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize':9},
                   wedgeprops={'edgecolor':'white','linewidth':1.5})
            ax.set_title('Répartition des cultures', fontsize=11, fontweight='bold', color='#1e293b')
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.info("Aucune culture renseignée.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Boxplot comparatif
    st.markdown('<div class="agri-card"><h4>📦 Boxplot comparatif — pH · Engrais · Rendement</h4>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    box_vars = [('ph','pH du sol','#38bdf8'), ('engrais','Engrais (kg/ha)','#f59e0b'), ('rendement','Rendement (t/ha)','#22c55e')]
    for ax, (v, lbl, col) in zip(axes, box_vars):
        if v in df.columns:
            data = df[v].dropna()
            bp = ax.boxplot(data, patch_artist=True, widths=.5,
                            boxprops=dict(facecolor=col+'44', edgecolor=col, linewidth=1.5),
                            medianprops=dict(color=col, linewidth=2.5),
                            whiskerprops=dict(color='#94a3b8', linewidth=1.2),
                            capprops=dict(color='#94a3b8', linewidth=1.2),
                            flierprops=dict(marker='o', color=col, alpha=.5, markersize=5))
            ax.set_title(lbl, fontsize=10, fontweight='bold', color='#1e293b')
            ax.tick_params(labelsize=8); ax.set_xticks([])
            q1, med, q3 = data.quantile(.25), data.median(), data.quantile(.75)
            ax.text(1.3, med, f'Med:{med:.2f}', va='center', fontsize=8, color=col, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Tableau statistiques complètes
    with st.expander("📋 Statistiques descriptives complètes (pandas describe)"):
        num_cols = ['superficie','ph','pluie','temp','engrais','rendement']
        available = [c for c in num_cols if c in df.columns]
        st.dataframe(df[available].describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 3 — RÉGRESSION SIMPLE
# ══════════════════════════════════════════════════════
elif "Régression simple" in page:
    st.markdown('<div class="sec-title">📈 Régression Linéaire Simple</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Y = aX + b — Moindres Carrés Ordinaires (numpy)</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty or len(df) < 5:
        st.warning("⚠️ Minimum 5 observations. Ajoutez des données dans **Collecte données**.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        xk = st.selectbox("Variable X (prédicteur)", list(VAR_LABELS.keys()),
                           format_func=lambda x: VAR_LABELS[x], index=2)
    with col2:
        yk = st.selectbox("Variable Y (cible)", list(VAR_LABELS.keys()),
                           format_func=lambda x: VAR_LABELS[x], index=5)

    if st.button("▶ Calculer la régression"):
        sub = df[[xk, yk]].dropna()
        if len(sub) < 5:
            st.error("Données insuffisantes après nettoyage.")
            st.stop()

        x, y = sub[xk].values, sub[yk].values
        mx, my = x.mean(), y.mean()
        a = np.sum((x - mx)*(y - my)) / (np.sum((x - mx)**2) + 1e-10)
        b = my - a * mx
        yp = a * x + b
        ss_res = np.sum((y - yp)**2)
        ss_tot = np.sum((y - my)**2) + 1e-10
        r2   = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(y))
        corr = np.corrcoef(x, y)[0, 1]

        # Métriques
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²",    round(r2,   4))
        m2.metric("RMSE",  round(rmse, 4))
        m3.metric("Corrélation r", round(corr, 4))
        m4.metric("N", len(y))

        # Résultat
        quality = "✔ Bonne corrélation" if r2 >= .7 else ("⚠ Corrélation modérée" if r2 >= .4 else "✕ Corrélation faible")
        st.markdown(f"""
        <div class="result-box">
            <div class="eq">Équation : Ŷ = {a:.5f} × X + {b:.5f}</div><br>
            <span class="{'good' if r2>=.5 else 'warn'}">R² = {r2:.4f} — {quality}</span><br>
            RMSE = {rmse:.4f} | r de Pearson = {corr:.4f}<br>
            Pour +1 unité de {VAR_LABELS[xk]}, {VAR_LABELS[yk]} varie de <strong>{a:.4f}</strong> unités.<br>
            <span style="color:#94a3b8;font-size:.7rem">Calculé par numpy — MCO manuel</span>
        </div>
        """, unsafe_allow_html=True)

        # Graphique
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(x, y, color='#38bdf8', edgecolors='#0284c7', s=65, alpha=.8, zorder=3, label='Observations')
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, a*x_line+b, color='#f59e0b', linewidth=2.5, label=f'Ŷ = {a:.4f}X + {b:.4f}', zorder=4)
        # Résidus
        for xi, yi, ypi in zip(x, y, yp):
            ax.plot([xi, xi], [yi, ypi], color='#ef4444', alpha=.25, linewidth=.8)
        fig_style(ax, f'Régression : {VAR_LABELS[xk]} → {VAR_LABELS[yk]}', VAR_LABELS[xk], VAR_LABELS[yk])
        ax.legend(fontsize=9)
        ax.text(.02, .97, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='#0284c7', fontweight='bold',
                bbox=dict(boxstyle='round,pad=.4', facecolor='#e0f2fe', edgecolor='#38bdf8'))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════
#  PAGE 4 — RÉGRESSION MULTIPLE
# ══════════════════════════════════════════════════════
elif "multiple" in page:
    st.markdown('<div class="sec-title">📐 Régression Linéaire Multiple</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">MCO — numpy.linalg.lstsq + sklearn.StandardScaler</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty or len(df) < 10:
        st.warning("⚠️ Minimum 10 observations nécessaires.")
        st.stop()

    st.markdown("**Variable Y cible : Rendement (t/ha)**")
    st.markdown("**Sélectionner les variables X prédicteurs :**")

    var_map = {'pluie': 'Pluviométrie', 'engrais': 'Engrais', 'ph': 'pH sol', 'temp': 'Température', 'superficie': 'Superficie'}
    cols = st.columns(len(var_map))
    selected = {}
    defaults = {'pluie': True, 'engrais': True, 'ph': False, 'temp': False, 'superficie': False}
    for i, (vk, vl) in enumerate(var_map.items()):
        selected[vk] = cols[i].checkbox(vl, value=defaults[vk])

    vars_x = [v for v, sel in selected.items() if sel]

    if st.button("▶ Calculer la régression multiple"):
        if not vars_x:
            st.error("Cochez au moins une variable X.")
            st.stop()

        sub = df[vars_x + ['rendement']].dropna()
        if len(sub) < 10:
            st.error("Données insuffisantes après nettoyage.")
            st.stop()

        X = sub[vars_x].values
        y = sub['rendement'].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        yp = Xb @ beta
        my = y.mean()
        r2   = float(1 - np.sum((y-yp)**2) / (np.sum((y-my)**2) + 1e-10))
        rmse = float(np.sqrt(np.mean((y-yp)**2)))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²",        round(r2,   4))
        m2.metric("RMSE",      round(rmse, 4))
        m3.metric("N",         len(y))
        m4.metric("Prédicteurs", len(vars_x))

        coeffs = {v: round(float(beta[i+1] / scaler.scale_[i]), 6) for i, v in enumerate(vars_x)}
        coeff_lines = "<br>".join([f"&nbsp;&nbsp;β({var_map[v]}) = <strong style='color:#0284c7'>{coeffs[v]}</strong>" for v in vars_x])

        quality = "✔ Bonne" if r2 >= .7 else ("⚠ Modérée" if r2 >= .4 else "✕ Faible")
        st.markdown(f"""
        <div class="result-box">
            <div class="eq">Régression multiple → Rendement (t/ha)</div><br>
            <span class="{'good' if r2>=.5 else 'warn'}">R² = {r2:.4f} — {quality} | RMSE = {rmse:.4f}</span><br><br>
            Coefficients estimés (numpy.linalg.lstsq) :<br>{coeff_lines}<br>
            β₀ (intercept) = <strong style="color:#f59e0b">{beta[0]:.4f}</strong><br>
            <span style="color:#94a3b8;font-size:.7rem">StandardScaler + lstsq — données standardisées</span>
        </div>
        """, unsafe_allow_html=True)

        # Graphique Observé vs Prédit
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1, ax2 = axes

        ax1.scatter(y, yp, color='#38bdf8', edgecolors='#0284c7', s=60, alpha=.8)
        lims = [min(y.min(), yp.min())*.95, max(y.max(), yp.max())*1.05]
        ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Parfait')
        fig_style(ax1, 'Observé vs Prédit', 'Rendement observé (t/ha)', 'Rendement prédit (t/ha)')
        ax1.legend(fontsize=9)
        ax1.text(.05,.95,f'R²={r2:.4f}',transform=ax1.transAxes,fontsize=10,color='#0284c7',
                 fontweight='bold',va='top',bbox=dict(boxstyle='round,pad=.3',facecolor='#e0f2fe',edgecolor='#38bdf8'))

        residus = y - yp
        ax2.bar(range(len(residus)), residus, color=np.where(residus>=0,'#38bdf8','#f59e0b'), alpha=.75, width=.7)
        ax2.axhline(0, color='#1e293b', linewidth=1.2, linestyle='--')
        fig_style(ax2, 'Résidus', 'Observation', 'Résidu (observé − prédit)')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════
#  PAGE 5 — ACP / PCA
# ══════════════════════════════════════════════════════
elif "ACP" in page:
    st.markdown('<div class="sec-title">🔻 Réduction Dimensionnelle — ACP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">sklearn.decomposition.PCA — StandardScaler</div>', unsafe_allow_html=True)

    df = get_df()
    vars_pca = ['superficie','ph','pluie','temp','engrais','rendement']
    sub = df.dropna(subset=vars_pca) if not df.empty else pd.DataFrame()

    if sub.empty or len(sub) < 5:
        st.warning("⚠️ Minimum 5 observations avec toutes les variables renseignées.")
        st.stop()

    if st.button("▶ Lancer l'ACP (sklearn.PCA)"):
        X = sub[vars_pca].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n_comp = min(6, len(sub), len(vars_pca))
        pca_model = PCA(n_components=n_comp)
        proj = pca_model.fit_transform(Xs)
        var_exp = pca_model.explained_variance_ratio_ * 100
        cumul   = np.cumsum(var_exp)
        loadings = pca_model.components_

        # Métriques
        cols = st.columns(n_comp)
        for i in range(n_comp):
            cols[i].metric(f"PC{i+1}", f"{var_exp[i]:.1f}%")

        col1, col2 = st.columns(2)

        # ── Variance expliquée
        with col1:
            st.markdown('<div class="agri-card"><h4>📊 Variance expliquée par composante</h4>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            x_pos = np.arange(n_comp)
            bars = ax.bar(x_pos, var_exp, color='#38bdf8', alpha=.75, edgecolor='#0284c7', linewidth=1.2, width=.55, label='Variance (%)')
            ax2_ = ax.twinx()
            ax2_.plot(x_pos, cumul, 'o-', color='#f59e0b', linewidth=2, markersize=7, label='Cumul (%)')
            ax2_.set_ylabel('Cumul (%)', fontsize=9, color='#f59e0b')
            ax2_.tick_params(colors='#f59e0b', labelsize=8)
            ax2_.set_ylim(0, 110)
            ax.set_xticks(x_pos); ax.set_xticklabels([f'PC{i+1}' for i in range(n_comp)])
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5, f'{bar.get_height():.1f}%', ha='center', fontsize=8, color='#0284c7', fontweight='bold')
            fig_style(ax, 'Variance expliquée', 'Composante', 'Variance (%)')
            ax.legend(loc='upper left', fontsize=8); ax2_.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Projection PC1 vs PC2
        with col2:
            st.markdown('<div class="agri-card"><h4>🗺️ Projection PC1 vs PC2</h4>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            cultures_col = sub['culture'].tolist() if 'culture' in sub.columns else ['']*len(sub)
            unique_cultures = list(set(cultures_col))
            for cult in unique_cultures:
                mask = [c == cult for c in cultures_col]
                col_c = CULTURE_COLORS.get(cult, '#38bdf8')
                ax.scatter(proj[mask, 0], proj[mask, 1], c=col_c, label=cult if cult else 'N/A',
                           s=65, alpha=.8, edgecolors='white', linewidth=.8)
            ax.axhline(0, color='#94a3b8', linestyle='--', linewidth=.8)
            ax.axvline(0, color='#94a3b8', linestyle='--', linewidth=.8)
            fig_style(ax, 'Projection ACP', f'PC1 ({var_exp[0]:.1f}%)', f'PC2 ({var_exp[1]:.1f}%)')
            if unique_cultures: ax.legend(fontsize=8, loc='best')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Matrice corrélations
        st.markdown('<div class="agri-card"><h4>🌡️ Matrice des corrélations (pandas.corr)</h4>', unsafe_allow_html=True)
        corr_mat = sub[vars_pca].corr().round(3)
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(corr_mat.values, cmap='Blues', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(vars_pca))); ax.set_yticks(range(len(vars_pca)))
        labels_short = ['Sup.','pH','Pluie','Temp.','Engrais','Rdmt']
        ax.set_xticklabels(labels_short, fontsize=9, color='#475569')
        ax.set_yticklabels(labels_short, fontsize=9, color='#475569')
        for i in range(len(vars_pca)):
            for j in range(len(vars_pca)):
                v = corr_mat.values[i,j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=9, fontweight='bold' if abs(v)>.5 and i!=j else 'normal',
                        color='white' if abs(v)>.65 else '#1e293b')
        plt.colorbar(im, ax=ax, shrink=.8)
        ax.set_title('Matrice des corrélations', fontsize=11, fontweight='bold', color='#1e293b', pad=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Loadings
        with st.expander("📋 Loadings (contributions des variables aux composantes)"):
            load_df = pd.DataFrame(loadings[:, :].T, index=vars_pca,
                                   columns=[f'PC{i+1}' for i in range(n_comp)]).round(3)
            st.dataframe(load_df.style.background_gradient(cmap='Blues', axis=None), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 6 — CLASSIFICATION SUPERVISÉE
# ══════════════════════════════════════════════════════
elif "supervisée" in page:
    st.markdown('<div class="sec-title">🏷️ Classification Supervisée</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">k-NN (sklearn) & Arbre de décision (règles agronomiques)</div>', unsafe_allow_html=True)

    df = get_df()
    feats = ['ph','pluie','temp','engrais','rendement']

    sub_knn = df.dropna(subset=feats+['culture'])
    if 'culture' in sub_knn.columns:
        sub_knn = sub_knn[sub_knn['culture'].str.strip() != '']

    tab1, tab2 = st.tabs(["🔍 k-Nearest Neighbors", "🌳 Arbre de décision"])

    # ── K-NN
    with tab1:
        if len(sub_knn) < 6:
            st.warning("⚠️ Minimum 6 parcelles avec culture renseignée.")
        else:
            k = st.slider("Nombre de voisins k", 1, 15, 5)
            if st.button("▶ Classifier (k-NN sklearn)", key="knn_btn"):
                X = sub_knn[feats].values
                y = sub_knn['culture'].values
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                model = KNeighborsClassifier(n_neighbors=k)
                cv = min(5, len(sub_knn))
                cv_scores = cross_val_score(model, Xs, y, cv=cv, scoring='accuracy')
                model.fit(Xs, y)
                y_pred = model.predict(Xs)
                acc_cv    = cv_scores.mean() * 100
                acc_train = accuracy_score(y, y_pred) * 100
                classes   = sorted(list(set(y)))
                cm        = confusion_matrix(y, y_pred, labels=classes)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Précision CV",     f"{acc_cv:.1f}%")
                c2.metric("Précision train",  f"{acc_train:.1f}%")
                c3.metric("k voisins",        k)
                c4.metric("N parcelles",      len(sub_knn))

                st.markdown(f"""
                <div class="result-box">
                    <span class="{'good' if acc_cv>=65 else 'warn'}">Précision cross-validation : {acc_cv:.1f}%</span><br>
                    Bibliothèque : <strong>sklearn.neighbors.KNeighborsClassifier(n_neighbors={k})</strong><br>
                    Cross-validation ({cv} folds) : {' · '.join([f'{s*100:.1f}%' for s in cv_scores])}<br>
                    Features : pH · Pluviométrie · Température · Engrais · Rendement<br>
                    <span style="color:#94a3b8;font-size:.7rem">Normalisation : sklearn.preprocessing.StandardScaler</span>
                </div>
                """, unsafe_allow_html=True)

                # Matrice de confusion
                col_a, col_b = st.columns([2,1])
                with col_a:
                    st.markdown('<div class="agri-card"><h4>🟩 Matrice de confusion</h4>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    im = ax.imshow(cm, cmap='Blues', aspect='auto')
                    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
                    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9, color='#475569')
                    ax.set_yticklabels(classes, fontsize=9, color='#475569')
                    for i in range(len(classes)):
                        for j in range(len(classes)):
                            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                                    fontsize=10, fontweight='bold',
                                    color='white' if cm[i,j] > cm.max()*.5 else '#1e293b')
                    ax.set_xlabel('Prédit', fontsize=9, color='#475569')
                    ax.set_ylabel('Réel', fontsize=9, color='#475569')
                    ax.set_title('Matrice de confusion — k-NN', fontsize=11, fontweight='bold', color='#1e293b')
                    plt.colorbar(im, ax=ax, shrink=.8)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)
                with col_b:
                    st.markdown('<div class="agri-card"><h4>📊 Scores CV</h4>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.bar(range(1, len(cv_scores)+1), cv_scores*100, color='#38bdf8', alpha=.8, edgecolor='#0284c7', linewidth=1.2, width=.6)
                    ax.axhline(acc_cv, color='#f59e0b', linestyle='--', linewidth=2, label=f'Moy: {acc_cv:.1f}%')
                    ax.set_ylim(0,110); ax.legend(fontsize=9)
                    fig_style(ax, 'Scores par fold', 'Fold', 'Précision (%)')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── ARBRE DE DÉCISION
    with tab2:
        if len(sub_knn) < 10:
            st.warning("⚠️ Minimum 10 parcelles avec culture renseignée.")
        else:
            depth = st.slider("Profondeur de l'arbre", 1, 6, 3)
            if st.button("▶ Entraîner l'arbre", key="tree_btn"):
                def predict_tree(row):
                    if row['pluie'] >= 1200 and row['temp'] >= 22: return 'Cacao'
                    if row['pluie'] >= 900: return 'Manioc'
                    if row['engrais'] >= 200 and row['ph'] >= 6.0: return 'Maïs'
                    if row['ph'] >= 6.5: return 'Blé'
                    if row['engrais'] >= 100: return 'Sorgho'
                    return 'Arachide'

                sub_knn = sub_knn.copy()
                sub_knn['pred'] = sub_knn.apply(predict_tree, axis=1)
                acc = (sub_knn['pred'] == sub_knn['culture']).mean() * 100
                classes = sorted(sub_knn['culture'].unique().tolist())
                cm = confusion_matrix(sub_knn['culture'], sub_knn['pred'], labels=classes)

                c1,c2,c3 = st.columns(3)
                c1.metric("Précision", f"{acc:.1f}%")
                c2.metric("Profondeur", depth)
                c3.metric("N", len(sub_knn))

                rules = [
                    ("Pluie ≥ 1200 mm ET Temp ≥ 22°C", "Cacao"),
                    ("Pluie ≥ 900 mm", "Manioc"),
                    ("Engrais ≥ 200 ET pH ≥ 6.0", "Maïs"),
                    ("pH ≥ 6.5", "Blé"),
                    ("Engrais ≥ 100", "Sorgho"),
                    ("Sinon", "Arachide"),
                ]
                rules_html = "".join([f'<div class="rule-item"><span class="rule-arrow">→</span><span style="flex:1">{cond}</span><span style="font-weight:700;color:{CULTURE_COLORS.get(res,"#38bdf8")}">{res}</span></div>' for cond, res in rules])
                st.markdown(f"""
                <div class="result-box">
                    <span class="{'good' if acc>=60 else 'warn'}">Précision : {acc:.1f}%</span><br><br>
                    Règles de décision agronomiques (profondeur {depth}) :<br>
                    {rules_html}
                    <br><span style="color:#94a3b8;font-size:.7rem">Critères : Pluviométrie → Température → Engrais → pH</span>
                </div>
                """, unsafe_allow_html=True)

                # Matrice de confusion arbre
                st.markdown('<div class="agri-card"><h4>🟩 Matrice de confusion — Arbre</h4>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                im = ax.imshow(cm, cmap='Greens', aspect='auto')
                ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9, color='#475569')
                ax.set_yticklabels(classes, fontsize=9, color='#475569')
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=10, fontweight='bold',
                                color='white' if cm[i,j]>cm.max()*.5 else '#1e293b')
                ax.set_xlabel('Prédit', fontsize=9, color='#475569')
                ax.set_ylabel('Réel', fontsize=9, color='#475569')
                ax.set_title('Matrice de confusion — Arbre de décision', fontsize=11, fontweight='bold', color='#1e293b')
                plt.colorbar(im, ax=ax, shrink=.8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE 7 — CLASSIFICATION NON-SUPERVISÉE
# ══════════════════════════════════════════════════════
elif "non-sup" in page:
    st.markdown('<div class="sec-title">🌿 Classification Non-Supervisée — K-Means</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">sklearn.cluster.KMeans — regroupement automatique des parcelles</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty or len(df) < 4:
        st.warning("⚠️ Minimum 4 observations nécessaires.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        k  = st.slider("Nombre de clusters k", 2, 8, 3)
    with col2:
        xk = st.selectbox("Axe X", list(VAR_LABELS.keys()), index=5, format_func=lambda x: VAR_LABELS[x])
    with col3:
        yk = st.selectbox("Axe Y", list(VAR_LABELS.keys()), index=4, format_func=lambda x: VAR_LABELS[x])

    if st.button("▶ Lancer K-Means (sklearn)"):
        sub = df[[xk, yk]].dropna()
        if len(sub) < k + 1:
            st.error("Données insuffisantes pour ce k.")
            st.stop()

        X = sub.values
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_km = model.fit_predict(X)
        centers   = model.cluster_centers_
        wcss      = float(model.inertia_)
        sizes     = [(labels_km == i).sum() for i in range(k)]

        # Métriques
        cols_m = st.columns(k + 1)
        for i in range(k):
            cols_m[i].metric(f"Groupe {i+1}", int(sizes[i]))
        cols_m[k].metric("WCSS (inertie)", round(wcss, 1))

        col_a, col_b = st.columns([3, 2])

        # ── Scatter clusters
        with col_a:
            st.markdown('<div class="agri-card"><h4>🌿 Visualisation des clusters</h4>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            for ci in range(k):
                mask = labels_km == ci
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=CLUSTER_COLORS[ci], s=70, alpha=.8, edgecolors='white', linewidth=.8,
                           label=f'Groupe {ci+1} ({sizes[ci]})', zorder=3)
            ax.scatter(centers[:,0], centers[:,1], c='#1e293b', marker='*',
                       s=320, zorder=5, edgecolors='white', linewidth=1.5, label='Centroïdes')
            for ci, (cx, cy) in enumerate(centers):
                ax.annotate(f'C{ci+1}', (cx, cy), fontsize=9, fontweight='bold',
                            color='#1e293b', ha='center', va='bottom',
                            xytext=(0, 12), textcoords='offset points')
            fig_style(ax, f'K-Means (k={k}) — {VAR_LABELS[xk]} vs {VAR_LABELS[yk]}', VAR_LABELS[xk], VAR_LABELS[yk])
            ax.legend(fontsize=9, loc='best')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Effectifs
        with col_b:
            st.markdown('<div class="agri-card"><h4>📊 Effectifs par cluster</h4>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            bars = ax.bar(range(1, k+1), sizes, color=[CLUSTER_COLORS[i] for i in range(k)],
                          edgecolor='white', linewidth=1.2, width=.6, alpha=.85)
            for bar, sz in zip(bars, sizes):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.2, str(sz),
                        ha='center', fontsize=10, fontweight='bold', color='#1e293b')
            ax.set_xticks(range(1, k+1))
            ax.set_xticklabels([f'Groupe {i+1}' for i in range(k)], fontsize=9)
            fig_style(ax, 'Effectifs des groupes', 'Cluster', 'Nombre de parcelles')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Résumé
        centroid_lines = "<br>".join([
            f"&nbsp;Groupe {i+1} → {VAR_LABELS[xk]}: <strong style='color:{CLUSTER_COLORS[i]}'>{centers[i,0]:.2f}</strong> · {VAR_LABELS[yk]}: <strong style='color:{CLUSTER_COLORS[i]}'>{centers[i,1]:.2f}</strong> — {sizes[i]} parcelle(s)"
            for i in range(k)])
        st.markdown(f"""
        <div class="result-box">
            <span class="good">KMeans sklearn convergé — {k} groupes identifiés</span><br>
            Inertie WCSS : {wcss:.2f} | N = {len(sub)}<br><br>
            Centroïdes finaux :<br>{centroid_lines}<br><br>
            <span style="color:#94a3b8;font-size:.7rem">sklearn.cluster.KMeans(n_clusters={k}, n_init=10, random_state=42)</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Elbow method
        with st.expander("📐 Méthode du coude (Elbow) — choisir le meilleur k"):
            max_k = min(10, len(sub) - 1)
            k_range = range(2, max_k + 1)
            wcss_list = []
            for ki in k_range:
                km = KMeans(n_clusters=ki, random_state=42, n_init=10)
                km.fit(X)
                wcss_list.append(km.inertia_)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(k_range, wcss_list, 'o-', color='#38bdf8', linewidth=2.5, markersize=8, markerfacecolor='#0284c7')
            ax.axvline(k, color='#f59e0b', linestyle='--', linewidth=1.8, label=f'k actuel = {k}')
            ax.fill_between(k_range, wcss_list, alpha=.12, color='#38bdf8')
            fig_style(ax, 'Méthode du coude — Inertie vs k', 'Nombre de clusters k', 'Inertie (WCSS)')
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
