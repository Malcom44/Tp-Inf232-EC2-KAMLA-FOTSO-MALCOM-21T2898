"""
AgricolApp — TP INF232 EC2
Application Streamlit · Agriculture
Corrections : boutons fonctionnels + boutons retour accueil
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgricolApp — INF232",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
#  CSS  — NOTE : on ne touche PAS aux boutons Streamlit
#  pour ne pas casser leur interactivité
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0f1e 0%,#0c2340 100%);
    border-right: 1px solid rgba(56,189,248,.2);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 9px 12px !important; border-radius: 8px !important;
    font-size: .82rem !important; font-weight: 500 !important;
    transition: all .17s !important; cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(56,189,248,.12) !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(10,15,30,.07);
}
[data-testid="metric-container"] label {
    color: #94a3b8 !important; font-size: .63rem !important;
    text-transform: uppercase; letter-spacing: .08em;
}
[data-testid="stMetricValue"] {
    color: #0284c7 !important; font-size: 1.55rem !important;
    font-weight: 800 !important; font-family: 'DM Mono',monospace !important;
}

/* ── Cards ── */
.agri-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(10,15,30,.07);
}
.agri-card-title {
    font-size: .87rem; font-weight: 700; color: #1e293b;
    margin-bottom: 12px; padding-bottom: 10px;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Hero ── */
.hero-box {
    background: linear-gradient(135deg,#0a0f1e 0%,#0c2340 55%,#0c4a6e 100%);
    border-radius: 16px; padding: 34px; margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.hero-box::before {
    content:'🌾'; position:absolute; right:28px; top:50%;
    transform:translateY(-50%); font-size:5rem; opacity:.08;
}
.hero-box h1 { font-size:1.9rem; font-weight:800; color:#fff; letter-spacing:-1px; margin-bottom:8px; }
.hero-box h1 span { color:#38bdf8; }
.hero-box p { color:#94a3b8; font-size:.8rem; line-height:1.75; max-width:500px; }
.hero-chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; }
.hero-chip {
    background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.22);
    color:#bae6fd; padding:3px 11px; border-radius:20px; font-size:.63rem; font-weight:600;
}

/* ── Result box ── */
.result-box {
    background:#f0f9ff; border:1.5px solid #bae6fd;
    border-radius:10px; padding:14px; margin-top:10px;
    font-family:'DM Mono',monospace; font-size:.76rem; line-height:1.9;
}

/* ── Section head ── */
.sec-title { font-size:1.2rem; font-weight:800; color:#1e293b; margin-bottom:3px; }
.sec-sub   { font-size:.72rem; color:#94a3b8; margin-bottom:16px; }

/* ── Sidebar brand ── */
.sb-brand {
    background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.2);
    border-radius:10px; padding:13px 15px; margin-bottom:16px; text-align:center;
}
.sb-brand-title { font-size:1.25rem; font-weight:800; color:#fff; }
.sb-brand-title em { color:#38bdf8; font-style:normal; }
.sb-brand-sub { font-size:.58rem; color:#94a3b8; margin-top:2px; font-family:'DM Mono',monospace; }

/* ── Module card ── */
.mod-card {
    background:#fff; border:1.5px solid #e2e8f0; border-radius:12px;
    padding:16px; margin-bottom:10px; text-align:center;
    box-shadow:0 1px 4px rgba(10,15,30,.06);
}
.mod-card:hover { border-color:#38bdf8; box-shadow:0 6px 20px rgba(56,189,248,.12); }
.mod-icon { font-size:1.5rem; margin-bottom:6px; }
.mod-name { font-size:.82rem; font-weight:700; color:#1e293b; margin-bottom:3px; }
.mod-desc { font-size:.63rem; color:#94a3b8; line-height:1.5; }

/* ── Bouton retour (custom pill) ── */
.back-btn-wrap { margin-bottom: 18px; }

/* ── Rule items ── */
.rule-item {
    display:flex; align-items:center; gap:10px; padding:7px 12px;
    background:#f8fafc; border-radius:8px; margin:3px 0; font-size:.77rem;
}
.rule-arrow { color:#38bdf8; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════
CULTURE_COLORS = {
    'Maïs':'#f59e0b','Blé':'#fbbf24','Sorgho':'#d97706',
    'Manioc':'#22c55e','Arachide':'#84cc16','Cacao':'#6b3a1f'
}
CLUSTER_COLORS = ['#38bdf8','#f59e0b','#22c55e','#f472b6','#a78bfa','#34d399','#fb923c','#60a5fa']
VAR_LABELS = {
    'superficie':'Superficie (ha)','ph':'pH du sol',
    'pluie':'Pluviométrie (mm)','temp':'Température (°C)',
    'engrais':'Engrais (kg/ha)','rendement':'Rendement (t/ha)'
}
CULTURES = ['Maïs','Blé','Sorgho','Manioc','Arachide','Cacao']
SOLS     = ['Argileux','Sableux','Limoneux','Ferralitique']

PAGES = [
    "🏠  Tableau de bord",
    "📋  Collecte données",
    "📊  Analyse descriptive",
    "📈  Régression simple",
    "📐  Régression multiple",
    "🔻  Réduction dim. (ACP)",
    "🏷️  Classification supervisée",
    "🌿  Classification non-sup.",
]

# ══════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════
if 'parcelles' not in st.session_state:
    st.session_state.parcelles = []
if 'page' not in st.session_state:
    st.session_state.page = PAGES[0]

def go_home():
    st.session_state.page = PAGES[0]

def go_to(p):
    st.session_state.page = p

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
    'figure.facecolor':'#ffffff','axes.facecolor':'#f8fafc',
    'axes.edgecolor':'#e2e8f0','axes.labelcolor':'#475569',
    'xtick.color':'#94a3b8','ytick.color':'#94a3b8',
    'text.color':'#1e293b','grid.color':'#e2e8f0','grid.alpha':.5,
    'axes.grid':True,'font.family':'DejaVu Sans',
    'axes.spines.top':False,'axes.spines.right':False,
})

def fig_style(ax, title='', xlabel='', ylabel=''):
    ax.set_title(title, fontsize=11, fontweight='bold', color='#1e293b', pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color='#475569')
    ax.set_ylabel(ylabel, fontsize=9, color='#475569')
    ax.tick_params(labelsize=8)

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-title">🌾 Agricol<em>App</em></div>
        <div class="sb-brand-sub">TP INF232 · EC2 · Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    n = len(st.session_state.parcelles)
    st.markdown(
        f"<div style='text-align:center;margin-bottom:12px'>"
        f"<span style='background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.3);"
        f"color:#38bdf8;padding:3px 13px;border-radius:20px;font-size:.68rem;font-weight:700'>"
        f"📦 {n} parcelle(s)</span></div>",
        unsafe_allow_html=True
    )

    # Navigation par radio — synchronisé avec session_state
    selected = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.page),
        label_visibility="collapsed",
        key="sidebar_radio"
    )
    # Synchroniser la sélection radio → session state
    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.58rem;color:#475569;font-family:monospace'>"
        "🐍 Python · Streamlit<br>📊 pandas · numpy<br>🔬 scikit-learn · matplotlib</div>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════
#  HELPER — BOUTON RETOUR ACCUEIL
# ══════════════════════════════════════════════════════
def show_back_button():
    """Affiche un bouton 🏠 Accueil en haut de chaque page secondaire."""
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("🏠 Accueil", key=f"back_{st.session_state.page}"):
            go_home()
            st.rerun()
    st.markdown("<hr style='margin:6px 0 18px;border-color:#e2e8f0'>", unsafe_allow_html=True)

# Récupérer la page courante
page = st.session_state.page

# ══════════════════════════════════════════════════════
#  PAGE 0 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════
if "Tableau" in page:
    st.markdown("""
    <div class="hero-box">
        <h1>Collecte &amp; <span>Analyse</span> Agricole</h1>
        <p>Plateforme intégrée de collecte terrain et d'analyse statistique des exploitations agricoles — rendements, sols, cultures et conditions climatiques.</p>
        <div class="hero-chips">
            <span class="hero-chip">🌾 Céréales &amp; Cultures</span>
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

    st.markdown("### 🗂️ Accès rapide aux modules")

    # Ligne 1 : 3 modules
    c1, c2, c3 = st.columns(3)
    modules = [
        (PAGES[1], "📋", "Collecte données",       "Formulaire parcelle — sol, culture, rendement"),
        (PAGES[2], "📊", "Analyse descriptive",    "Statistiques pandas — histogrammes, boxplots"),
        (PAGES[3], "📈", "Régression simple",      "Y = aX + b — MCO numpy"),
        (PAGES[4], "📐", "Régression multiple",    "numpy.linalg.lstsq — plusieurs prédicteurs"),
        (PAGES[5], "🔻", "Réduction dim. (ACP)",   "sklearn.decomposition.PCA"),
        (PAGES[6], "🏷️", "Classification sup.",    "k-NN sklearn + arbre de décision"),
    ]
    cols_row1 = [c1, c2, c3]
    cols_row2 = st.columns(3)

    for i, (target_page, icon, name, desc) in enumerate(modules):
        col = cols_row1[i] if i < 3 else cols_row2[i-3]
        with col:
            st.markdown(f"""
            <div class="mod-card">
                <div class="mod-icon">{icon}</div>
                <div class="mod-name">{name}</div>
                <div class="mod-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Ouvrir {name}", key=f"btn_mod_{i}", use_container_width=True):
                go_to(target_page)
                st.rerun()

    if not df.empty:
        st.markdown("---")
        st.markdown("### 📋 Aperçu des données récentes")
        st.dataframe(df.tail(8), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 1 — COLLECTE
# ══════════════════════════════════════════════════════
elif "Collecte" in page:
    show_back_button()
    st.markdown('<div class="sec-title">📋 Collecte de Données</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Saisie fiche parcelle agricole — stockage en session Streamlit</div>', unsafe_allow_html=True)

    with st.expander("➕ Ajouter une nouvelle parcelle", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_id        = st.text_input("Identifiant parcelle", placeholder="PARC-2024-001", key="f_id")
            f_superficie = st.number_input("Superficie (ha)", min_value=0.1, value=5.0, step=0.1, key="f_sup")
            f_culture   = st.selectbox("Type de culture", [""] + CULTURES, key="f_cult")
        with c2:
            f_sol       = st.selectbox("Type de sol", [""] + SOLS, key="f_sol")
            f_ph        = st.number_input("pH du sol", min_value=0.0, max_value=14.0, value=6.5, step=0.1, key="f_ph")
            f_pluie     = st.number_input("Pluviométrie (mm/an)", min_value=0, value=1200, step=10, key="f_pluie")
        with c3:
            f_temp      = st.number_input("Température moy. (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5, key="f_temp")
            f_engrais   = st.number_input("Engrais (kg/ha)", min_value=0, value=150, step=5, key="f_engrais")
            f_rendement = st.number_input("Rendement (t/ha)", min_value=0.0, value=3.5, step=0.1, key="f_rdt")
        f_obs = st.text_area("Observations terrain", placeholder="Conditions, maladies, irrigation…", key="f_obs")

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("✔ Enregistrer la parcelle", use_container_width=True, key="btn_save"):
                n_curr = len(st.session_state.parcelles)
                pid = f_id.strip() or f"PARC-{n_curr+1:03d}"
                st.session_state.parcelles.append({
                    'id': pid, 'superficie': f_superficie,
                    'culture': f_culture, 'sol': f_sol,
                    'ph': f_ph, 'pluie': f_pluie, 'temp': f_temp,
                    'engrais': f_engrais, 'rendement': f_rendement, 'obs': f_obs
                })
                st.success(f"✅ Parcelle **{pid}** enregistrée avec succès !")
                st.rerun()

        with col_btn2:
            if st.button("⚡ Charger 30 données démo", use_container_width=True, key="btn_demo"):
                np.random.seed(42 + len(st.session_state.parcelles))
                cultures_l = ['Maïs','Maïs','Blé','Sorgho','Manioc','Arachide','Cacao']
                for i in range(30):
                    c_  = cultures_l[np.random.randint(0, len(cultures_l))]
                    s_  = SOLS[np.random.randint(0, len(SOLS))]
                    pl  = float(np.round(600  + np.random.random()*1400, 0))
                    en  = float(np.round(50   + np.random.random()*300,  0))
                    ph_ = float(np.round(4.5  + np.random.random()*3.5,  1))
                    te  = float(np.round(18   + np.random.random()*15,   1))
                    su  = float(np.round(0.5  + np.random.random()*20,   1))
                    rd  = float(np.round(max(0.3,
                        0.5+(pl/1000)*2+(en/200)*1.5+(ph_-4)*0.3-(abs(te-25)/10)*0.4+(np.random.random()*1.4-.6)
                    ), 2))
                    nc = len(st.session_state.parcelles)
                    st.session_state.parcelles.append({
                        'id': f"DEMO-{nc+1:03d}", 'superficie': su,
                        'culture': c_, 'sol': s_,
                        'ph': ph_, 'pluie': pl, 'temp': te,
                        'engrais': en, 'rendement': rd, 'obs': ''
                    })
                st.success("⚡ 30 parcelles démo chargées !")
                st.rerun()

        with col_btn3:
            if st.button("🗑️ Effacer toutes les données", use_container_width=True, key="btn_clear"):
                st.session_state.parcelles = []
                st.warning("🗑️ Toutes les données ont été effacées.")
                st.rerun()

    # ── Tableau
    df = get_df()
    if df.empty:
        st.info("📭 Aucune donnée. Remplissez le formulaire ou chargez les données démo.")
    else:
        st.markdown(f"**📂 Base de données — {len(df)} parcelle(s)**")
        st.dataframe(
            df.drop(columns=['obs'], errors='ignore').reset_index(drop=True),
            use_container_width=True, height=320
        )

        col_d1, col_d2 = st.columns([2, 5])
        with col_d1:
            idx_del = st.number_input("Supprimer la ligne n°", min_value=1,
                                      max_value=max(len(df),1), value=1, step=1, key="del_idx")
            if st.button("🗑️ Supprimer cette ligne", key="btn_del_row"):
                st.session_state.parcelles.pop(int(idx_del) - 1)
                st.success(f"Ligne {idx_del} supprimée.")
                st.rerun()

        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Télécharger CSV",
            csv_bytes,
            "AgricolApp_export.csv",
            "text/csv",
            key="btn_csv"
        )

# ══════════════════════════════════════════════════════
#  PAGE 2 — ANALYSE DESCRIPTIVE
# ══════════════════════════════════════════════════════
elif "descriptive" in page:
    show_back_button()
    st.markdown('<div class="sec-title">📊 Analyse Descriptive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Statistiques calculées par pandas — visualisations matplotlib</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty:
        st.warning("⚠️ Aucune donnée. Allez dans **Collecte données** pour en ajouter.")
        if st.button("📋 Aller à Collecte données", key="desc_go_collecte"):
            go_to(PAGES[1]); st.rerun()
        st.stop()

    var = st.selectbox("Variable à analyser", list(VAR_LABELS.keys()),
                       format_func=lambda x: VAR_LABELS[x], key="desc_var")

    if st.button("▶ Lancer l'analyse", key="btn_desc", use_container_width=False):
        s = df[var].dropna()
        c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
        c1.metric("N",          int(len(s)))
        c2.metric("Moyenne",    round(float(s.mean()),   3))
        c3.metric("Écart-type", round(float(s.std()),    3))
        c4.metric("Médiane",    round(float(s.median()), 3))
        c5.metric("Min",        round(float(s.min()),    3))
        c6.metric("Max",        round(float(s.max()),    3))
        c7.metric("Q1 (25%)",   round(float(s.quantile(.25)), 3))
        c8.metric("Q3 (75%)",   round(float(s.quantile(.75)), 3))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Histogramme</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.hist(s, bins=10, color='#38bdf8', edgecolor='#0284c7', alpha=.8, linewidth=1.2, rwidth=.88)
            fig_style(ax, f'Distribution — {VAR_LABELS[var]}', VAR_LABELS[var], 'Fréquence')
            st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="agri-card"><div class="agri-card-title">🥧 Répartition des cultures</div>', unsafe_allow_html=True)
            if 'culture' in df.columns and df['culture'].notna().any():
                cult_cnt = df['culture'].value_counts()
                colors   = [CULTURE_COLORS.get(c,'#888') for c in cult_cnt.index]
                fig, ax  = plt.subplots(figsize=(6, 3.4))
                ax.pie(cult_cnt.values, labels=cult_cnt.index, colors=colors,
                       autopct='%1.1f%%', startangle=90, textprops={'fontsize':9},
                       wedgeprops={'edgecolor':'white','linewidth':1.5})
                ax.set_title('Répartition des cultures', fontsize=11, fontweight='bold', color='#1e293b')
                st.pyplot(fig, use_container_width=True); plt.close()
            else:
                st.info("Aucune culture renseignée.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="agri-card"><div class="agri-card-title">📦 Boxplot comparatif — pH · Engrais · Rendement</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax_, (v_, lbl_, col_) in zip(axes, [('ph','pH du sol','#38bdf8'),('engrais','Engrais (kg/ha)','#f59e0b'),('rendement','Rendement (t/ha)','#22c55e')]):
            if v_ in df.columns:
                data_ = df[v_].dropna()
                ax_.boxplot(data_, patch_artist=True, widths=.5,
                            boxprops=dict(facecolor=col_+'44', edgecolor=col_, linewidth=1.5),
                            medianprops=dict(color=col_, linewidth=2.5),
                            whiskerprops=dict(color='#94a3b8', linewidth=1.2),
                            capprops=dict(color='#94a3b8', linewidth=1.2),
                            flierprops=dict(marker='o', color=col_, alpha=.5, markersize=5))
                ax_.set_title(lbl_, fontsize=10, fontweight='bold', color='#1e293b')
                ax_.tick_params(labelsize=8); ax_.set_xticks([])
                med_ = float(data_.median())
                ax_.text(1.32, med_, f'Med:{med_:.2f}', va='center', fontsize=8, color=col_, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📋 describe() pandas complet"):
            num_cols = [c for c in ['superficie','ph','pluie','temp','engrais','rendement'] if c in df.columns]
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 3 — RÉGRESSION SIMPLE
# ══════════════════════════════════════════════════════
elif "Régression simple" in page or ("simple" in page and "Régression" in page):
    show_back_button()
    st.markdown('<div class="sec-title">📈 Régression Linéaire Simple</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Y = aX + b — Moindres Carrés Ordinaires (numpy)</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty or len(df) < 5:
        st.warning("⚠️ Minimum 5 observations requises.")
        if st.button("📋 Aller à Collecte données", key="reg_s_go"):
            go_to(PAGES[1]); st.rerun()
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        xk = st.selectbox("Variable X (prédicteur)", list(VAR_LABELS.keys()),
                           format_func=lambda x: VAR_LABELS[x], index=2, key="regx")
    with col2:
        yk = st.selectbox("Variable Y (cible)", list(VAR_LABELS.keys()),
                           format_func=lambda x: VAR_LABELS[x], index=5, key="regy")

    if st.button("▶ Calculer la régression simple", key="btn_regs", use_container_width=False):
        sub = df[[xk, yk]].dropna()
        if len(sub) < 5:
            st.error("Données insuffisantes après nettoyage."); st.stop()
        x, y = sub[xk].values, sub[yk].values
        mx, my = x.mean(), y.mean()
        a    = np.sum((x-mx)*(y-my)) / (np.sum((x-mx)**2) + 1e-10)
        b    = my - a*mx
        yp   = a*x + b
        r2   = float(1 - np.sum((y-yp)**2) / (np.sum((y-my)**2) + 1e-10))
        rmse = float(np.sqrt(np.sum((y-yp)**2)/len(y)))
        corr = float(np.corrcoef(x, y)[0, 1])

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R²",    round(r2,   4))
        m2.metric("RMSE",  round(rmse, 4))
        m3.metric("r Pearson", round(corr, 4))
        m4.metric("N", int(len(y)))

        quality = "✔ Bonne corrélation" if r2>=.7 else ("⚠ Corrélation modérée" if r2>=.4 else "✕ Corrélation faible")
        color   = "good" if r2>=.5 else "warn"
        st.markdown(f"""
        <div class="result-box">
            <strong style="color:#0284c7;font-size:.88rem">Ŷ = {a:.5f} × X + {b:.5f}</strong><br>
            <span style="color:{'#16a34a' if r2>=.5 else '#d97706'};font-weight:700">R² = {r2:.4f} — {quality}</span><br>
            RMSE = {rmse:.4f} | r de Pearson = {corr:.4f}<br>
            Pour +1 unité de {VAR_LABELS[xk]}, {VAR_LABELS[yk]} varie de <strong>{a:.4f}</strong> unités.<br>
            <span style="color:#94a3b8;font-size:.69rem">Calculé par numpy MCO</span>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(x, y, color='#38bdf8', edgecolors='#0284c7', s=65, alpha=.8, zorder=3, label='Observations')
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, a*x_line+b, color='#f59e0b', linewidth=2.5,
                label=f'Ŷ = {a:.4f}·X + {b:.4f}', zorder=4)
        for xi_, yi_, ypi_ in zip(x, y, yp):
            ax.plot([xi_,xi_],[yi_,ypi_], color='#ef4444', alpha=.2, linewidth=.8)
        fig_style(ax, f'Régression : {VAR_LABELS[xk]} → {VAR_LABELS[yk]}', VAR_LABELS[xk], VAR_LABELS[yk])
        ax.legend(fontsize=9)
        ax.text(.02,.97,f'R² = {r2:.4f}',transform=ax.transAxes,fontsize=10,
                va='top',color='#0284c7',fontweight='bold',
                bbox=dict(boxstyle='round,pad=.4',facecolor='#e0f2fe',edgecolor='#38bdf8'))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════
#  PAGE 4 — RÉGRESSION MULTIPLE
# ══════════════════════════════════════════════════════
elif "multiple" in page:
    show_back_button()
    st.markdown('<div class="sec-title">📐 Régression Linéaire Multiple</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">MCO — numpy.linalg.lstsq + sklearn.StandardScaler</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty or len(df) < 10:
        st.warning("⚠️ Minimum 10 observations requises.")
        if st.button("📋 Aller à Collecte données", key="regm_go"):
            go_to(PAGES[1]); st.rerun()
        st.stop()

    st.markdown("**Variable Y cible fixe : Rendement (t/ha)**")
    st.markdown("**Sélectionner les variables X :**")
    var_map = {'pluie':'Pluviométrie','engrais':'Engrais','ph':'pH sol','temp':'Température','superficie':'Superficie'}
    cols_cb = st.columns(5)
    selected_vars = {}
    defaults_     = {'pluie':True,'engrais':True,'ph':False,'temp':False,'superficie':False}
    for i,(vk,vl) in enumerate(var_map.items()):
        selected_vars[vk] = cols_cb[i].checkbox(vl, value=defaults_[vk], key=f"cb_{vk}")

    vars_x = [v for v,sel in selected_vars.items() if sel]

    if st.button("▶ Calculer la régression multiple", key="btn_regm", use_container_width=False):
        if not vars_x:
            st.error("Cochez au moins une variable X."); st.stop()
        sub = df[vars_x+['rendement']].dropna()
        if len(sub) < 10:
            st.error("Données insuffisantes après nettoyage."); st.stop()

        X  = sub[vars_x].values
        y  = sub['rendement'].values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        yp   = Xb @ beta
        my   = y.mean()
        r2   = float(1 - np.sum((y-yp)**2)/(np.sum((y-my)**2)+1e-10))
        rmse = float(np.sqrt(np.mean((y-yp)**2)))

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R²",          round(r2,  4))
        m2.metric("RMSE",        round(rmse,4))
        m3.metric("N",           int(len(y)))
        m4.metric("Prédicteurs", len(vars_x))

        coeffs = {v:round(float(beta[i+1]/sc.scale_[i]),6) for i,v in enumerate(vars_x)}
        coeff_html = "<br>".join([f"&nbsp;&nbsp;β({var_map[v]}) = <strong style='color:#0284c7'>{coeffs[v]}</strong>" for v in vars_x])
        quality = "✔ Bonne" if r2>=.7 else ("⚠ Modérée" if r2>=.4 else "✕ Faible")
        st.markdown(f"""
        <div class="result-box">
            <strong style="color:#0284c7;font-size:.88rem">Régression multiple → Rendement (t/ha)</strong><br>
            <span style="color:{'#16a34a' if r2>=.5 else '#d97706'};font-weight:700">R² = {r2:.4f} — {quality} | RMSE = {rmse:.4f}</span><br><br>
            Coefficients (numpy.linalg.lstsq) :<br>{coeff_html}<br>
            β₀ (intercept) = <strong style="color:#f59e0b">{beta[0]:.4f}</strong><br>
            <span style="color:#94a3b8;font-size:.69rem">StandardScaler + lstsq — données standardisées</span>
        </div>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1, ax2  = axes
        ax1.scatter(y, yp, color='#38bdf8', edgecolors='#0284c7', s=60, alpha=.8)
        lims = [min(y.min(),yp.min())*.95, max(y.max(),yp.max())*1.05]
        ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Parfait')
        fig_style(ax1,'Observé vs Prédit','Rendement observé (t/ha)','Rendement prédit (t/ha)')
        ax1.legend(fontsize=9)
        ax1.text(.05,.95,f'R²={r2:.4f}',transform=ax1.transAxes,fontsize=10,color='#0284c7',
                 fontweight='bold',va='top',bbox=dict(boxstyle='round,pad=.3',facecolor='#e0f2fe',edgecolor='#38bdf8'))
        res = y - yp
        ax2.bar(range(len(res)), res, color=np.where(res>=0,'#38bdf8','#f59e0b'), alpha=.75, width=.7)
        ax2.axhline(0, color='#1e293b', linewidth=1.2, linestyle='--')
        fig_style(ax2,'Résidus','Observation','Résidu (observé − prédit)')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════
#  PAGE 5 — ACP
# ══════════════════════════════════════════════════════
elif "ACP" in page:
    show_back_button()
    st.markdown('<div class="sec-title">🔻 Réduction Dimensionnelle — ACP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">sklearn.decomposition.PCA — StandardScaler</div>', unsafe_allow_html=True)

    df  = get_df()
    vars_pca = ['superficie','ph','pluie','temp','engrais','rendement']
    sub = df.dropna(subset=vars_pca) if not df.empty else pd.DataFrame()

    if sub.empty or len(sub) < 5:
        st.warning("⚠️ Minimum 5 observations avec toutes les variables numériques renseignées.")
        if st.button("📋 Aller à Collecte données", key="pca_go"):
            go_to(PAGES[1]); st.rerun()
        st.stop()

    if st.button("▶ Lancer l'ACP (sklearn.PCA)", key="btn_pca", use_container_width=False):
        X = sub[vars_pca].values
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        n_comp    = min(6, len(sub), len(vars_pca))
        pca_model = PCA(n_components=n_comp)
        proj      = pca_model.fit_transform(Xs)
        var_exp   = pca_model.explained_variance_ratio_ * 100
        cumul     = np.cumsum(var_exp)

        cols_m = st.columns(n_comp)
        for i in range(n_comp):
            cols_m[i].metric(f"PC{i+1}", f"{var_exp[i]:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Variance expliquée</div>', unsafe_allow_html=True)
            fig, ax  = plt.subplots(figsize=(6, 3.5))
            ax2_twin = ax.twinx()
            x_pos = np.arange(n_comp)
            bars  = ax.bar(x_pos, var_exp, color='#38bdf8', alpha=.75, edgecolor='#0284c7', linewidth=1.2, width=.55, label='Variance (%)')
            ax2_twin.plot(x_pos, cumul,'o-',color='#f59e0b',linewidth=2,markersize=7,label='Cumul (%)')
            ax2_twin.set_ylabel('Cumul (%)', fontsize=9, color='#f59e0b')
            ax2_twin.tick_params(colors='#f59e0b', labelsize=8); ax2_twin.set_ylim(0,115)
            ax.set_xticks(x_pos); ax.set_xticklabels([f'PC{i+1}' for i in range(n_comp)])
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5, f'{bar.get_height():.1f}%',
                        ha='center', fontsize=8, color='#0284c7', fontweight='bold')
            fig_style(ax,'Variance expliquée','Composante','Variance (%)')
            ax.legend(loc='upper left',fontsize=8); ax2_twin.legend(loc='upper right',fontsize=8)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="agri-card"><div class="agri-card-title">🗺️ Projection PC1 vs PC2</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            cultures_col  = sub['culture'].tolist() if 'culture' in sub.columns else ['']*len(sub)
            unique_cult   = list(set(cultures_col))
            for cult in unique_cult:
                mask_ = [c==cult for c in cultures_col]
                ax.scatter(proj[mask_,0], proj[mask_,1], c=CULTURE_COLORS.get(cult,'#38bdf8'),
                           label=cult or 'N/A', s=65, alpha=.8, edgecolors='white', linewidth=.8)
            ax.axhline(0,color='#94a3b8',linestyle='--',linewidth=.7)
            ax.axvline(0,color='#94a3b8',linestyle='--',linewidth=.7)
            fig_style(ax,'Projection ACP',f'PC1 ({var_exp[0]:.1f}%)',f'PC2 ({var_exp[1]:.1f}%)')
            if unique_cult: ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="agri-card"><div class="agri-card-title">🌡️ Matrice des corrélations</div>', unsafe_allow_html=True)
        corr_mat = sub[vars_pca].corr().round(3)
        fig, ax  = plt.subplots(figsize=(8, 5))
        im = ax.imshow(corr_mat.values, cmap='Blues', vmin=-1, vmax=1, aspect='auto')
        short = ['Sup.','pH','Pluie','Temp.','Engrais','Rdmt']
        ax.set_xticks(range(len(vars_pca))); ax.set_yticks(range(len(vars_pca)))
        ax.set_xticklabels(short, fontsize=9, color='#475569')
        ax.set_yticklabels(short, fontsize=9, color='#475569')
        for i in range(len(vars_pca)):
            for j in range(len(vars_pca)):
                v_ = corr_mat.values[i,j]
                ax.text(j,i,f'{v_:.2f}',ha='center',va='center',fontsize=9,
                        fontweight='bold' if abs(v_)>.5 and i!=j else 'normal',
                        color='white' if abs(v_)>.65 else '#1e293b')
        plt.colorbar(im,ax=ax,shrink=.8)
        ax.set_title('Matrice des corrélations',fontsize=11,fontweight='bold',color='#1e293b',pad=10)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📋 Loadings (contributions des variables)"):
            load_df = pd.DataFrame(
                pca_model.components_.T,
                index=vars_pca,
                columns=[f'PC{i+1}' for i in range(n_comp)]
            ).round(3)
            st.dataframe(load_df.style.background_gradient(cmap='Blues', axis=None), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 6 — CLASSIFICATION SUPERVISÉE
# ══════════════════════════════════════════════════════
elif "supervisée" in page:
    show_back_button()
    st.markdown('<div class="sec-title">🏷️ Classification Supervisée</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">k-NN (sklearn) & Arbre de décision agronomique</div>', unsafe_allow_html=True)

    df   = get_df()
    feats = ['ph','pluie','temp','engrais','rendement']
    sub_knn = pd.DataFrame()
    if not df.empty:
        sub_knn = df.dropna(subset=feats+['culture'])
        if 'culture' in sub_knn.columns:
            sub_knn = sub_knn[sub_knn['culture'].str.strip() != '']

    tab1, tab2 = st.tabs(["🔍 k-Nearest Neighbors", "🌳 Arbre de décision"])

    # ── K-NN
    with tab1:
        if len(sub_knn) < 6:
            st.warning("⚠️ Minimum 6 parcelles avec culture renseignée.")
            if st.button("📋 Aller à Collecte", key="knn_go"):
                go_to(PAGES[1]); st.rerun()
        else:
            k = st.slider("Nombre de voisins k", 1, min(15, len(sub_knn)-1), 5, key="knn_k")
            if st.button("▶ Classifier avec k-NN (sklearn)", key="btn_knn", use_container_width=False):
                X  = sub_knn[feats].values
                y  = sub_knn['culture'].values
                sc = StandardScaler(); Xs = sc.fit_transform(X)
                model     = KNeighborsClassifier(n_neighbors=k)
                cv_n      = min(5, len(sub_knn))
                cv_scores = cross_val_score(model, Xs, y, cv=cv_n, scoring='accuracy')
                model.fit(Xs, y)
                y_pred    = model.predict(Xs)
                acc_cv    = cv_scores.mean() * 100
                acc_train = accuracy_score(y, y_pred) * 100
                classes   = sorted(list(set(y)))
                cm_       = confusion_matrix(y, y_pred, labels=classes)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Précision CV",    f"{acc_cv:.1f}%")
                c2.metric("Précision train", f"{acc_train:.1f}%")
                c3.metric("k voisins",       k)
                c4.metric("N parcelles",     len(sub_knn))

                st.markdown(f"""
                <div class="result-box">
                    <span style="color:{'#16a34a' if acc_cv>=65 else '#d97706'};font-weight:700">
                    Précision cross-validation : {acc_cv:.1f}%</span><br>
                    Bibliothèque : <strong>sklearn.neighbors.KNeighborsClassifier(n_neighbors={k})</strong><br>
                    Scores CV ({cv_n} folds) : {' · '.join([f'{s*100:.1f}%' for s in cv_scores])}<br>
                    Features : pH · Pluviométrie · Température · Engrais · Rendement<br>
                    <span style="color:#94a3b8;font-size:.69rem">Normalisation : sklearn.preprocessing.StandardScaler</span>
                </div>
                """, unsafe_allow_html=True)

                col_a, col_b = st.columns([3, 2])
                with col_a:
                    st.markdown('<div class="agri-card"><div class="agri-card-title">🟩 Matrice de confusion</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    im = ax.imshow(cm_, cmap='Blues', aspect='auto')
                    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
                    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9, color='#475569')
                    ax.set_yticklabels(classes, fontsize=9, color='#475569')
                    for i in range(len(classes)):
                        for j in range(len(classes)):
                            ax.text(j,i,str(cm_[i,j]),ha='center',va='center',fontsize=10,fontweight='bold',
                                    color='white' if cm_[i,j]>cm_.max()*.5 else '#1e293b')
                    ax.set_xlabel('Prédit',fontsize=9,color='#475569')
                    ax.set_ylabel('Réel',fontsize=9,color='#475569')
                    ax.set_title('Matrice de confusion — k-NN',fontsize=11,fontweight='bold',color='#1e293b')
                    plt.colorbar(im,ax=ax,shrink=.8)
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_b:
                    st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Scores par fold</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.bar(range(1,len(cv_scores)+1), cv_scores*100, color='#38bdf8', alpha=.8,
                           edgecolor='#0284c7', linewidth=1.2, width=.6)
                    ax.axhline(acc_cv, color='#f59e0b', linestyle='--', linewidth=2, label=f'Moy: {acc_cv:.1f}%')
                    ax.set_ylim(0,110); ax.legend(fontsize=9)
                    fig_style(ax,'Scores par fold','Fold','Précision (%)')
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── Arbre de décision
    with tab2:
        if len(sub_knn) < 10:
            st.warning("⚠️ Minimum 10 parcelles avec culture renseignée.")
            if st.button("📋 Aller à Collecte", key="tree_go"):
                go_to(PAGES[1]); st.rerun()
        else:
            depth = st.slider("Profondeur de l'arbre", 1, 6, 3, key="tree_d")
            if st.button("▶ Entraîner l'arbre de décision", key="btn_tree", use_container_width=False):
                def predict_tree(row):
                    if row['pluie'] >= 1200 and row['temp'] >= 22: return 'Cacao'
                    if row['pluie'] >= 900:  return 'Manioc'
                    if row['engrais'] >= 200 and row['ph'] >= 6.0: return 'Maïs'
                    if row['ph'] >= 6.5:     return 'Blé'
                    if row['engrais'] >= 100: return 'Sorgho'
                    return 'Arachide'

                sub_t = sub_knn.copy()
                sub_t['pred'] = sub_t.apply(predict_tree, axis=1)
                acc     = (sub_t['pred'] == sub_t['culture']).mean() * 100
                classes = sorted(sub_t['culture'].unique().tolist())
                cm_t    = confusion_matrix(sub_t['culture'], sub_t['pred'], labels=classes)

                c1,c2,c3 = st.columns(3)
                c1.metric("Précision", f"{acc:.1f}%")
                c2.metric("Profondeur", depth)
                c3.metric("N", len(sub_t))

                rules = [
                    ("Pluie ≥ 1200 mm ET Temp ≥ 22°C", "Cacao"),
                    ("Pluie ≥ 900 mm",                 "Manioc"),
                    ("Engrais ≥ 200 ET pH ≥ 6.0",      "Maïs"),
                    ("pH ≥ 6.5",                       "Blé"),
                    ("Engrais ≥ 100",                  "Sorgho"),
                    ("Sinon",                          "Arachide"),
                ]
                rules_html = "".join([
                    f'<div class="rule-item">'
                    f'<span class="rule-arrow">→</span>'
                    f'<span style="flex:1">{cond}</span>'
                    f'<span style="font-weight:700;color:{CULTURE_COLORS.get(res,"#38bdf8")}">{res}</span>'
                    f'</div>'
                    for cond, res in rules
                ])
                st.markdown(f"""
                <div class="result-box">
                    <span style="color:{'#16a34a' if acc>=60 else '#d97706'};font-weight:700">Précision : {acc:.1f}%</span><br><br>
                    Règles agronomiques (profondeur {depth}) :<br>{rules_html}
                    <br><span style="color:#94a3b8;font-size:.69rem">Critères : Pluviométrie → Température → Engrais → pH</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="agri-card"><div class="agri-card-title">🟩 Matrice de confusion — Arbre</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 4.5))
                im = ax.imshow(cm_t, cmap='Greens', aspect='auto')
                ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9, color='#475569')
                ax.set_yticklabels(classes, fontsize=9, color='#475569')
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        ax.text(j,i,str(cm_t[i,j]),ha='center',va='center',fontsize=10,fontweight='bold',
                                color='white' if cm_t[i,j]>cm_t.max()*.5 else '#1e293b')
                ax.set_xlabel('Prédit',fontsize=9,color='#475569')
                ax.set_ylabel('Réel',fontsize=9,color='#475569')
                ax.set_title('Matrice de confusion — Arbre de décision',fontsize=11,fontweight='bold',color='#1e293b')
                plt.colorbar(im,ax=ax,shrink=.8)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE 7 — CLASSIFICATION NON-SUPERVISÉE
# ══════════════════════════════════════════════════════
elif "non-sup" in page:
    show_back_button()
    st.markdown('<div class="sec-title">🌿 Classification Non-Supervisée — K-Means</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">sklearn.cluster.KMeans — regroupement automatique des parcelles</div>', unsafe_allow_html=True)

    df = get_df()
    if df.empty or len(df) < 4:
        st.warning("⚠️ Minimum 4 observations requises.")
        if st.button("📋 Aller à Collecte données", key="km_go"):
            go_to(PAGES[1]); st.rerun()
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        k  = st.slider("Nombre de clusters k", 2, min(8, len(df)-1), 3, key="km_k")
    with col2:
        xk = st.selectbox("Axe X", list(VAR_LABELS.keys()), index=4,
                           format_func=lambda x: VAR_LABELS[x], key="km_x")
    with col3:
        yk = st.selectbox("Axe Y", list(VAR_LABELS.keys()), index=5,
                           format_func=lambda x: VAR_LABELS[x], key="km_y")

    if st.button("▶ Lancer K-Means (sklearn)", key="btn_km", use_container_width=False):
        sub = df[[xk, yk]].dropna()
        if len(sub) < k+1:
            st.error(f"Données insuffisantes pour k={k}."); st.stop()

        X          = sub.values
        model      = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_km  = model.fit_predict(X)
        centers    = model.cluster_centers_
        wcss       = float(model.inertia_)
        sizes      = [(labels_km==i).sum() for i in range(k)]

        cols_m = st.columns(k+1)
        for i in range(k): cols_m[i].metric(f"Groupe {i+1}", int(sizes[i]))
        cols_m[k].metric("WCSS (inertie)", round(wcss, 1))

        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown('<div class="agri-card"><div class="agri-card-title">🌿 Visualisation des clusters</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            for ci in range(k):
                mask_ = labels_km == ci
                ax.scatter(X[mask_,0], X[mask_,1], c=CLUSTER_COLORS[ci], s=70, alpha=.8,
                           edgecolors='white', linewidth=.8, label=f'Groupe {ci+1} ({sizes[ci]})', zorder=3)
            ax.scatter(centers[:,0], centers[:,1], c='#1e293b', marker='*', s=320,
                       zorder=5, edgecolors='white', linewidth=1.5, label='Centroïdes')
            for ci,(cx,cy) in enumerate(centers):
                ax.annotate(f'C{ci+1}',(cx,cy),fontsize=9,fontweight='bold',color='#1e293b',
                            ha='center',va='bottom',xytext=(0,12),textcoords='offset points')
            fig_style(ax,f'K-Means (k={k})',VAR_LABELS[xk],VAR_LABELS[yk])
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Effectifs par cluster</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            bars_ = ax.bar(range(1,k+1), sizes, color=[CLUSTER_COLORS[i] for i in range(k)],
                           edgecolor='white', linewidth=1.2, width=.6, alpha=.85)
            for bar_,sz_ in zip(bars_,sizes):
                ax.text(bar_.get_x()+bar_.get_width()/2, bar_.get_height()+.2, str(sz_),
                        ha='center',fontsize=10,fontweight='bold',color='#1e293b')
            ax.set_xticks(range(1,k+1)); ax.set_xticklabels([f'G{i+1}' for i in range(k)],fontsize=9)
            fig_style(ax,'Effectifs des groupes','Cluster','Nombre de parcelles')
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        centroid_lines = "<br>".join([
            f"&nbsp;Groupe {i+1} → {VAR_LABELS[xk]}: "
            f"<strong style='color:{CLUSTER_COLORS[i]}'>{centers[i,0]:.2f}</strong> · "
            f"{VAR_LABELS[yk]}: <strong style='color:{CLUSTER_COLORS[i]}'>{centers[i,1]:.2f}</strong> "
            f"— {sizes[i]} parcelle(s)"
            for i in range(k)
        ])
        st.markdown(f"""
        <div class="result-box">
            <span style="color:#16a34a;font-weight:700">KMeans sklearn convergé — {k} groupes identifiés</span><br>
            Inertie WCSS : {wcss:.2f} | N = {len(sub)}<br><br>
            Centroïdes finaux :<br>{centroid_lines}<br><br>
            <span style="color:#94a3b8;font-size:.69rem">sklearn.cluster.KMeans(n_clusters={k}, n_init=10, random_state=42)</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📐 Méthode du coude — choisir le meilleur k"):
            max_k     = min(10, len(sub)-1)
            k_range   = range(2, max_k+1)
            wcss_list = []
            for ki in k_range:
                km_ = KMeans(n_clusters=ki, random_state=42, n_init=10)
                km_.fit(X); wcss_list.append(km_.inertia_)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(k_range, wcss_list,'o-',color='#38bdf8',linewidth=2.5,markersize=8,markerfacecolor='#0284c7')
            ax.axvline(k, color='#f59e0b', linestyle='--', linewidth=1.8, label=f'k actuel = {k}')
            ax.fill_between(k_range, wcss_list, alpha=.1, color='#38bdf8')
            fig_style(ax,'Méthode du coude','Nombre de clusters k','Inertie (WCSS)')
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
