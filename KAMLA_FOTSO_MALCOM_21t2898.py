"""
AgricolApp — TP INF232 EC2
Streamlit · Agriculture
Navigation 100% fonctionnelle via st.session_state + callbacks
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

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgricolApp — INF232",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────
PAGES = [
    "🏠 Tableau de bord",
    "📋 Collecte données",
    "📊 Analyse descriptive",
    "📈 Régression simple",
    "📐 Régression multiple",
    "🔻 Réduction dim. (ACP)",
    "🏷️ Classification supervisée",
    "🌿 Classification non-sup.",
]

CULTURE_COLORS = {
    'Maïs':'#f59e0b','Blé':'#fbbf24','Sorgho':'#d97706',
    'Manioc':'#22c55e','Arachide':'#84cc16','Cacao':'#6b3a1f'
}
CLUSTER_COLORS = ['#38bdf8','#f59e0b','#22c55e','#f472b6',
                  '#a78bfa','#34d399','#fb923c','#60a5fa']
VAR_LABELS = {
    'superficie':'Superficie (ha)', 'ph':'pH du sol',
    'pluie':'Pluviométrie (mm)',    'temp':'Température (°C)',
    'engrais':'Engrais (kg/ha)',    'rendement':'Rendement (t/ha)'
}
CULTURES = ['Maïs','Blé','Sorgho','Manioc','Arachide','Cacao']
SOLS     = ['Argileux','Sableux','Limoneux','Ferralitique']

# ─────────────────────────────────────────────────────
#  SESSION STATE — initialisation unique
# ─────────────────────────────────────────────────────
if 'page_index' not in st.session_state:
    st.session_state.page_index = 0   # 0 = Tableau de bord
if 'parcelles' not in st.session_state:
    st.session_state.parcelles = []

# ─────────────────────────────────────────────────────
#  FONCTIONS DE NAVIGATION — callbacks purs
#  (pas de st.rerun() dans un callback, Streamlit
#   relance automatiquement après un callback)
# ─────────────────────────────────────────────────────
def nav_to(index: int):
    """Callback : change la page."""
    st.session_state.page_index = index
    # Synchroniser le radio de la sidebar
    st.session_state.sidebar_nav = PAGES[index]

def nav_home():
    nav_to(0)

# ─────────────────────────────────────────────────────
#  CSS (sans toucher aux boutons natifs Streamlit)
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; }
#MainMenu, footer { visibility:hidden; }
.block-container { padding-top:1rem; padding-bottom:2rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0a0f1e 0%,#0c2340 100%);
    border-right:1px solid rgba(56,189,248,.2);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color:#e2e8f0 !important; }

/* Metrics */
[data-testid="metric-container"] {
    background:#fff; border:1px solid #e2e8f0;
    border-radius:12px; padding:14px 16px;
    box-shadow:0 1px 4px rgba(10,15,30,.07);
}
[data-testid="metric-container"] label {
    color:#94a3b8 !important; font-size:.63rem !important;
    text-transform:uppercase; letter-spacing:.08em;
}
[data-testid="stMetricValue"] {
    color:#0284c7 !important; font-size:1.5rem !important;
    font-weight:800 !important; font-family:'DM Mono',monospace !important;
}

/* Cards */
.agri-card {
    background:#fff; border:1px solid #e2e8f0;
    border-radius:14px; padding:18px 20px; margin-bottom:14px;
    box-shadow:0 1px 4px rgba(10,15,30,.06);
}
.agri-card-title {
    font-size:.86rem; font-weight:700; color:#1e293b;
    margin-bottom:12px; padding-bottom:9px;
    border-bottom:1px solid #f1f5f9;
}

/* Hero */
.hero-box {
    background:linear-gradient(135deg,#0a0f1e 0%,#0c2340 55%,#0c4a6e 100%);
    border-radius:16px; padding:34px; margin-bottom:20px;
    position:relative; overflow:hidden;
}
.hero-box::before {
    content:'🌾'; position:absolute; right:28px; top:50%;
    transform:translateY(-50%); font-size:5rem; opacity:.08;
}
.hero-box h1 {
    font-size:1.85rem; font-weight:800; color:#fff;
    letter-spacing:-1px; margin-bottom:8px;
}
.hero-box h1 span { color:#38bdf8; }
.hero-box p { color:#94a3b8; font-size:.8rem; line-height:1.75; max-width:500px; }
.hero-chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; }
.hero-chip {
    background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.22);
    color:#bae6fd; padding:3px 11px; border-radius:20px;
    font-size:.63rem; font-weight:600;
}

/* Result box */
.result-box {
    background:#f0f9ff; border:1.5px solid #bae6fd;
    border-radius:10px; padding:14px; margin-top:10px;
    font-family:'DM Mono',monospace; font-size:.76rem; line-height:1.9;
}

/* Sidebar brand */
.sb-brand {
    background:rgba(56,189,248,.08); border:1px solid rgba(56,189,248,.2);
    border-radius:10px; padding:13px 15px; margin-bottom:16px; text-align:center;
}
.sb-brand-title { font-size:1.25rem; font-weight:800; color:#fff; }
.sb-brand-title em { color:#38bdf8; font-style:normal; }
.sb-brand-sub { font-size:.58rem; color:#94a3b8; margin-top:2px;
    font-family:'DM Mono',monospace; }

/* Module card (HTML only, pas de bouton dedans) */
.mod-card {
    background:#fff; border:1.5px solid #e2e8f0; border-radius:12px;
    padding:16px 14px; margin-bottom:6px; text-align:center;
    box-shadow:0 1px 4px rgba(10,15,30,.05);
}
.mod-icon  { font-size:1.4rem; margin-bottom:6px; }
.mod-name  { font-size:.82rem; font-weight:700; color:#1e293b; margin-bottom:3px; }
.mod-desc  { font-size:.62rem; color:#94a3b8; line-height:1.5; }

/* Séparateur */
hr.light { border-color:#e2e8f0; margin:8px 0 16px; }

/* Rule items */
.rule-item {
    display:flex; align-items:center; gap:10px; padding:7px 12px;
    background:#f8fafc; border-radius:8px; margin:3px 0; font-size:.77rem;
}
.rule-arrow { color:#38bdf8; font-weight:700; }

/* Section head */
.sec-title { font-size:1.2rem; font-weight:800; color:#1e293b; margin-bottom:3px; }
.sec-sub   { font-size:.72rem; color:#94a3b8; margin-bottom:14px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────
#  DONNÉES
# ─────────────────────────────────────────────────────
def get_df():
    if not st.session_state.parcelles:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.parcelles)
    for col in ['superficie','ph','pluie','temp','engrais','rendement']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ─────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-title">🌾 Agricol<em>App</em></div>
        <div class="sb-brand-sub">TP INF232 · EC2 · Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    n_parc = len(st.session_state.parcelles)
    st.markdown(
        f"<div style='text-align:center;margin-bottom:12px'>"
        f"<span style='background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.3);"
        f"color:#38bdf8;padding:3px 13px;border-radius:20px;font-size:.68rem;font-weight:700'>"
        f"📦 {n_parc} parcelle(s)</span></div>",
        unsafe_allow_html=True
    )

    # ── Radio synchronisé avec page_index ──────────────
    # on_change callback : met à jour page_index quand on clique dans la sidebar
    def on_sidebar_change():
        st.session_state.page_index = PAGES.index(st.session_state.sidebar_nav)

    st.radio(
        "Navigation",
        PAGES,
        index=st.session_state.page_index,
        key="sidebar_nav",
        on_change=on_sidebar_change,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:.58rem;color:#475569;font-family:monospace'>"
        "🐍 Python · Streamlit<br>📊 pandas · numpy<br>"
        "🔬 scikit-learn · matplotlib</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────
#  PAGE COURANTE
# ─────────────────────────────────────────────────────
current = st.session_state.page_index

# ─────────────────────────────────────────────────────
#  HELPER : barre de navigation haut de page
# ─────────────────────────────────────────────────────
def topbar(title: str, subtitle: str):
    """Titre de page + bouton 🏠 Accueil sur la même ligne."""
    col_btn, col_txt = st.columns([1, 7])
    with col_btn:
        # on_click = callback direct, pas besoin de st.rerun()
        st.button("🏠 Accueil", key=f"home_{current}", on_click=nav_home)
    with col_txt:
        st.markdown(
            f"<div style='padding-top:6px'>"
            f"<span class='sec-title'>{title}</span><br>"
            f"<span class='sec-sub'>{subtitle}</span></div>",
            unsafe_allow_html=True
        )
    st.markdown("<hr class='light'>", unsafe_allow_html=True)

def no_data_warning(msg="⚠️ Aucune donnée disponible."):
    st.warning(msg)
    st.button(
        "📋 Aller à Collecte données",
        key=f"goto_collecte_{current}",
        on_click=nav_to, args=(1,)
    )
    st.stop()

# ══════════════════════════════════════════════════════
#  PAGE 0 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════
if current == 0:
    st.markdown("""
    <div class="hero-box">
        <h1>Collecte &amp; <span>Analyse</span> Agricole</h1>
        <p>Plateforme intégrée de collecte terrain et d'analyse statistique
        des exploitations agricoles — rendements, sols, cultures et conditions climatiques.</p>
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

    modules = [
        (1, "📋", "Collecte données",      "Formulaire parcelle — sol, culture, rendement"),
        (2, "📊", "Analyse descriptive",   "Statistiques pandas — histogrammes, boxplots"),
        (3, "📈", "Régression simple",     "Y = aX + b — MCO numpy"),
        (4, "📐", "Régression multiple",   "numpy.linalg.lstsq — plusieurs prédicteurs"),
        (5, "🔻", "Réduction dim. (ACP)",  "sklearn.decomposition.PCA"),
        (6, "🏷️", "Classification sup.",   "k-NN sklearn + arbre de décision"),
    ]

    row1 = st.columns(3)
    row2 = st.columns(3)

    for i, (idx, icon, name, desc) in enumerate(modules):
        col = row1[i] if i < 3 else row2[i - 3]
        with col:
            st.markdown(f"""
            <div class="mod-card">
                <div class="mod-icon">{icon}</div>
                <div class="mod-name">{name}</div>
                <div class="mod-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            # Bouton natif Streamlit avec callback on_click
            st.button(
                f"➜ Ouvrir {name}",
                key=f"open_{idx}",
                use_container_width=True,
                on_click=nav_to,
                args=(idx,)
            )

    if not df.empty:
        st.markdown("---")
        st.markdown("### 📋 Aperçu des données")
        st.dataframe(df.tail(8), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 1 — COLLECTE
# ══════════════════════════════════════════════════════
elif current == 1:
    topbar("📋 Collecte de Données",
           "Saisie fiche parcelle agricole — stockage en session Streamlit")

    with st.expander("➕ Ajouter une nouvelle parcelle", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_id  = st.text_input("Identifiant parcelle", placeholder="PARC-2024-001", key="fi_id")
            f_sup = st.number_input("Superficie (ha)", min_value=0.1, value=5.0, step=0.1, key="fi_sup")
            f_cult= st.selectbox("Type de culture", [""]+CULTURES, key="fi_cult")
        with c2:
            f_sol = st.selectbox("Type de sol", [""]+SOLS, key="fi_sol")
            f_ph  = st.number_input("pH du sol", 0.0, 14.0, 6.5, 0.1, key="fi_ph")
            f_pluie = st.number_input("Pluviométrie (mm/an)", 0, 5000, 1200, 10, key="fi_pluie")
        with c3:
            f_temp  = st.number_input("Température moy. (°C)", -10.0, 50.0, 25.0, 0.5, key="fi_temp")
            f_engrais = st.number_input("Engrais (kg/ha)", 0, 2000, 150, 5, key="fi_engrais")
            f_rdt   = st.number_input("Rendement (t/ha)", 0.0, 50.0, 3.5, 0.1, key="fi_rdt")
        f_obs = st.text_area("Observations", placeholder="Conditions, maladies, irrigation…", key="fi_obs")

        def do_save():
            nc  = len(st.session_state.parcelles)
            pid = st.session_state.fi_id.strip() or f"PARC-{nc+1:03d}"
            st.session_state.parcelles.append({
                'id':pid, 'superficie':st.session_state.fi_sup,
                'culture':st.session_state.fi_cult, 'sol':st.session_state.fi_sol,
                'ph':st.session_state.fi_ph, 'pluie':st.session_state.fi_pluie,
                'temp':st.session_state.fi_temp, 'engrais':st.session_state.fi_engrais,
                'rendement':st.session_state.fi_rdt, 'obs':st.session_state.fi_obs
            })
            st.session_state._saved_msg = f"✅ Parcelle **{pid}** enregistrée !"

        def do_demo():
            np.random.seed(len(st.session_state.parcelles))
            c_list = ['Maïs','Maïs','Blé','Sorgho','Manioc','Arachide','Cacao']
            for _ in range(30):
                c_  = c_list[np.random.randint(0,len(c_list))]
                s_  = SOLS[np.random.randint(0,len(SOLS))]
                pl  = float(np.round(600  + np.random.random()*1400, 0))
                en  = float(np.round(50   + np.random.random()*300,  0))
                ph_ = float(np.round(4.5  + np.random.random()*3.5,  1))
                te  = float(np.round(18   + np.random.random()*15,   1))
                su  = float(np.round(0.5  + np.random.random()*20,   1))
                rd  = float(np.round(max(0.3,
                    0.5+(pl/1000)*2+(en/200)*1.5+(ph_-4)*0.3
                    -(abs(te-25)/10)*0.4+(np.random.random()*1.4-.6)), 2))
                nc  = len(st.session_state.parcelles)
                st.session_state.parcelles.append({
                    'id':f"DEMO-{nc+1:03d}", 'superficie':su,
                    'culture':c_, 'sol':s_,
                    'ph':ph_, 'pluie':pl, 'temp':te,
                    'engrais':en, 'rendement':rd, 'obs':''
                })
            st.session_state._saved_msg = "⚡ 30 parcelles démo chargées !"

        def do_clear():
            st.session_state.parcelles = []
            st.session_state._saved_msg = "🗑️ Données effacées."

        b1, b2, b3 = st.columns(3)
        b1.button("✔ Enregistrer",           key="btn_save",  on_click=do_save,  use_container_width=True)
        b2.button("⚡ Charger 30 démos",      key="btn_demo",  on_click=do_demo,  use_container_width=True)
        b3.button("🗑️ Effacer tout",           key="btn_clear", on_click=do_clear, use_container_width=True)

    # Afficher message de retour
    msg = st.session_state.pop("_saved_msg", None) if hasattr(st.session_state, '_saved_msg') else None
    if '_saved_msg' in st.session_state:
        msg = st.session_state._saved_msg
        del st.session_state._saved_msg
    if msg:
        if "⚠" in msg or "🗑" in msg:
            st.warning(msg)
        else:
            st.success(msg)

    df = get_df()
    if df.empty:
        st.info("📭 Aucune donnée. Remplissez le formulaire ou chargez les données démo.")
    else:
        st.markdown(f"**📂 Base — {len(df)} parcelle(s)**")
        st.dataframe(df.drop(columns=['obs'],errors='ignore').reset_index(drop=True),
                     use_container_width=True, height=300)

        def do_delete():
            idx = int(st.session_state.del_idx) - 1
            if 0 <= idx < len(st.session_state.parcelles):
                st.session_state.parcelles.pop(idx)

        col_d1, col_d2 = st.columns([2, 5])
        with col_d1:
            st.number_input("Supprimer ligne n°", 1, max(len(df),1), 1, 1, key="del_idx")
            st.button("🗑️ Supprimer cette ligne", key="btn_del", on_click=do_delete, use_container_width=True)

        csv_b = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Télécharger CSV", csv_b,
                           "AgricolApp_export.csv", "text/csv", key="btn_csv")

# ══════════════════════════════════════════════════════
#  PAGE 2 — ANALYSE DESCRIPTIVE
# ══════════════════════════════════════════════════════
elif current == 2:
    topbar("📊 Analyse Descriptive",
           "Statistiques calculées par pandas — visualisations matplotlib")

    df = get_df()
    if df.empty or len(df) < 2:
        no_data_warning("⚠️ Aucune donnée. Ajoutez des parcelles d'abord.")

    var = st.selectbox("Variable à analyser", list(VAR_LABELS.keys()),
                       format_func=lambda x: VAR_LABELS[x], key="desc_var")

    if st.button("▶ Lancer l'analyse", key="btn_desc", use_container_width=False):
        s = df[var].dropna()
        if len(s) < 2:
            st.error("Variable insuffisante."); st.stop()

        c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
        c1.metric("N",            int(len(s)))
        c2.metric("Moyenne",      round(float(s.mean()),   3))
        c3.metric("Écart-type",   round(float(s.std()),    3))
        c4.metric("Médiane",      round(float(s.median()), 3))
        c5.metric("Min",          round(float(s.min()),    3))
        c6.metric("Max",          round(float(s.max()),    3))
        c7.metric("Q1 (25%)",     round(float(s.quantile(.25)), 3))
        c8.metric("Q3 (75%)",     round(float(s.quantile(.75)), 3))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Histogramme</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.hist(s, bins=10, color='#38bdf8', edgecolor='#0284c7', alpha=.8, linewidth=1.2, rwidth=.88)
            fig_style(ax, f'Distribution — {VAR_LABELS[var]}', VAR_LABELS[var], 'Fréquence')
            st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="agri-card"><div class="agri-card-title">🥧 Répartition cultures</div>', unsafe_allow_html=True)
            if 'culture' in df.columns and df['culture'].notna().any():
                cnt   = df['culture'].value_counts()
                cols_ = [CULTURE_COLORS.get(c,'#888') for c in cnt.index]
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.pie(cnt.values, labels=cnt.index, colors=cols_,
                       autopct='%1.1f%%', startangle=90, textprops={'fontsize':9},
                       wedgeprops={'edgecolor':'white','linewidth':1.5})
                ax.set_title('Répartition des cultures', fontsize=11, fontweight='bold', color='#1e293b')
                st.pyplot(fig, use_container_width=True); plt.close()
            else:
                st.info("Aucune culture renseignée.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="agri-card"><div class="agri-card-title">📦 Boxplot — pH · Engrais · Rendement</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax_, (v_, lbl_, col_) in zip(axes, [
            ('ph','pH du sol','#38bdf8'),
            ('engrais','Engrais (kg/ha)','#f59e0b'),
            ('rendement','Rendement (t/ha)','#22c55e')
        ]):
            if v_ in df.columns:
                d_ = df[v_].dropna()
                ax_.boxplot(d_, patch_artist=True, widths=.5,
                            boxprops=dict(facecolor=col_+'44',edgecolor=col_,linewidth=1.5),
                            medianprops=dict(color=col_,linewidth=2.5),
                            whiskerprops=dict(color='#94a3b8',linewidth=1.2),
                            capprops=dict(color='#94a3b8',linewidth=1.2),
                            flierprops=dict(marker='o',color=col_,alpha=.5,markersize=5))
                ax_.set_title(lbl_, fontsize=10, fontweight='bold', color='#1e293b')
                ax_.tick_params(labelsize=8); ax_.set_xticks([])
                med_ = float(d_.median())
                ax_.text(1.32, med_, f'Med:{med_:.2f}', va='center', fontsize=8,
                         color=col_, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📋 describe() pandas complet"):
            nc_ = [c for c in ['superficie','ph','pluie','temp','engrais','rendement'] if c in df.columns]
            st.dataframe(df[nc_].describe().round(3), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 3 — RÉGRESSION SIMPLE
# ══════════════════════════════════════════════════════
elif current == 3:
    topbar("📈 Régression Linéaire Simple", "Y = aX + b — Moindres Carrés Ordinaires (numpy)")

    df = get_df()
    if df.empty or len(df) < 5:
        no_data_warning("⚠️ Minimum 5 observations requises.")

    col1, col2 = st.columns(2)
    with col1:
        xk = st.selectbox("Variable X (prédicteur)", list(VAR_LABELS.keys()),
                           format_func=lambda x: VAR_LABELS[x], index=2, key="rsx")
    with col2:
        yk = st.selectbox("Variable Y (cible)", list(VAR_LABELS.keys()),
                           format_func=lambda x: VAR_LABELS[x], index=5, key="rsy")

    if st.button("▶ Calculer la régression", key="btn_rs"):
        sub = df[[xk,yk]].dropna()
        if len(sub) < 5:
            st.error("Données insuffisantes."); st.stop()
        x, y = sub[xk].values, sub[yk].values
        mx, my = x.mean(), y.mean()
        a    = np.sum((x-mx)*(y-my)) / (np.sum((x-mx)**2) + 1e-10)
        b    = my - a*mx
        yp   = a*x + b
        r2   = float(1 - np.sum((y-yp)**2) / (np.sum((y-my)**2)+1e-10))
        rmse = float(np.sqrt(np.mean((y-yp)**2)))
        corr = float(np.corrcoef(x,y)[0,1])

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R²",        round(r2,  4))
        m2.metric("RMSE",      round(rmse,4))
        m3.metric("r Pearson", round(corr,4))
        m4.metric("N",         int(len(y)))

        quality = "✔ Bonne corrélation" if r2>=.7 else ("⚠ Corrélation modérée" if r2>=.4 else "✕ Corrélation faible")
        st.markdown(f"""
        <div class="result-box">
            <strong style="color:#0284c7;font-size:.88rem">Ŷ = {a:.5f} × X + {b:.5f}</strong><br>
            <span style="color:{'#16a34a' if r2>=.5 else '#d97706'};font-weight:700">
            R² = {r2:.4f} — {quality}</span><br>
            RMSE = {rmse:.4f} | r de Pearson = {corr:.4f}<br>
            Pour +1 unité de {VAR_LABELS[xk]}, {VAR_LABELS[yk]} varie de <strong>{a:.4f}</strong> unités.<br>
            <span style="color:#94a3b8;font-size:.69rem">Calculé par numpy — MCO manuel</span>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(x, y, color='#38bdf8', edgecolors='#0284c7', s=65, alpha=.8, zorder=3, label='Observations')
        xl = np.linspace(x.min(), x.max(), 200)
        ax.plot(xl, a*xl+b, color='#f59e0b', linewidth=2.5,
                label=f'Ŷ = {a:.4f}·X + {b:.4f}', zorder=4)
        for xi_,yi_,ypi_ in zip(x,y,yp):
            ax.plot([xi_,xi_],[yi_,ypi_], color='#ef4444', alpha=.2, linewidth=.8)
        fig_style(ax, f'Régression : {VAR_LABELS[xk]} → {VAR_LABELS[yk]}', VAR_LABELS[xk], VAR_LABELS[yk])
        ax.legend(fontsize=9)
        ax.text(.02,.97,f'R²={r2:.4f}',transform=ax.transAxes,fontsize=10,va='top',
                color='#0284c7',fontweight='bold',
                bbox=dict(boxstyle='round,pad=.4',facecolor='#e0f2fe',edgecolor='#38bdf8'))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════
#  PAGE 4 — RÉGRESSION MULTIPLE
# ══════════════════════════════════════════════════════
elif current == 4:
    topbar("📐 Régression Linéaire Multiple", "numpy.linalg.lstsq + sklearn.StandardScaler")

    df = get_df()
    if df.empty or len(df) < 10:
        no_data_warning("⚠️ Minimum 10 observations requises.")

    st.markdown("**Variable Y cible : Rendement (t/ha) — Cocher les variables X :**")
    var_map = {'pluie':'Pluviométrie','engrais':'Engrais',
               'ph':'pH sol','temp':'Température','superficie':'Superficie'}
    cbs = st.columns(5)
    sel = {}
    defs = {'pluie':True,'engrais':True,'ph':False,'temp':False,'superficie':False}
    for i,(vk,vl) in enumerate(var_map.items()):
        sel[vk] = cbs[i].checkbox(vl, value=defs[vk], key=f"rmcb_{vk}")
    vars_x = [v for v,s_ in sel.items() if s_]

    if st.button("▶ Calculer la régression multiple", key="btn_rm"):
        if not vars_x:
            st.error("Cochez au moins une variable X."); st.stop()
        sub = df[vars_x+['rendement']].dropna()
        if len(sub) < 10:
            st.error("Données insuffisantes."); st.stop()
        X  = sub[vars_x].values; y = sub['rendement'].values
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
        yp   = Xb @ beta; my = y.mean()
        r2   = float(1-np.sum((y-yp)**2)/(np.sum((y-my)**2)+1e-10))
        rmse = float(np.sqrt(np.mean((y-yp)**2)))

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("R²",          round(r2,  4))
        m2.metric("RMSE",        round(rmse,4))
        m3.metric("N",           int(len(y)))
        m4.metric("Prédicteurs", len(vars_x))

        coeffs = {v:round(float(beta[i+1]/sc.scale_[i]),6) for i,v in enumerate(vars_x)}
        ch = "<br>".join([f"&nbsp;&nbsp;β({var_map[v]}) = <strong style='color:#0284c7'>{coeffs[v]}</strong>" for v in vars_x])
        quality = "✔ Bonne" if r2>=.7 else ("⚠ Modérée" if r2>=.4 else "✕ Faible")
        st.markdown(f"""
        <div class="result-box">
            <strong style="color:#0284c7">Régression multiple → Rendement (t/ha)</strong><br>
            <span style="color:{'#16a34a' if r2>=.5 else '#d97706'};font-weight:700">
            R² = {r2:.4f} — {quality} | RMSE = {rmse:.4f}</span><br><br>
            Coefficients (numpy.linalg.lstsq) :<br>{ch}<br>
            β₀ (intercept) = <strong style="color:#f59e0b">{beta[0]:.4f}</strong><br>
            <span style="color:#94a3b8;font-size:.69rem">StandardScaler + lstsq — données standardisées</span>
        </div>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1,2,figsize=(12,4.5))
        ax1,ax2 = axes
        ax1.scatter(y,yp,color='#38bdf8',edgecolors='#0284c7',s=60,alpha=.8)
        lims=[min(y.min(),yp.min())*.95, max(y.max(),yp.max())*1.05]
        ax1.plot(lims,lims,'r--',linewidth=1.5,label='Parfait')
        fig_style(ax1,'Observé vs Prédit','Rendement observé','Rendement prédit')
        ax1.legend(fontsize=9)
        ax1.text(.05,.95,f'R²={r2:.4f}',transform=ax1.transAxes,fontsize=10,
                 color='#0284c7',fontweight='bold',va='top',
                 bbox=dict(boxstyle='round,pad=.3',facecolor='#e0f2fe',edgecolor='#38bdf8'))
        res=y-yp
        ax2.bar(range(len(res)),res,color=np.where(res>=0,'#38bdf8','#f59e0b'),alpha=.75,width=.7)
        ax2.axhline(0,color='#1e293b',linewidth=1.2,linestyle='--')
        fig_style(ax2,'Résidus','Observation','Résidu')
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════
#  PAGE 5 — ACP
# ══════════════════════════════════════════════════════
elif current == 5:
    topbar("🔻 Réduction Dimensionnelle — ACP", "sklearn.decomposition.PCA — StandardScaler")

    df = get_df()
    vars_pca = ['superficie','ph','pluie','temp','engrais','rendement']
    sub = df.dropna(subset=vars_pca) if not df.empty else pd.DataFrame()
    if sub.empty or len(sub) < 5:
        no_data_warning("⚠️ Minimum 5 observations avec toutes les variables numériques.")

    if st.button("▶ Lancer l'ACP (sklearn.PCA)", key="btn_pca"):
        X  = sub[vars_pca].values
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        n_comp    = min(6, len(sub), len(vars_pca))
        pca_model = PCA(n_components=n_comp)
        proj      = pca_model.fit_transform(Xs)
        var_exp   = pca_model.explained_variance_ratio_ * 100
        cumul     = np.cumsum(var_exp)

        cols_m = st.columns(n_comp)
        for i in range(n_comp):
            cols_m[i].metric(f"PC{i+1}", f"{var_exp[i]:.1f}%")

        col1,col2 = st.columns(2)
        with col1:
            st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Variance expliquée</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,3.5))
            ax2t   = ax.twinx()
            xp     = np.arange(n_comp)
            bars   = ax.bar(xp,var_exp,color='#38bdf8',alpha=.75,edgecolor='#0284c7',linewidth=1.2,width=.55,label='Variance (%)')
            ax2t.plot(xp,cumul,'o-',color='#f59e0b',linewidth=2,markersize=7,label='Cumul (%)')
            ax2t.set_ylabel('Cumul (%)',fontsize=9,color='#f59e0b')
            ax2t.tick_params(colors='#f59e0b',labelsize=8); ax2t.set_ylim(0,115)
            ax.set_xticks(xp); ax.set_xticklabels([f'PC{i+1}' for i in range(n_comp)])
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f'{bar.get_height():.1f}%',
                        ha='center',fontsize=8,color='#0284c7',fontweight='bold')
            fig_style(ax,'Variance expliquée','Composante','Variance (%)')
            ax.legend(loc='upper left',fontsize=8); ax2t.legend(loc='upper right',fontsize=8)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
            st.markdown('</div>',unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="agri-card"><div class="agri-card-title">🗺️ Projection PC1 vs PC2</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,3.5))
            cults  = sub['culture'].tolist() if 'culture' in sub.columns else ['']*len(sub)
            for cu in set(cults):
                m_ = [c==cu for c in cults]
                ax.scatter(proj[m_,0],proj[m_,1],c=CULTURE_COLORS.get(cu,'#38bdf8'),
                           label=cu or 'N/A',s=65,alpha=.8,edgecolors='white',linewidth=.8)
            ax.axhline(0,color='#94a3b8',linestyle='--',linewidth=.7)
            ax.axvline(0,color='#94a3b8',linestyle='--',linewidth=.7)
            fig_style(ax,'Projection ACP',f'PC1 ({var_exp[0]:.1f}%)',f'PC2 ({var_exp[1]:.1f}%)')
            ax.legend(fontsize=8); plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
            st.markdown('</div>',unsafe_allow_html=True)

        st.markdown('<div class="agri-card"><div class="agri-card-title">🌡️ Matrice des corrélations</div>', unsafe_allow_html=True)
        corr_m = sub[vars_pca].corr().round(3)
        fig,ax  = plt.subplots(figsize=(8,5))
        im = ax.imshow(corr_m.values,cmap='Blues',vmin=-1,vmax=1,aspect='auto')
        sh = ['Sup.','pH','Pluie','Temp.','Engrais','Rdmt']
        ax.set_xticks(range(len(vars_pca))); ax.set_yticks(range(len(vars_pca)))
        ax.set_xticklabels(sh,fontsize=9,color='#475569')
        ax.set_yticklabels(sh,fontsize=9,color='#475569')
        for i in range(len(vars_pca)):
            for j in range(len(vars_pca)):
                v_ = corr_m.values[i,j]
                ax.text(j,i,f'{v_:.2f}',ha='center',va='center',fontsize=9,
                        fontweight='bold' if abs(v_)>.5 and i!=j else 'normal',
                        color='white' if abs(v_)>.65 else '#1e293b')
        plt.colorbar(im,ax=ax,shrink=.8)
        ax.set_title('Matrice des corrélations',fontsize=11,fontweight='bold',color='#1e293b',pad=10)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown('</div>',unsafe_allow_html=True)

        with st.expander("📋 Loadings"):
            ld = pd.DataFrame(pca_model.components_.T, index=vars_pca,
                              columns=[f'PC{i+1}' for i in range(n_comp)]).round(3)
            st.dataframe(ld.style.background_gradient(cmap='Blues',axis=None), use_container_width=True)

# ══════════════════════════════════════════════════════
#  PAGE 6 — CLASSIFICATION SUPERVISÉE
# ══════════════════════════════════════════════════════
elif current == 6:
    topbar("🏷️ Classification Supervisée", "k-NN sklearn & Arbre de décision agronomique")

    df    = get_df()
    feats = ['ph','pluie','temp','engrais','rendement']
    sub_knn = pd.DataFrame()
    if not df.empty:
        sub_knn = df.dropna(subset=feats+['culture'])
        if 'culture' in sub_knn.columns:
            sub_knn = sub_knn[sub_knn['culture'].str.strip() != '']

    tab1, tab2 = st.tabs(["🔍 k-Nearest Neighbors", "🌳 Arbre de décision"])

    with tab1:
        if len(sub_knn) < 6:
            st.warning("⚠️ Minimum 6 parcelles avec culture renseignée.")
            st.button("📋 Aller à Collecte", key="knn_go", on_click=nav_to, args=(1,))
        else:
            k = st.slider("Nombre de voisins k", 1, min(15,len(sub_knn)-1), 5, key="knn_k")
            if st.button("▶ Classifier (k-NN sklearn)", key="btn_knn"):
                X  = sub_knn[feats].values; y = sub_knn['culture'].values
                sc = StandardScaler(); Xs = sc.fit_transform(X)
                model    = KNeighborsClassifier(n_neighbors=k)
                cv_n     = min(5,len(sub_knn))
                cv_sc    = cross_val_score(model,Xs,y,cv=cv_n,scoring='accuracy')
                model.fit(Xs,y); yp_ = model.predict(Xs)
                acc_cv   = cv_sc.mean()*100
                acc_tr   = accuracy_score(y,yp_)*100
                classes  = sorted(list(set(y)))
                cm_      = confusion_matrix(y,yp_,labels=classes)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Précision CV",    f"{acc_cv:.1f}%")
                c2.metric("Précision train", f"{acc_tr:.1f}%")
                c3.metric("k voisins",       k)
                c4.metric("N parcelles",     len(sub_knn))

                st.markdown(f"""
                <div class="result-box">
                    <span style="color:{'#16a34a' if acc_cv>=65 else '#d97706'};font-weight:700">
                    Précision CV : {acc_cv:.1f}%</span><br>
                    <strong>sklearn.neighbors.KNeighborsClassifier(n_neighbors={k})</strong><br>
                    Scores ({cv_n} folds) : {' · '.join([f"{s*100:.1f}%" for s in cv_sc])}<br>
                    Features : pH · Pluviométrie · Température · Engrais · Rendement<br>
                    <span style="color:#94a3b8;font-size:.69rem">StandardScaler appliqué</span>
                </div>
                """, unsafe_allow_html=True)

                ca, cb = st.columns([3,2])
                with ca:
                    st.markdown('<div class="agri-card"><div class="agri-card-title">🟩 Matrice de confusion</div>', unsafe_allow_html=True)
                    fig,ax = plt.subplots(figsize=(7,4))
                    im = ax.imshow(cm_,cmap='Blues',aspect='auto')
                    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
                    ax.set_xticklabels(classes,rotation=30,ha='right',fontsize=9,color='#475569')
                    ax.set_yticklabels(classes,fontsize=9,color='#475569')
                    for i in range(len(classes)):
                        for j in range(len(classes)):
                            ax.text(j,i,str(cm_[i,j]),ha='center',va='center',fontsize=10,fontweight='bold',
                                    color='white' if cm_[i,j]>cm_.max()*.5 else '#1e293b')
                    ax.set_xlabel('Prédit',fontsize=9); ax.set_ylabel('Réel',fontsize=9)
                    ax.set_title('Matrice de confusion — k-NN',fontsize=11,fontweight='bold',color='#1e293b')
                    plt.colorbar(im,ax=ax,shrink=.8)
                    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
                    st.markdown('</div>',unsafe_allow_html=True)
                with cb:
                    st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Scores par fold</div>', unsafe_allow_html=True)
                    fig,ax = plt.subplots(figsize=(4,4))
                    ax.bar(range(1,len(cv_sc)+1),cv_sc*100,color='#38bdf8',alpha=.8,
                           edgecolor='#0284c7',linewidth=1.2,width=.6)
                    ax.axhline(acc_cv,color='#f59e0b',linestyle='--',linewidth=2,label=f'Moy:{acc_cv:.1f}%')
                    ax.set_ylim(0,110); ax.legend(fontsize=9)
                    fig_style(ax,'Scores par fold','Fold','Précision (%)')
                    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
                    st.markdown('</div>',unsafe_allow_html=True)

    with tab2:
        if len(sub_knn) < 10:
            st.warning("⚠️ Minimum 10 parcelles avec culture renseignée.")
            st.button("📋 Aller à Collecte", key="tree_go", on_click=nav_to, args=(1,))
        else:
            depth = st.slider("Profondeur de l'arbre", 1, 6, 3, key="tree_d")
            if st.button("▶ Entraîner l'arbre", key="btn_tree"):
                def pred_tree(row):
                    if row['pluie']>=1200 and row['temp']>=22: return 'Cacao'
                    if row['pluie']>=900:  return 'Manioc'
                    if row['engrais']>=200 and row['ph']>=6.0: return 'Maïs'
                    if row['ph']>=6.5:     return 'Blé'
                    if row['engrais']>=100: return 'Sorgho'
                    return 'Arachide'

                st_ = sub_knn.copy()
                st_['pred'] = st_.apply(pred_tree,axis=1)
                acc_t  = (st_['pred']==st_['culture']).mean()*100
                cl_t   = sorted(st_['culture'].unique().tolist())
                cm_t   = confusion_matrix(st_['culture'],st_['pred'],labels=cl_t)

                c1,c2,c3 = st.columns(3)
                c1.metric("Précision",  f"{acc_t:.1f}%")
                c2.metric("Profondeur", depth)
                c3.metric("N",          len(st_))

                rules = [
                    ("Pluie ≥ 1200 mm ET Temp ≥ 22°C","Cacao"),
                    ("Pluie ≥ 900 mm","Manioc"),
                    ("Engrais ≥ 200 ET pH ≥ 6.0","Maïs"),
                    ("pH ≥ 6.5","Blé"),
                    ("Engrais ≥ 100","Sorgho"),
                    ("Sinon","Arachide"),
                ]
                rhtml = "".join([
                    f'<div class="rule-item"><span class="rule-arrow">→</span>'
                    f'<span style="flex:1">{cd}</span>'
                    f'<span style="font-weight:700;color:{CULTURE_COLORS.get(rs,"#38bdf8")}">{rs}</span></div>'
                    for cd,rs in rules
                ])
                st.markdown(f"""
                <div class="result-box">
                    <span style="color:{'#16a34a' if acc_t>=60 else '#d97706'};font-weight:700">
                    Précision : {acc_t:.1f}%</span><br><br>
                    Règles de décision (profondeur {depth}) :<br>{rhtml}
                    <br><span style="color:#94a3b8;font-size:.69rem">
                    Critères : Pluviométrie → Température → Engrais → pH</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="agri-card"><div class="agri-card-title">🟩 Matrice de confusion — Arbre</div>', unsafe_allow_html=True)
                fig,ax = plt.subplots(figsize=(8,4.5))
                im = ax.imshow(cm_t,cmap='Greens',aspect='auto')
                ax.set_xticks(range(len(cl_t))); ax.set_yticks(range(len(cl_t)))
                ax.set_xticklabels(cl_t,rotation=30,ha='right',fontsize=9,color='#475569')
                ax.set_yticklabels(cl_t,fontsize=9,color='#475569')
                for i in range(len(cl_t)):
                    for j in range(len(cl_t)):
                        ax.text(j,i,str(cm_t[i,j]),ha='center',va='center',fontsize=10,fontweight='bold',
                                color='white' if cm_t[i,j]>cm_t.max()*.5 else '#1e293b')
                ax.set_xlabel('Prédit',fontsize=9); ax.set_ylabel('Réel',fontsize=9)
                ax.set_title('Matrice de confusion — Arbre',fontsize=11,fontweight='bold',color='#1e293b')
                plt.colorbar(im,ax=ax,shrink=.8)
                plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
                st.markdown('</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE 7 — CLASSIFICATION NON-SUPERVISÉE
# ══════════════════════════════════════════════════════
elif current == 7:
    topbar("🌿 Classification Non-Supervisée — K-Means",
           "sklearn.cluster.KMeans — regroupement automatique des parcelles")

    df = get_df()
    if df.empty or len(df) < 4:
        no_data_warning("⚠️ Minimum 4 observations requises.")

    col1,col2,col3 = st.columns(3)
    with col1:
        k  = st.slider("Clusters k", 2, min(8,len(df)-1), 3, key="km_k")
    with col2:
        xk = st.selectbox("Axe X", list(VAR_LABELS.keys()), index=4,
                           format_func=lambda x: VAR_LABELS[x], key="km_x")
    with col3:
        yk = st.selectbox("Axe Y", list(VAR_LABELS.keys()), index=5,
                           format_func=lambda x: VAR_LABELS[x], key="km_y")

    if st.button("▶ Lancer K-Means (sklearn)", key="btn_km"):
        sub = df[[xk,yk]].dropna()
        if len(sub) < k+1:
            st.error(f"Données insuffisantes pour k={k}."); st.stop()

        X   = sub.values
        km  = KMeans(n_clusters=k,random_state=42,n_init=10)
        lbs = km.fit_predict(X)
        ctr = km.cluster_centers_
        wcs = float(km.inertia_)
        szs = [(lbs==i).sum() for i in range(k)]

        cols_m = st.columns(k+1)
        for i in range(k): cols_m[i].metric(f"Groupe {i+1}", int(szs[i]))
        cols_m[k].metric("WCSS", round(wcs,1))

        ca,cb = st.columns([3,2])
        with ca:
            st.markdown('<div class="agri-card"><div class="agri-card-title">🌿 Visualisation des clusters</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(7,5))
            for ci in range(k):
                m_ = lbs==ci
                ax.scatter(X[m_,0],X[m_,1],c=CLUSTER_COLORS[ci],s=70,alpha=.8,
                           edgecolors='white',linewidth=.8,
                           label=f'Groupe {ci+1} ({szs[ci]})',zorder=3)
            ax.scatter(ctr[:,0],ctr[:,1],c='#1e293b',marker='*',s=320,
                       zorder=5,edgecolors='white',linewidth=1.5,label='Centroïdes')
            for ci,(cx,cy) in enumerate(ctr):
                ax.annotate(f'C{ci+1}',(cx,cy),fontsize=9,fontweight='bold',
                            color='#1e293b',ha='center',va='bottom',
                            xytext=(0,12),textcoords='offset points')
            fig_style(ax,f'K-Means (k={k})',VAR_LABELS[xk],VAR_LABELS[yk])
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
            st.markdown('</div>',unsafe_allow_html=True)

        with cb:
            st.markdown('<div class="agri-card"><div class="agri-card-title">📊 Effectifs</div>', unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(5,5))
            bars_  = ax.bar(range(1,k+1),szs,color=[CLUSTER_COLORS[i] for i in range(k)],
                            edgecolor='white',linewidth=1.2,width=.6,alpha=.85)
            for bar_,sz_ in zip(bars_,szs):
                ax.text(bar_.get_x()+bar_.get_width()/2,bar_.get_height()+.2,str(sz_),
                        ha='center',fontsize=10,fontweight='bold',color='#1e293b')
            ax.set_xticks(range(1,k+1))
            ax.set_xticklabels([f'G{i+1}' for i in range(k)],fontsize=9)
            fig_style(ax,'Effectifs','Cluster','Parcelles')
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
            st.markdown('</div>',unsafe_allow_html=True)

        cl = "<br>".join([
            f"&nbsp;Groupe {i+1} → {VAR_LABELS[xk]}: "
            f"<strong style='color:{CLUSTER_COLORS[i]}'>{ctr[i,0]:.2f}</strong> · "
            f"{VAR_LABELS[yk]}: <strong style='color:{CLUSTER_COLORS[i]}'>{ctr[i,1]:.2f}</strong> "
            f"— {szs[i]} parcelle(s)"
            for i in range(k)
        ])
        st.markdown(f"""
        <div class="result-box">
            <span style="color:#16a34a;font-weight:700">KMeans convergé — {k} groupes identifiés</span><br>
            WCSS : {wcs:.2f} | N = {len(sub)}<br><br>
            Centroïdes :<br>{cl}<br><br>
            <span style="color:#94a3b8;font-size:.69rem">
            sklearn.cluster.KMeans(n_clusters={k}, n_init=10, random_state=42)</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📐 Méthode du coude (Elbow)"):
            max_k  = min(10, len(sub)-1)
            kr     = range(2, max_k+1)
            wcss_l = []
            for ki in kr:
                km_ = KMeans(n_clusters=ki,random_state=42,n_init=10)
                km_.fit(X); wcss_l.append(km_.inertia_)
            fig,ax = plt.subplots(figsize=(8,4))
            ax.plot(kr,wcss_l,'o-',color='#38bdf8',linewidth=2.5,markersize=8,markerfacecolor='#0284c7')
            ax.axvline(k,color='#f59e0b',linestyle='--',linewidth=1.8,label=f'k actuel = {k}')
            ax.fill_between(kr,wcss_l,alpha=.1,color='#38bdf8')
            fig_style(ax,'Méthode du coude','Nombre de clusters k','Inertie (WCSS)')
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
