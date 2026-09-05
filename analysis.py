"""Tableau de Bord Prédictif Maintenance.

Analyse de survie (Kaplan-Meier), scoring de santé machine, prédiction du
RLU (Remaining Life Until failure) par Random Forest, et chiffrage de
l'impact financier (ROI) d'une stratégie de maintenance préventive.
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lifelines import KaplanMeierFitter
from lifelines.exceptions import ApproximationWarning
from lifelines.utils import median_survival_times
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ApproximationWarning)

st.set_page_config(
    page_title="Tableau de Bord Prédictif Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "Predictive_Table.csv"

REQUIRED_COLUMNS = [
    "machineID", "model", "age", "time", "event",
    "error_count", "maint_count", "volt", "rotate", "pressure", "vibration",
]

# Valeurs par défaut ; toutes surchargeables depuis la sidebar.
SEUIL_CRITIQUE_DEFAULT = 7  # jours
SEUIL_ALERTE_DEFAULT = 30  # jours
COUT_PANNE_CRITIQUE_DEFAULT = 50_000  # €
COUT_PANNE_MOYEN_DEFAULT = 20_000  # €
COUT_MAINTENANCE_PREVENTIVE_DEFAULT = 3_000  # € par intervention

COUT_HORAIRE_TECHNICIEN = 250  # € / heure
DUREE_INTERVENTION_HEURES = {3: 8, 2: 4, 1: 2}  # heures, par niveau de sévérité

RLU_FEATURES = [
    "age", "error_count", "maint_count", "failure_severity",
    "telemetry_mean", "health_score", "time",
]


# ============================== DONNEES ================================

@st.cache_data(show_spinner=False)
def load_and_preprocess(path: str) -> pd.DataFrame:
    """Charge le CSV source, valide son schéma et calcule les features dérivées."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Fichier de données introuvable : `{path}`.")
        st.stop()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans le fichier source : {', '.join(missing)}.")
        st.stop()

    df = df.rename(columns={"event": "death"})  # 1 = panne observée (convention lifelines)

    df["failure_severity"] = np.select(
        [df["error_count"] > 40, df["error_count"] > 30],
        [3, 2],
        default=1,
    )
    df["maintenance_ratio"] = df["maint_count"] / (df["age"] * 12 + 1)
    df["health_score"] = (100 - (df["error_count"] * 2 + df["maint_count"] * 1.5)).clip(0, 100)

    telemetry_cols = ["volt", "rotate", "pressure", "vibration"]
    df["telemetry_mean"] = df[telemetry_cols].mean(axis=1)
    df["telemetry_std"] = df[telemetry_cols].std(axis=1)
    df["telemetry_range"] = df[telemetry_cols].max(axis=1) - df[telemetry_cols].min(axis=1)

    return df


@st.cache_data(show_spinner=False)
def calculate_rlu_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le RLU et les probabilités de survie à 30/60/90j, par modèle.

    Le RLU est la médiane de survie Kaplan-Meier. Si l'estimation échoue
    (échantillon trop petit, non-convergence), on retombe sur la médiane
    empirique du groupe et on le signale dans `rlu_method`.
    """
    rows = []

    for model_name, group in df.groupby("model"):
        empirical_median = group["time"].median()
        rlu_val = empirical_median
        survival_30 = survival_60 = survival_90 = np.nan
        method = "empirique (fallback)"

        if len(group) >= 5:
            try:
                kmf = KaplanMeierFitter().fit(group["time"], group["death"])

                if not kmf.survival_function_.empty:
                    median_times = median_survival_times(kmf.survival_function_)
                    if hasattr(median_times, "iloc") and not median_times.empty:
                        candidate = float(median_times.iloc[0, 0])
                    elif isinstance(median_times, (int, float, np.number)):
                        candidate = float(median_times)
                    else:
                        candidate = np.nan

                    if np.isfinite(candidate):
                        rlu_val, method = candidate, "Kaplan-Meier"

                    for horizon in (30, 60, 90):
                        try:
                            pred = float(kmf.predict(horizon))
                        except Exception:
                            pred = np.nan
                        if horizon == 30:
                            survival_30 = pred
                        elif horizon == 60:
                            survival_60 = pred
                        else:
                            survival_90 = pred
            except Exception:
                pass  # on garde le fallback empirique déjà initialisé

        if not np.isfinite(rlu_val):
            rlu_val = empirical_median

        for _, row in group.iterrows():
            rows.append({
                "machineID": row["machineID"],
                "model": model_name,
                "age": row["age"],
                "time": row["time"],
                "death": row["death"],
                "error_count": row["error_count"],
                "maint_count": row["maint_count"],
                "health_score": row["health_score"],
                "failure_severity": row["failure_severity"],
                "RLU_jours": int(round(rlu_val)) if np.isfinite(rlu_val) else 0,
                "rlu_method": method,
                "survival_30": survival_30,
                "survival_60": survival_60,
                "survival_90": survival_90,
                "telemetry_mean": row["telemetry_mean"],
            })

    return pd.DataFrame(rows)


# ============================ LOGIQUE METIER ============================

def calculate_business_impact(df, seuil_critique, seuil_alerte,
                               cout_panne_critique, cout_panne_moyen,
                               cout_maintenance):
    """Chiffre l'impact financier selon les seuils et coûts choisis par l'utilisateur."""
    machines_critiques = df[df["RLU_jours"] < seuil_critique]
    machines_alerte = df[(df["RLU_jours"] >= seuil_critique) & (df["RLU_jours"] < seuil_alerte)]

    n_critiques = len(machines_critiques)
    n_alertes = len(machines_alerte)
    cout_pannes_critiques = n_critiques * cout_panne_critique
    cout_pannes_alertes = n_alertes * cout_panne_moyen

    interventions_recommandees = n_critiques + int(n_alertes * 0.5)
    cout_maintenance_preventive = interventions_recommandees * cout_maintenance
    economie_potentielle = cout_pannes_critiques + cout_pannes_alertes - cout_maintenance_preventive
    roi_maintenance = (
        economie_potentielle / cout_maintenance_preventive * 100
        if cout_maintenance_preventive > 0 else 0
    )

    return {
        "n_critiques": n_critiques,
        "n_alertes": n_alertes,
        "cout_pannes_critiques": cout_pannes_critiques,
        "cout_pannes_alertes": cout_pannes_alertes,
        "cout_maintenance_preventive": cout_maintenance_preventive,
        "economie_potentielle": economie_potentielle,
        "roi_maintenance": roi_maintenance,
    }


def generate_maintenance_schedule(df, seuil_critique, seuil_alerte):
    """Construit un planning priorisé selon les seuils choisis par l'utilisateur."""
    schedule = df.copy()

    schedule["priority_score"] = (
        100 / (schedule["RLU_jours"] + 1)
        + schedule["failure_severity"] * 20
        + schedule["error_count"] * 0.5
        - schedule["health_score"] * 0.3
    )

    def recommended_action(rlu):
        if rlu < seuil_critique:
            return "🔴 INTERVENTION URGENTE"
        if rlu < seuil_alerte:
            return "🟠 MAINTENANCE PLANIFIÉE"
        if rlu < 60:
            return "🟡 SURVEILLANCE RENFORCÉE"
        return "🟢 MAINTENANCE STANDARD"

    schedule["recommended_action"] = schedule["RLU_jours"].apply(recommended_action)
    schedule["recommended_date"] = pd.to_datetime("today") + pd.to_timedelta(
        np.clip(schedule["RLU_jours"] * 0.7, 1, 90), unit="D"
    )
    schedule["estimated_duration"] = schedule["failure_severity"].map(DUREE_INTERVENTION_HEURES)
    schedule["estimated_cost"] = schedule["estimated_duration"] * COUT_HORAIRE_TECHNICIEN
    schedule["risque_categorie"] = pd.cut(
        schedule["RLU_jours"],
        bins=[-1, seuil_critique, seuil_alerte, 60, np.inf],
        labels=["Critique", "Élevé", "Modéré", "Faible"],
    )

    return schedule.sort_values("priority_score", ascending=False)


# ============================= MODELE ML =================================

@st.cache_resource(show_spinner=False)
def train_rlu_model(rlu_df: pd.DataFrame):
    """Entraîne le Random Forest et évalue MAE/R² sur un jeu de test dédié (25%)."""
    X = rlu_df[RLU_FEATURES].fillna(0)
    y = rlu_df["RLU_jours"]

    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
    X_clean, y_clean = X[mask], y[mask]

    if len(X_clean) < 20:
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.25, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6)
    model.fit(scaler.transform(X_train), y_train)

    y_pred_test = model.predict(scaler.transform(X_test))
    mae = float(np.mean(np.abs(y_pred_test - y_test)))
    ss_res = float(np.sum((y_pred_test - y_test) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return model, scaler, mae, r2


def apply_predictions(rlu_df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Applique un modèle déjà entraîné à l'ensemble des machines (sans re-fit)."""
    df = rlu_df.copy()
    X = df[RLU_FEATURES].fillna(0)
    df["RLU_pred"] = model.predict(scaler.transform(X))
    df["pred_error"] = np.abs(df["RLU_pred"] - df["RLU_jours"])
    return df


# ============================== INTERFACE ================================

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Paramètres d'Analyse")

        seuil_critique = st.number_input(
            "Seuil critique (jours)", 1, 30, SEUIL_CRITIQUE_DEFAULT,
            help="En dessous de ce RLU, une machine est classée critique.",
        )
        seuil_alerte = st.number_input(
            "Seuil alerte (jours)", seuil_critique + 1, 90, SEUIL_ALERTE_DEFAULT,
            help="Au-delà du seuil critique et en dessous de celui-ci, la machine est en alerte.",
        )

        st.markdown("---")
        st.subheader("Coûts d'exploitation")
        cout_panne_critique = st.number_input(
            "Coût panne critique (€)", 1_000, 200_000, COUT_PANNE_CRITIQUE_DEFAULT, step=1_000
        )
        cout_panne_alerte = st.number_input(
            "Coût panne en alerte (€)", 500, 100_000, COUT_PANNE_MOYEN_DEFAULT, step=500
        )
        cout_maintenance = st.number_input(
            "Coût maintenance préventive (€)", 500, 20_000, COUT_MAINTENANCE_PREVENTIVE_DEFAULT, step=250
        )

        st.markdown("---")
        show_predictive = st.checkbox("Afficher les prédictions ML", value=True)

    return seuil_critique, seuil_alerte, cout_panne_critique, cout_panne_alerte, cout_maintenance, show_predictive


def render_kpis(rlu_df, impact, seuil_critique):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n_critiques = int((rlu_df["RLU_jours"] < seuil_critique).sum())
        st.metric("Machines Critiques", n_critiques,
                   delta=f"{n_critiques / len(rlu_df) * 100:.1f}% du parc")
    with col2:
        st.metric("Économie Potentielle", f"{impact['economie_potentielle']:,.0f} €",
                   delta=f"ROI: {impact['roi_maintenance']:.0f}%")
    with col3:
        disponibilite = rlu_df["survival_30"].mean(skipna=True) * 100
        st.metric("Disponibilité à 30j (moy. modèles KM)",
                   f"{disponibilite:.1f}%" if np.isfinite(disponibilite) else "n/d")
    with col4:
        rlu_moyen = rlu_df["RLU_jours"].mean()
        st.metric("RLU Moyen", f"{rlu_moyen:.0f} j", delta=f"{rlu_moyen - 45:.0f} j vs objectif")


def render_survival_tab(df, rlu_df):
    col1, col2 = st.columns(2)

    with col1:
        kmf_global = KaplanMeierFitter().fit(df["time"], df["death"])
        surv = kmf_global.survival_function_
        fig = go.Figure(go.Scatter(
            x=surv.index, y=surv["KM_estimate"], mode="lines",
            name="Survie globale", line=dict(width=3), fill="tozeroy",
        ))
        for h in (30, 60, 90):
            fig.add_vline(x=h, line_dash="dash", line_color="gray",
                          annotation_text=f"{h}j: {kmf_global.predict(h):.1%}")
        fig.update_layout(title="Fonction de Survie — Parc Complet",
                           xaxis_title="Temps (jours)", yaxis_title="Probabilité de survie")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        for i, (model_name, group) in enumerate(df.groupby("model")):
            kmf = KaplanMeierFitter().fit(group["time"], group["death"])
            fig.add_trace(go.Scatter(
                x=kmf.survival_function_.index, y=kmf.survival_function_["KM_estimate"],
                mode="lines", name=model_name, line=dict(width=2, color=colors[i % len(colors)]),
            ))
        fig.update_layout(title="Survie par Modèle de Machine",
                           xaxis_title="Temps (jours)", yaxis_title="Probabilité de survie")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Transparence statistique"):
        st.dataframe(
            rlu_df.groupby("model")["rlu_method"].first().rename("Méthode utilisée"),
            use_container_width=True,
        )
        st.caption(
            "Un modèle passe en fallback empirique (médiane brute) quand l'échantillon "
            "est trop petit (< 5 machines) pour un ajustement Kaplan-Meier fiable."
        )


def render_priority_tab(schedule, rlu_df, seuil_critique, seuil_alerte):
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.scatter(
            schedule, x="error_count", y="RLU_jours", color="risque_categorie",
            size="maint_count", hover_data=["machineID", "model", "age", "health_score"],
            color_discrete_map={"Critique": "red", "Élevé": "orange", "Modéré": "gold", "Faible": "green"},
            title="Matrice de Décision : Erreurs vs RLU",
        )
        fig.add_hrect(y0=0, y1=seuil_critique, line_width=0, fillcolor="red", opacity=0.1)
        fig.add_hrect(y0=seuil_critique, y1=seuil_alerte, line_width=0, fillcolor="orange", opacity=0.1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 5 Machines Prioritaires")
        top5 = rlu_df.nsmallest(5, "RLU_jours")[
            ["machineID", "model", "RLU_jours", "error_count", "health_score"]
        ]
        for _, row in top5.iterrows():
            if row["RLU_jours"] < 3:
                action = "🔴 ARRÊT IMMÉDIAT"
            elif row["RLU_jours"] < 7:
                action = "🟠 MAINTENANCE < 24h"
            else:
                action = "🟡 PLANIFIER < 7j"

            col_a, col_b = st.columns([1, 2])
            col_a.metric(f"ID {row['machineID']}", f"{row['RLU_jours']}j")
            col_b.caption(f"{row['model']} — {action}")
            st.progress(int(row["health_score"]) / 100)


def render_financial_tab(impact):
    col1, col2 = st.columns(2)

    with col1:
        couts_data = pd.DataFrame({
            "Scénario": ["Pannes Critiques", "Pannes Alertes", "Maintenance Préventive"],
            "Coût (k€)": [
                impact["cout_pannes_critiques"] / 1000,
                impact["cout_pannes_alertes"] / 1000,
                impact["cout_maintenance_preventive"] / 1000,
            ],
        })
        fig = px.bar(couts_data, x="Scénario", y="Coût (k€)", color="Scénario",
                     title="Analyse Coûts / Bénéfices", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            values=[impact["cout_pannes_critiques"], impact["cout_pannes_alertes"],
                    max(impact["economie_potentielle"], 0)],
            names=["Coût pannes critiques", "Coût pannes alertes", "Économie potentielle"],
            title="Répartition Financière",
            color_discrete_sequence=["red", "orange", "green"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("💰 Économie Annuelle Estimée", f"{impact['economie_potentielle']:,.0f} €")
        st.metric("📈 ROI sur 12 mois", f"{impact['roi_maintenance']:.0f}%")


def render_planning_tab(schedule, rlu_df):
    st.subheader("📅 Planning des 30 prochains jours")
    schedule_30j = schedule[schedule["recommended_date"] <= pd.to_datetime("today") + timedelta(days=30)]

    if schedule_30j.empty:
        st.info("Aucune intervention recommandée dans les 30 prochains jours avec les seuils actuels.")
    else:
        gantt = schedule_30j.head(15).copy()
        gantt["start"] = pd.to_datetime("today")
        gantt["end"] = gantt["recommended_date"]
        fig = px.timeline(
            gantt, x_start="start", x_end="end", y="machineID", color="recommended_action",
            hover_data=["model", "RLU_jours", "estimated_duration", "estimated_cost"],
            title="Planning des Interventions",
            color_discrete_map={
                "🔴 INTERVENTION URGENTE": "red", "🟠 MAINTENANCE PLANIFIÉE": "orange",
                "🟡 SURVEILLANCE RENFORCÉE": "gold", "🟢 MAINTENANCE STANDARD": "green",
            },
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Liste des Interventions Recommandées")
    col1, col2, col3 = st.columns(3)
    filter_model = col1.multiselect("Filtrer par modèle", options=sorted(rlu_df["model"].unique()))
    filter_risk = col2.multiselect("Filtrer par risque", options=["Critique", "Élevé", "Modéré", "Faible"])
    filter_rlu = col3.slider("Filtrer par RLU max (jours)", 0, 365, 90)

    filtered = schedule.copy()
    if filter_model:
        filtered = filtered[filtered["model"].isin(filter_model)]
    if filter_risk:
        filtered = filtered[filtered["risque_categorie"].isin(filter_risk)]
    filtered = filtered[filtered["RLU_jours"] <= filter_rlu]

    display_cols = ["machineID", "model", "RLU_jours", "health_score",
                     "recommended_action", "recommended_date", "estimated_duration", "estimated_cost"]
    st.dataframe(
        filtered[display_cols].head(20),
        use_container_width=True,
        column_config={
            "RLU_jours": st.column_config.NumberColumn(format="%d j"),
            "health_score": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "estimated_cost": st.column_config.NumberColumn(format="%d €"),
            "recommended_date": st.column_config.DateColumn(format="DD/MM/YYYY"),
        },
    )


def render_ml_expander(df, rlu_df, model, scaler, mae, r2, seuil_critique, seuil_alerte):
    with st.expander("🔮 Analyse Prédictive Avancée (Random Forest)", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Performance du Modèle (jeu de test, 25%)")
            m1, m2 = st.columns(2)
            m1.metric("Erreur Moyenne Absolue", f"{mae:.1f} jours")
            m2.metric("Score R²", f"{r2:.3f}")

            fig = px.histogram(
                rlu_df, x="pred_error", nbins=30,
                title="Distribution des Erreurs de Prédiction (parc complet)",
                labels={"pred_error": "Erreur (jours)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Simulateur de Scénario")
            with st.form("scenario_simulator"):
                age_input = st.number_input("Âge machine (années)", 1, 20, 5)
                error_input = st.number_input("Nombre d'erreurs", 0, 100, 20)
                maint_input = st.number_input("Maintenances passées", 0, 100, 15)
                severity_input = st.slider("Sévérité historique", 1, 3, 2)
                submitted = st.form_submit_button("Prédire RLU")

            if submitted:
                telemetry_mean = df["telemetry_mean"].mean()
                health_score = np.clip(100 - (error_input * 2 + maint_input * 1.5), 0, 100)
                time_estimate = age_input * 365 * 0.7

                input_data = pd.DataFrame(
                    [[age_input, error_input, maint_input, severity_input,
                      telemetry_mean, health_score, time_estimate]],
                    columns=RLU_FEATURES,
                )
                prediction = model.predict(scaler.transform(input_data))[0]
                st.metric("🔮 RLU Prédit", f"{prediction:.0f} jours")

                if prediction < seuil_critique:
                    st.error("🔴 Intervention requise à très court terme")
                    st.info("**Actions recommandées :** arrêt programmé, commande pièces critiques.")
                elif prediction < seuil_alerte:
                    st.warning(f"🟠 Planifier une maintenance sous {seuil_alerte} jours")
                    st.info("**Actions recommandées :** planification maintenance, surveillance accrue.")
                else:
                    st.success("🟢 Maintenance planifiée standard")
                    st.info("**Actions recommandées :** maintenance préventive de routine.")


def render_exports(rlu_df, schedule, impact):
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.download_button("📥 Exporter données RLU", rlu_df.to_csv(index=False),
                          "rlu_analysis.csv", "text/csv")
    col2.download_button("📅 Exporter planning", schedule.to_csv(index=False),
                          "maintenance_schedule.csv", "text/csv")

    rapport = f"""# Rapport d'Analyse Prédictive — Maintenance
Date : {pd.Timestamp.now().strftime('%d/%m/%Y')}

## Synthèse
- Machines analysées : {len(rlu_df)}
- Machines critiques : {impact['n_critiques']} ({impact['n_critiques'] / len(rlu_df) * 100:.1f}%)
- RLU moyen : {rlu_df['RLU_jours'].mean():.0f} jours
- Disponibilité à 30j (moy. modèles KM) : {rlu_df['survival_30'].mean(skipna=True) * 100:.1f}%

## Impact Financier
- Économie potentielle : {impact['economie_potentielle']:,.0f} €
- ROI maintenance préventive : {impact['roi_maintenance']:.0f}%

## Recommandations
1. Intervenir sur les {impact['n_critiques']} machines critiques.
2. Planifier la maintenance pour {impact['n_alertes']} machines en alerte.
3. Revoir la stratégie de maintenance du modèle le plus fragile : {rlu_df.groupby('model')['RLU_jours'].mean().idxmin()}.
"""
    col3.download_button("📄 Télécharger rapport", rapport, "rapport_maintenance.md", "text/markdown")


def main():
    (seuil_critique, seuil_alerte, cout_panne_critique, cout_panne_alerte,
     cout_maintenance, show_predictive) = render_sidebar()

    with st.spinner("Chargement et traitement des données..."):
        df = load_and_preprocess(DATA_PATH)
        rlu_df = calculate_rlu_by_model(df)

    model = scaler = mae = r2 = None
    if show_predictive:
        model, scaler, mae, r2 = train_rlu_model(rlu_df)
        if model is not None:
            rlu_df = apply_predictions(rlu_df, model, scaler)
        else:
            st.sidebar.warning(
                "Pas assez de données propres (post-filtrage des valeurs "
                "aberrantes) pour entraîner un modèle fiable."
            )

    impact = calculate_business_impact(
        rlu_df, seuil_critique, seuil_alerte,
        cout_panne_critique, cout_panne_alerte, cout_maintenance,
    )
    schedule = generate_maintenance_schedule(rlu_df, seuil_critique, seuil_alerte)

    st.title("Tableau de Bord Prédictif Maintenance")
    st.caption(
        f"{len(rlu_df)} machines analysées — "
        f"{(rlu_df['rlu_method'] == 'Kaplan-Meier').mean() * 100:.0f}% via Kaplan-Meier, "
        f"reste en fallback statistique (voir détail plus bas)."
    )

    render_kpis(rlu_df, impact, seuil_critique)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Analyse de Survie", "Priorisation", "💰 Impact Financier", "📋 Planning"]
    )
    with tab1:
        render_survival_tab(df, rlu_df)
    with tab2:
        render_priority_tab(schedule, rlu_df, seuil_critique, seuil_alerte)
    with tab3:
        render_financial_tab(impact)
    with tab4:
        render_planning_tab(schedule, rlu_df)

    if show_predictive and model is not None:
        render_ml_expander(df, rlu_df, model, scaler, mae, r2, seuil_critique, seuil_alerte)

    render_exports(rlu_df, schedule, impact)


if __name__ == "__main__":
    main()
