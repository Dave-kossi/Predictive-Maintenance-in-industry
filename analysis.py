# app_industrial_optimized.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="Tableau de Bord Prédictif Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CONSTANTES METIER ==========
SEUIL_CRITIQUE = 7  # jours
SEUIL_ALERTE = 30   # jours
COUT_PANNE = {
    'critique': 50000,  # €
    'moyen': 20000,
    'faible': 5000
}
COUT_MAINTENANCE_PREVENTIVE = 3000  # € par intervention

# ========== CHARGEMENT & PREPROCESSING ==========
@st.cache_data
def load_and_preprocess():
    """Chargement et enrichissement des données"""
    df = pd.read_csv("Predictive_Table.csv")
    
    # Standardisation des noms
    df.rename(columns={'event': 'death'}, inplace=True)
    
    # Création de features additionnelles
    df['utilisation_rate'] = np.random.uniform(0.5, 1.0, len(df))  # Simulation
    df['failure_severity'] = np.where(df['error_count'] > 40, 3, 
                                     np.where(df['error_count'] > 30, 2, 1))
    df['maintenance_ratio'] = df['maint_count'] / (df['age'] * 12 + 1)
    df['health_score'] = 100 - (df['error_count'] * 2 + df['maint_count'] * 1.5)
    df['health_score'] = np.clip(df['health_score'], 0, 100)
    
    # Features télémétriques agrégées
    telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
    df['telemetry_mean'] = df[telemetry_cols].mean(axis=1)
    df['telemetry_std'] = df[telemetry_cols].std(axis=1)
    df['telemetry_range'] = df[telemetry_cols].max(axis=1) - df[telemetry_cols].min(axis=1)
    
    return df

def calculate_rlu_by_model(df):
    """Calcul du RLU par modèle de machine - Version ultra-robuste"""
    rlu_results = []
    
    for model_name, group in df.groupby('model'):
        # Calcul de la médiane empirique comme fallback
        empirical_median = group['time'].median()
        
        # Initialiser les valeurs par défaut
        rlu_val = empirical_median
        survival_30 = 0.5
        survival_60 = 0.3
        survival_90 = 0.1
        
        # Essayer le calcul Kaplan-Meier
        if len(group) >= 5:  # Besoin d'un minimum de données
            try:
                kmf = KaplanMeierFitter()
                kmf.fit(group['time'], group['death'])
                
                # Vérifier si la fonction de survie est valide
                if not kmf.survival_function_.empty:
                    # Récupérer la médiane de survie
                    median_times = median_survival_times(kmf.survival_function_)
                    
                    # Gestion robuste du type de retour
                    if hasattr(median_times, 'iloc'):  # C'est un DataFrame
                        if not median_times.empty:
                            rlu_val = float(median_times.iloc[0, 0])
                    elif isinstance(median_times, (int, float, np.number)):
                        rlu_val = float(median_times)
                    
                    # Calcul des probabilités de survie
                    for horizon, var_name in [(30, 'survival_30'), (60, 'survival_60'), (90, 'survival_90')]:
                        try:
                            if horizon in kmf.survival_function_.index:
                                locals()[var_name] = float(kmf.predict(horizon))
                        except:
                            pass  # Garder la valeur par défaut
            except:
                pass  # En cas d'erreur, garder les valeurs par défaut
        
        # Ajouter les résultats pour chaque machine
        for _, row in group.iterrows():
            rlu_results.append({
                'machineID': row['machineID'],
                'model': model_name,
                'age': row['age'],
                'time': row['time'],
                'death': row['death'],
                'error_count': row['error_count'],
                'maint_count': row['maint_count'],
                'health_score': row['health_score'],
                'failure_severity': row['failure_severity'],
                'RLU_jours': int(rlu_val),
                'survival_30': survival_30,
                'survival_60': survival_60,
                'survival_90': survival_90,
                'telemetry_mean': row['telemetry_mean']
            })
    
    return pd.DataFrame(rlu_results)

def calculate_business_impact(df):
    """Calcul des impacts financiers"""
    impact = {}
    
    # Machines critiques
    machines_critiques = df[df['RLU_jours'] < SEUIL_CRITIQUE]
    impact['n_critiques'] = len(machines_critiques)
    impact['cout_pannes_critiques'] = len(machines_critiques) * COUT_PANNE['critique']
    
    # Machines en alerte
    machines_alerte = df[(df['RLU_jours'] >= SEUIL_CRITIQUE) & (df['RLU_jours'] < SEUIL_ALERTE)]
    impact['n_alertes'] = len(machines_alerte)
    impact['cout_pannes_alertes'] = len(machines_alerte) * COUT_PANNE['moyen']
    
    # Économies potentielles
    interventions_recommandees = impact['n_critiques'] + int(impact['n_alertes'] * 0.5)
    impact['cout_maintenance_preventive'] = interventions_recommandees * COUT_MAINTENANCE_PREVENTIVE
    impact['economie_potentielle'] = impact['cout_pannes_critiques'] + impact['cout_pannes_alertes'] - impact['cout_maintenance_preventive']
    
    # ROI
    impact['roi_maintenance'] = (impact['economie_potentielle'] / impact['cout_maintenance_preventive']) * 100 if impact['cout_maintenance_preventive'] > 0 else 0
    
    return impact

def generate_maintenance_schedule(df):
    """Génération d'un planning de maintenance optimisé"""
    schedule = df.copy()
    
    # Calcul du score de priorité
    schedule['priority_score'] = (
        100 / (schedule['RLU_jours'] + 1) + 
        schedule['failure_severity'] * 20 +
        schedule['error_count'] * 0.5 -
        schedule['health_score'] * 0.3
    )
    
    # Recommandation d'action
    schedule['recommended_action'] = schedule.apply(
        lambda x: "🔴 INTERVENTION URGENTE" if x['RLU_jours'] < 7 
        else "🟠 MAINTENANCE PLANIFIÉE" if x['RLU_jours'] < 30 
        else "🟡 SURVEILLANCE RENFORCÉE" if x['RLU_jours'] < 60 
        else "🟢 MAINTENANCE STANDARD", axis=1
    )
    
    # Date recommandée
    schedule['recommended_date'] = pd.to_datetime('today') + pd.to_timedelta(
        np.clip(schedule['RLU_jours'] * 0.7, 1, 90), unit='D'
    )
    
    # Durée estimée
    schedule['estimated_duration'] = schedule.apply(
        lambda x: 8 if x['failure_severity'] == 3 else 4 if x['failure_severity'] == 2 else 2, axis=1
    )
    
    # Coût estimé
    schedule['estimated_cost'] = schedule['estimated_duration'] * 250  # 250€/heure
    
    return schedule.sort_values('priority_score', ascending=False)

@st.cache_resource
def train_rlu_predictor(df):
    """Entraînement d'un modèle pour prédire le RLU"""
    features = ['age', 'error_count', 'maint_count', 'failure_severity', 
                'telemetry_mean', 'health_score', 'time']
    
    X = df[features].fillna(0)
    y = df['RLU_jours']
    
    # Suppression des outliers
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    mask = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) > 10:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_scaled, y_clean)
        
        # Prédiction sur toutes les données
        X_all_scaled = scaler.transform(X.fillna(0))
        df['RLU_pred'] = model.predict(X_all_scaled)
        df['pred_error'] = np.abs(df['RLU_pred'] - df['RLU_jours'])
        
        return df, model, scaler
    
    return df, None, None

# ========== INTERFACE PRINCIPALE ==========
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Paramètres d'Analyse")
        
        horizon = st.slider("Horizon de prédiction (jours)", 7, 180, 30)
        seuil_critique = st.number_input("Seuil critique (jours)", 1, 30, SEUIL_CRITIQUE)
        seuil_alerte = st.number_input("Seuil alerte (jours)", 15, 90, SEUIL_ALERTE)
        
        st.markdown("---")
        st.subheader("Coûts d'exploitation")
        cout_panne_critique = st.number_input("Coût panne critique (€)", 1000, 100000, COUT_PANNE['critique'])
        cout_maintenance = st.number_input("Coût maintenance préventive (€)", 500, 10000, COUT_MAINTENANCE_PREVENTIVE)
        
        show_predictive = st.checkbox("Afficher les prédictions ML", value=True)
    
    # Chargement des données
    with st.spinner('Chargement et traitement des données...'):
        df = load_and_preprocess()
        rlu_df = calculate_rlu_by_model(df)
        
        if show_predictive:
            rlu_df, model, scaler = train_rlu_predictor(rlu_df)
    
    # ========== DASHBOARD PRINCIPAL ==========
    st.title("🏭 Tableau de Bord Prédictif Maintenance")
    st.markdown(f"**Analyse de {len(rlu_df)} machines** - Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_critiques = len(rlu_df[rlu_df['RLU_jours'] < seuil_critique])
        st.metric(
            label="Machines Critiques", 
            value=n_critiques,
            delta=f"{n_critiques/len(rlu_df)*100:.1f}% du parc"
        )
    
    with col2:
        impact = calculate_business_impact(rlu_df)
        st.metric(
            label="Économie Potentielle", 
            value=f"{impact['economie_potentielle']:,.0f} €",
            delta=f"ROI: {impact['roi_maintenance']:.0f}%"
        )
    
    with col3:
        disponibilite = rlu_df['survival_30'].mean() * 100
        st.metric(
            label="Disponibilité à 30j", 
            value=f"{disponibilite:.1f}%",
            delta_color="inverse" if disponibilite < 95 else "normal"
        )
    
    with col4:
        rlu_moyen = rlu_df['RLU_jours'].mean()
        st.metric(
            label="RLU Moyen", 
            value=f"{rlu_moyen:.0f} j",
            delta=f"{rlu_moyen - 45:.0f} j vs objectif"
        )
    
    # ========== VISUALISATIONS ==========
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analyse de Survie", "🎯 Priorisation", "💰 Impact Financier", "📋 Planning"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Courbe de survie globale
            kmf_global = KaplanMeierFitter()
            kmf_global.fit(df['time'], df['death'])
            surv_prob = kmf_global.survival_function_
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=surv_prob.index, 
                y=surv_prob['KM_estimate'],
                mode='lines',
                name='Survie globale',
                line=dict(width=3, color='blue'),
                fill='tozeroy'
            ))
            
            # Ajout des horizons
            for h in [30, 60, 90]:
                surv_at_h = kmf_global.predict(h)
                fig1.add_vline(x=h, line_dash="dash", line_color="gray",
                             annotation_text=f"{h}j: {surv_at_h:.1%}")
            
            fig1.update_layout(
                title="Fonction de Survie - Parc Complet",
                xaxis_title="Temps (jours)",
                yaxis_title="Probabilité de Survie",
                hovermode='x unified'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Courbes de survie par modèle
            fig2 = go.Figure()
            colors = px.colors.qualitative.Set1
            
            for i, (model_name, group) in enumerate(df.groupby('model')):
                kmf = KaplanMeierFitter()
                kmf.fit(group['time'], group['death'])
                
                fig2.add_trace(go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_['KM_estimate'],
                    mode='lines',
                    name=model_name,
                    line=dict(width=2, color=colors[i % len(colors)])
                ))
            
            fig2.update_layout(
                title="Survie par Modèle de Machine",
                xaxis_title="Temps (jours)",
                yaxis_title="Probabilité de Survie",
                hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Matrice de priorisation
            rlu_df['risque_categorie'] = pd.cut(
                rlu_df['RLU_jours'],
                bins=[0, seuil_critique, seuil_alerte, 60, np.inf],
                labels=['Critique', 'Élevé', 'Modéré', 'Faible']
            )
            
            fig3 = px.scatter(
                rlu_df,
                x='error_count',
                y='RLU_jours',
                color='risque_categorie',
                size='maint_count',
                hover_data=['machineID', 'model', 'age', 'health_score'],
                color_discrete_map={
                    'Critique': 'red',
                    'Élevé': 'orange',
                    'Modéré': 'yellow',
                    'Faible': 'green'
                },
                title="Matrice de Décision: Erreurs vs RLU"
            )
            
            fig3.add_hrect(y0=0, y1=seuil_critique, line_width=0, fillcolor="red", opacity=0.1)
            fig3.add_hrect(y0=seuil_critique, y1=seuil_alerte, line_width=0, fillcolor="orange", opacity=0.1)
            
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("Top 5 Machines Prioritaires")
            top_critiques = rlu_df.nsmallest(5, 'RLU_jours')[
                ['machineID', 'model', 'RLU_jours', 'error_count', 'health_score']
            ].copy()
            
            top_critiques['action'] = top_critiques['RLU_jours'].apply(
                lambda x: "🔴 ARRÊT IMMÉDIAT" if x < 3 
                else "🟠 MAINTENANCE < 24h" if x < 7 
                else "🟡 PLANIFIER < 7j"
            )
            
            for _, row in top_critiques.iterrows():
                with st.container():
                    col_a, col_b = st.columns([1, 2])
                    with col_a:
                        st.metric(f"ID {row['machineID']}", f"{row['RLU_jours']}j")
                    with col_b:
                        st.caption(f"{row['model']} - {row['action']}")
                    st.progress(row['health_score']/100)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Analyse financière
            couts_data = pd.DataFrame({
                'Scénario': ['Pannes Critiques', 'Pannes Alertes', 'Maintenance Préventive'],
                'Coût (k€)': [
                    impact['cout_pannes_critiques']/1000,
                    impact['cout_pannes_alertes']/1000,
                    impact['cout_maintenance_preventive']/1000
                ]
            })
            
            fig4 = px.bar(
                couts_data,
                x='Scénario',
                y='Coût (k€)',
                color='Scénario',
                title="Analyse Coûts / Bénéfices",
                text_auto='.1f'
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Répartition des coûts
            fig5 = px.pie(
                values=[impact['cout_pannes_critiques'], impact['cout_pannes_alertes'], impact['economie_potentielle']],
                names=['Coût pannes critiques', 'Coût pannes alertes', 'Économie potentielle'],
                title="Répartition Financière",
                color_discrete_sequence=['red', 'orange', 'green']
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            st.metric("💰 Économie Annuelle Estimée", f"{impact['economie_potentielle']:,.0f} €")
            st.metric("📈 ROI sur 12 mois", f"{impact['roi_maintenance']:.0f}%")
    
    with tab4:
        # Planning de maintenance
        schedule = generate_maintenance_schedule(rlu_df)
        
        st.subheader("📅 Planning des 30 prochains jours")
        
        # Filtre pour les 30 prochains jours
        schedule_30j = schedule[
            schedule['recommended_date'] <= pd.to_datetime('today') + timedelta(days=30)
        ]
        
        if not schedule_30j.empty:
            # Gantt chart
            schedule_gantt = schedule_30j.head(15).copy()
            schedule_gantt['start'] = pd.to_datetime('today')
            schedule_gantt['end'] = schedule_gantt['recommended_date']
            
            fig6 = px.timeline(
                schedule_gantt,
                x_start="start",
                x_end="end",
                y="machineID",
                color="recommended_action",
                hover_data=["model", "RLU_jours", "estimated_duration", "estimated_cost"],
                title="Planning des Interventions",
                color_discrete_map={
                    "🔴 INTERVENTION URGENTE": "red",
                    "🟠 MAINTENANCE PLANIFIÉE": "orange",
                    "🟡 SURVEILLANCE RENFORCÉE": "yellow",
                    "🟢 MAINTENANCE STANDARD": "green"
                }
            )
            fig6.update_yaxes(autorange="reversed")
            st.plotly_chart(fig6, use_container_width=True)
        
        # Liste détaillée
        st.subheader("📋 Liste des Interventions Recommandées")
        
        display_cols = ['machineID', 'model', 'RLU_jours', 'health_score', 
                       'recommended_action', 'recommended_date', 'estimated_duration', 'estimated_cost']
        
        # Filtre interactif
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            filter_model = st.multiselect("Filtrer par modèle", options=rlu_df['model'].unique())
        with col_filter2:
            filter_risk = st.multiselect("Filtrer par risque", options=['Critique', 'Élevé', 'Modéré', 'Faible'])
        with col_filter3:
            filter_rlu = st.slider("Filtrer par RLU max (jours)", 0, 365, 90)
        
        # Application des filtres
        filtered_schedule = schedule.copy()
        if filter_model:
            filtered_schedule = filtered_schedule[filtered_schedule['model'].isin(filter_model)]
        if filter_risk:
            filtered_schedule = filtered_schedule[filtered_schedule['risque_categorie'].isin(filter_risk)]
        filtered_schedule = filtered_schedule[filtered_schedule['RLU_jours'] <= filter_rlu]
        
        st.dataframe(
            filtered_schedule[display_cols].head(20),
            use_container_width=True,
            column_config={
                "RLU_jours": st.column_config.NumberColumn(format="%d j"),
                "health_score": st.column_config.ProgressColumn(format="%d%%"),
                "estimated_cost": st.column_config.NumberColumn(format="%d €"),
                "recommended_date": st.column_config.DateColumn(format="DD/MM/YYYY")
            }
        )
    
    # ========== SECTION PREDICTIVE ==========
    if show_predictive and 'RLU_pred' in rlu_df.columns:
        with st.expander("🔮 Analyse Prédictive Avancée", expanded=False):
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.subheader("Performance du Modèle")
                
                # Métriques
                mae = np.mean(np.abs(rlu_df['RLU_pred'] - rlu_df['RLU_jours']))
                r2 = 1 - np.sum((rlu_df['RLU_pred'] - rlu_df['RLU_jours'])**2) / np.sum((rlu_df['RLU_jours'] - rlu_df['RLU_jours'].mean())**2)
                
                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.metric("Erreur Moyenne Absolue", f"{mae:.1f} jours")
                with col_met2:
                    st.metric("Score R²", f"{r2:.3f}")
                
                # Distribution des erreurs
                fig_err = px.histogram(
                    rlu_df,
                    x='pred_error',
                    nbins=30,
                    title="Distribution des Erreurs de Prédiction",
                    labels={'pred_error': 'Erreur (jours)'}
                )
                st.plotly_chart(fig_err, use_container_width=True)
            
            with col_pred2:
                st.subheader("Simulateur de Scénario")
                
                with st.form("scenario_simulator"):
                    age_input = st.number_input("Âge machine (années)", 1, 20, 5)
                    error_input = st.number_input("Nombre d'erreurs", 0, 100, 20)
                    maint_input = st.number_input("Maintenances passées", 0, 100, 15)
                    severity_input = st.slider("Sévérité historique", 1, 3, 2)
                    
                    if st.form_submit_button("Prédire RLU"):
                        if model is not None and scaler is not None:
                            # Préparation des données
                            telemetry_mean = df['telemetry_mean'].mean()
                            health_score = 100 - (error_input * 2 + maint_input * 1.5)
                            health_score = max(0, min(100, health_score))
                            time_estimate = age_input * 365 * 0.7  # Estimation du temps déjà passé
                            
                            input_data = pd.DataFrame([[
                                age_input, error_input, maint_input, severity_input,
                                telemetry_mean, health_score, time_estimate
                            ]], columns=['age', 'error_count', 'maint_count', 'failure_severity',
                                       'telemetry_mean', 'health_score', 'time'])
                            
                            input_scaled = scaler.transform(input_data)
                            prediction = model.predict(input_scaled)[0]
                            
                            st.metric("🔮 RLU Prédit", f"{prediction:.0f} jours")
                            
                            # Recommandation
                            if prediction < 7:
                                st.error("🔴 Intervention requise dans la semaine")
                                st.info("**Actions recommandées:** Arrêt programmé, commande pièces critiques")
                            elif prediction < 30:
                                st.warning("🟠 Planifier maintenance sous 30 jours")
                                st.info("**Actions recommandées:** Planification maintenance, surveillance accrue")
                            else:
                                st.success("🟢 Maintenance planifiée standard")
                                st.info("**Actions recommandées:** Maintenance préventive routine")
    
    # ========== EXPORT & RAPPORT ==========
    st.markdown("---")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        csv_data = rlu_df.to_csv(index=False)
        st.download_button(
            "📥 Exporter données RLU",
            csv_data,
            "rlu_analysis.csv",
            "text/csv"
        )
    
    with col_exp2:
        schedule_csv = schedule.to_csv(index=False)
        st.download_button(
            "📅 Exporter planning",
            schedule_csv,
            "maintenance_schedule.csv",
            "text/csv"
        )
    
    with col_exp3:
        # Rapport synthétique
        rapport = f"""
        # Rapport d'Analyse Prédictive - Maintenance
        Date: {datetime.now().strftime('%d/%m/%Y')}
        
        ## Synthèse
        - Machines analysées: {len(rlu_df)}
        - Machines critiques: {impact['n_critiques']} ({impact['n_critiques']/len(rlu_df)*100:.1f}%)
        - RLU moyen: {rlu_df['RLU_jours'].mean():.0f} jours
        - Disponibilité à 30j: {rlu_df['survival_30'].mean()*100:.1f}%
        
        ## Impact Financier
        - Économie potentielle: {impact['economie_potentielle']:,.0f} €
        - ROI maintenance préventive: {impact['roi_maintenance']:.0f}%
        
        ## Recommandations
        1. Intervenir sur les {impact['n_critiques']} machines critiques
        2. Planifier maintenance pour {impact['n_alertes']} machines à risque
        3. Réviser stratégie maintenance modèle {rlu_df.groupby('model')['RLU_jours'].mean().idxmin()}
        """
        
        st.download_button(
            "📄 Télécharger rapport",
            rapport,
            "rapport_maintenance.md",
            "text/markdown"
        )

if __name__ == "__main__":
    main()