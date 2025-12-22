# 🏭 Predictive Maintenance in Industry: Health Score & RLU
> **Statut du projet :** 🚧 En cours de développement  
> **Objectif :** Transformer la maintenance curative en stratégie prédictive pour optimiser la disponibilité industrielle.

---

## 📖 Présentation du Projet
Ce projet s'appuie sur le jeu de données **Microsoft Azure Predictive Maintenance** (Kaggle) pour fournir une solution complète de monitoring et de prédiction de pannes. L'application calcule en temps réel le **RLU (Remaining Useful Life)** et génère un planning d'intervention automatisé.

### ⚠️ Problématique Métier
Dans l'industrie, une panne non planifiée peut coûter jusqu'à **50 000 €** par incident (perte de production, main d'œuvre d'urgence, dommages collatéraux). Ce tableau de bord permet d'anticiper ces coûts et de maximiser le **ROI** des équipes de maintenance.

---

## 🎯 Objectifs Principaux
* **Réduction des coûts :** Diminuer les dépenses liées aux pannes imprévues.
* **Analyse de Résilience :** Identifier les modèles de machines les plus robustes.
* **Aide à la Décision :** Alerter les décideurs via un **Health Score** intuitif (0-100).
* **Optimisation du Planning :** Prioriser les interventions selon l'urgence réelle (RLU).

---

## 🖥️ Aperçu du Tableau de Bord

### 1. Indicateurs de Performance (KPI)
Le tableau de bord affiche immédiatement le nombre de machines en état critique et l'économie potentielle réalisable sur l'année.
> **[Barre des KPI (Machines Critiques, ROI, Disponibilité)](Dashboard.png)**

### 2. Analyse de Survie & Fiabilité
Grâce à l'estimateur de **Kaplan-Meier**, nous visualisons la probabilité de survie du parc machine au cours du temps.
> **[Courbes de survie Kaplan-Meier par modèle de machines](Kaplan_models.png)**

### 3. Matrice de Risque et Planning
Une visualisation scatter plot croisant le nombre d'erreurs et le RLU permet de cibler les machines à remplacer prioritairement.
> **[La Remaining Useful Life des machines](RLU.png)**

---

## 🧠 Méthodologie & Data Science

L'intelligence du projet repose sur un pipeline de données structuré :

1.  **Ingénierie des Variables (Feature Engineering) :**
    * `Health Score` : Algorithme personnalisé pondérant les erreurs et l'historique de maintenance.
    * `Telemetry Aggregation` : Calcul des moyennes et variations des capteurs (vibration, pression, etc.).
2.  **Modélisation Statistique :** Utilisation de la bibliothèque `lifelines` pour l'analyse de survie.
3.  **Machine Learning :** Modèle **Random Forest Regressor** pour prédire le nombre de jours restants avant la prochaine défaillance.



---

## 🛠️ Installation et Lancement

### Prérequis
* Python 3.9+
* Pandas, Streamlit, Scikit-Learn, Plotly, Lifelines

### Installation
```bash
# Cloner le dépôt
git clone [https://github.com/ton-profil/predictive-maintenance-industry.git](https://github.com/ton-profil/predictive-maintenance-industry.git)

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app_industrial_optimized.py
