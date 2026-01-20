#  Maintenance Prédictive Industrielle — RLU & ROI

> **Objectif principal :** Transformer la maintenance curative en une stratégie **prédictive orientée ROI**, en exploitant les données industrielles et l’IA pour optimiser la disponibilité des actifs.

---

##  Problématique Business & Enjeux

Dans le secteur industriel, une panne imprévue est un gouffre financier : **jusqu'à 50 000 € de perte par incident** (arrêts de ligne, logistique d'urgence, pénalités contractuelles).

###  Le Défi : L'arbitrage du "Juste à Temps"
Le succès d'une stratégie de maintenance repose sur une précision temporelle critique :
* **Intervenir trop tôt :** Génère des coûts inutiles en remplaçant des pièces encore fonctionnelles (gaspillage de ressources).
* **Intervenir trop tard :** Provoque la panne critique, entraînant des arrêts de production coûteux et des risques sécuritaires.

###  Ma Solution
J'ai développé un **outil d'aide à la décision** interactif qui transforme la télémétrie brute en indicateurs stratégiques :
1. **Prédiction du RLU (Remaining Useful Life) :** Estimation de la durée de vie restante des équipements.
2. **Calcul du ROI (Return On Investment) :** Quantification de la rentabilité financière générée par l'anticipation des pannes.

---

## Objectifs du Projet

* **Réduire les coûts opérationnels** liés aux arrêts non planifiés.
* **Anticiper les défaillances** via des algorithmes de Machine Learning.
* **Aider à la décision** grâce à un *Health Score* métier (0–100).
* **Optimiser le planning** selon la criticité réelle des machines.
* **Mesurer l’impact financier** pour justifier l'investissement technologique.

---

##  Concepts Clés

### 🔹 RLU — Remaining Useful Life
Nombre de **jours restants avant défaillance probable** d’une machine. C'est l'indicateur central pour décider quand intervenir au moment optimal.

### 🔹 ROI — Return On Investment
Mesure la **rentabilité économique** de la solution :
$$ROI = \frac{\text{Coûts évités} - \text{Coûts de maintenance}}{\text{Coûts de maintenance}}$$

---

## Données Utilisées
* **Source :** Microsoft Azure Predictive Maintenance Dataset (Kaggle).
* **Caractéristiques :** Télémétrie (vibration, pression, rotation, voltage), historique de maintenance, compteurs d’erreurs et spécificités machines (âge, modèle).

---

##  Méthodologie & Approche Data Science

### 1️⃣ Feature Engineering Métier
* **Health Score :** Indicateur de santé synthétique (0-100) basé sur la dérive des capteurs.
* **Agrégations Temporelles :** Moyenne et écart-type glissants pour capter l'usure progressive.
* **Sévérité :** Scoring de criticité pour prioriser les interventions.

### 2️⃣ Analyse de Survie (Statistique)
* Implémentation de l'estimateur de **Kaplan-Meier**.
* Calcul des probabilités de survie à **30 / 60 / 90 jours** par modèle de machine.

### 3️⃣ Machine Learning
* **Modèle :** Random Forest Regressor (prédit le RLU en jours).
* **Performance :** Évalué via la MAE (Erreur Moyenne Absolue) et le score $R^2$.

---

## Aperçu et Interprétation du Dashboard

### 🔹 Indicateurs Clés (KPI) & ROI
![KPI Dashboard](Dashboard.png)
> **Analyse :** Ce panneau permet un pilotage financier direct. Le **ROI** permet de valider immédiatement la valeur générée par l'outil, tandis que la **Disponibilité à 30 jours** aide à la planification de la production.

### 🔹 Analyse de Survie
![Kaplan-Meier par modèle](Kaplan_models.png)
> **Analyse :** Ce graphique identifie les modèles de machines les plus fragiles statistiquement. Il permet d'adapter les contrats de maintenance selon la fiabilité réelle de chaque segment de parc.

### 🔹 Matrice de Risque & Priorisation
![RLU Matrix](RLU.png)
> **Analyse :** Croisement critique du **Health Score** et du **RLU**. Les machines en zone rouge sont signalées pour une intervention immédiate, optimisant ainsi les déplacements des techniciens.

---

##  Planning de Maintenance Intelligent
Le dashboard génère automatiquement :
* Une recommandation d'action (🔴 Urgent, 🟠 Planifié, 🟢 Standard).
* Un **planning Gantt** prévisionnel.
* Une estimation des coûts et de la durée d'intervention pour chaque actif critique.

---

## ⚙️ Installation & Lancement

### Prérequis
Python 3.9+, pandas, numpy, streamlit, scikit-learn, plotly, lifelines.

### Lancement
```bash
# Cloner le dépôt
git clone [https://github.com/Dave-kossi/predictive-maintenance-industry.git](https://github.com/Dave-kossi/predictive-maintenance-industry.git)
cd predictive-maintenance-industry

# Installer les bibliothèques
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
