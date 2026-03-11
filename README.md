# 🔥 UKgas — Analyse de Séries Temporelles

> Analyse complète de la consommation trimestrielle de gaz au Royaume-Uni (1960–1986)  
> TP Séries Temporelles · Polytech Lyon MAM 4A · 2023

## 📊 Description

App Streamlit interactive réécrivant en Python l'analyse R originale de la série temporelle **UKgas**  
(108 observations trimestrielles, 1960–1986).

## 🧠 Démarche Box-Jenkins — A → Z

| Étape | Onglet | Contenu |
|---|---|---|
| 1 | **Exploration** | Chronogramme brut + log, saisonnalité par trimestre, évolution annuelle |
| 2 | **Décomposition** | Tendance / saisonnalité / résidus, tests ADF, ACF/PACF différenciée |
| 3 | **Lissage Exponentiel** | LES, LED, Holt-Winters additif & multiplicatif — RMSE/MAPE |
| 4 | **Régression + AR(4)** | OLS → résidus autocorrélés → SARIMAX(4,0,0) + Ljung-Box + prévisions 5 ans |
| 5 | **SARIMA Box-Jenkins** | AR(8) vs ARMA(8,5) vs SARIMA(8,1,0)(0,1,0)₄ — validation + bilan comparatif |

## 🏆 Résultats (jeu de test 1986)

| Méthode | RMSE | MAPE |
|---|---|---|
| Holt-Winters | ~42 | ~6.5% |
| **Régression + AR(4)** | **~33** | **~5.6%** |
| SARIMA(8,1,0)(0,1,0)₄ | ~62 | ~9.7% |

## 🛠 Stack

`Python` · `statsmodels` · `Streamlit` · `Plotly` · `SciPy` · `Pandas` · `NumPy`

## 🚀 Lancement local

```bash
pip install -r requirements.txt
streamlit run app.py
```

---
*Badreddine EL KHAMLICHI · Polytech Lyon MAM 4A · 2023*
