# 🔥 UKgas — Séries Temporelles

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ukgas-time-series.streamlit.app/)

> Analyse complète de la consommation de gaz au Royaume-Uni (1960–1986) — TP Polytech Lyon MAM 4A

## 📊 Description

App Streamlit interactive réécrivant en Python l’analyse R originale de la série temporelle **UKgas** (108 observations trimestrielles, 1960–1986).

## 🧠 Méthodologie

| Section | Contenu |
|---|---|
| **Exploration** | Chronogramme, passage au log, saisonnalité par trimestre, stats descriptives |
| **Décomposition** | Additif vs multiplicatif, tests ADF/KPSS, ACF/PACF différenciée |
| **Lissage Exponentiel** | LES, LED, Holt-Winters additif & multiplicatif — comparaison RMSE/MAPE |
| **Régression Linéaire** | LM automatique → AR(4) (SARIMAX) — QQ-Plot, Ljung-Box, prévisions 5 ans |
| **SARIMA Box-Jenkins** | AR(8) vs ARMA(8,5) vs SARIMA(8,1,0)(0,1,0)₄ — validation, bilan comparatif |

## 🏆 Résultats

| Méthode | RMSE | MAPE |
|---|---|---|
| Holt-Winters | 41.85 | 6.5% |
| **Régression + AR(4)** | **33.24** | **5.6%** |
| SARIMA(8,1,0)(0,1,0)₄ | 61.79 | 9.7% |

## 🛠 Stack

`Python` · `statsmodels` · `Streamlit` · `Plotly` · `SciPy` · `Pandas`

## 📚 Référence

TP Séries Temporelles — Badreddine EL KHAMLICHI & Nadir EL KHALFIOUI — Polytech Lyon MAM 4A (2023)
