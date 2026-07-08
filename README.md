# 🏆 Simulatore Predittivo - FIFA World Cup 2026

👉 [PROVA IL SIMULATORE LIVE QUI](https://simulatore-wc2026.streamlit.app/) 👈


## 📌 Descrizione del Progetto

Questa repository contiene l'intera pipeline di Data Science sviluppata per un progetto d'esame "Simulatore Predittivo per la FIFA World Cup 2026". 

L'obiettivo del progetto è prevedere l'esito delle partite della FIFA World Cup 2026 e simulare l'intero andamento del torneo (fino alla determinazione del campione) attraverso l'utilizzo di algoritmi di Machine Learning e simulazioni stocastiche.



## 🔬 Metodologia

Il motore di previsione si basa su un **Soft Ensemble** che combina le probabilità calcolate da tre modelli ad albero avanzati:

* **Random Forest**

* **XGBoost**

* **LightGBM**



Le features principali utilizzate per l'addestramento includono:

1. **Gap prestazionale storico:** Calcolato tramite il sistema di **Rating Elo** applicato alle nazionali di calcio.

2. **Stato di forma recente:** Differenza tra le medie dei gol segnati dalle squadre nelle ultime 3 partite disputate.

3. **Fattore campo:** Una variabile binaria che identifica la presenza di una nazione ospitante in un match altrimenti considerato in "campo neutro" (Stati Uniti, Messico, Canada).



La simulazione dell'intero torneo avviene tramite il metodo **Monte Carlo**, replicando la competizione migliaia di volte per convergere verso probabilità statisticamente robuste, includendo la varianza e l'imprevedibilità tipiche del gioco reale.



## 📂 Struttura della Repository

```text

Esame\_WC\_Prediction/

│

├── Data/                       # Dataset storici (es. risultati, rating Elo, dati correnti)

├── models/                     # Modelli di ML serializzati (.pkl)

├── src/                        # Script sorgente di backend (training e configurazione JSON)

├── docs/                       # Documentazione accademica (PDF, presentazione)

├── World\_Cup\_Prediction.py     # Frontend Streamlit per l'interfaccia utente

├── config\_mondiale.json        # File di configurazione per il torneo

├── requirements.txt            # Dipendenze e librerie Python necessarie

└── README.md                   # Questo file

