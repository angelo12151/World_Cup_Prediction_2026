# 🏆 Simulatore Predittivo - FIFA World Cup 2026




## 📌 Descrizione del Progetto

Questa repository contiene l'intera pipeline di Data Science sviluppata per un progetto d'esame "Simulatore Predittivo per la FIFA World Cup 2026". 

L'obiettivo del progetto è prevedere l'esito delle partite della FIFA World Cup 2026 e simulare l'intero andamento del torneo (fino alla determinazione del campione) attraverso l'utilizzo di algoritmi di Machine Learning e simulazioni stocastiche.

## 🌐 Demo Live
È possibile testare il simulatore direttamente online, senza necessità di installare alcun software in locale, tramite l'applicazione web:
👉 **[PROVA IL SIMULATORE LIVE QUI](https://simulatore-wc2026.streamlit.app/)** 👈

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

Di seguito il dettaglio di tutti i file presenti:

```text
World_Cup_Prediction_2026/
│
├── Data/                                                            # Cartella contenente i dataset di partenza e intermedi
│   ├── results.csv                                                  # Dataset storico con i risultati delle partite internazionali passate
│   ├── elo_ratings_wc2026.csv                                       # Ranking Elo aggiornato per pesare la forza delle nazionali
│   └── dati_squadre_correnti.csv                                    # Statistiche di forma e score calcolate durante il training
│
├── Docs/                                                            # Materiale accademico 
│   ├── World_Cup_Prediction_2026.pdf                                # Relazione tecnica con l'analisi delle scelte matematiche e algoritmiche
│   └── Simulatore_Predittivo_FIFA_World_Cup_2026.pptx               # Slide per l'esposizione orale e la discussione dei risultati
│
├── Models/                                                          # Cartella per l'archiviazione dell'intelligenza artificiale
│   └── modelli_wc2026.pkl                                           # Modello Ensemble pre-addestrato e serializzato pronto per l'inferenza
│
├── Src/                                                             # Cartella contenente gli script sorgente e logici
│   ├── esporta_json.py                                              # Script di setup per inizializzare le nazioni ospitanti e i gironi
│   └── Train_wc_model.py                                            # Cuore del Machine Learning: calcola le feature, allena i modelli ed esporta
│
├── config_mondiale.json                                             # Dizionario strutturato generato automaticamente con l'albero del torneo
├── requirements.txt                                                 # Elenco delle dipendenze per replicare l'ambiente di sviluppo
├── World_Cup_Prediction_2026.py                                     # File principale del Frontend: esegue l'applicazione web con Streamlit
├── .gitignore                                                       # File di configurazione 
└── README.md                                                        # Questo documento, con la documentazione e la guida di esecuzione
