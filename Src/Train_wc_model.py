# Librerie standard di Python
import os          
import time        
import warnings    

# Manipolazione dati, matematica e serializzazione
import numpy as np 
import pandas as pd 
import joblib      

# Visualizzazione dati (EDA)
import matplotlib.pyplot as plt 
import seaborn as sns           

# Machine Learning: modelli predittivi e algoritmi
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Machine Learning: cross-validation e ottimizzazione 
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight 

# Machine Learning: metriche di valutazione
from sklearn.metrics import (
    accuracy_score, 
    log_loss, 
    classification_report, 
    precision_score
)

# Impostazioni di sistema e gestione warnings
os.system('cls')
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =====================================================================
# ---------- FASE 1: CARICAMENTO DATI E FEATURE ENGINEERING -----------
# =====================================================================

results_df = pd.read_csv('Data/results.csv')
elo_df = pd.read_csv('Data/elo_ratings_wc2026.csv')

# Gestione Not Aviable Values
results_df = results_df.dropna()

# Conversione stringhe in datetime
results_df['date'] = pd.to_datetime(results_df['date'], format='%Y-%m-%d')
elo_df['snapshot_date'] = pd.to_datetime(elo_df['snapshot_date'], format='%Y-%m-%d')

# Riordinamento dei dataframe in base alla data
results_df = results_df.sort_values('date').reset_index(drop = True)
elo_df = elo_df.sort_values('snapshot_date').reset_index(drop = True)

# Estrazione colonne utili per il merge dei datasets
elo_subset = elo_df[['snapshot_date', 'country', 'rating']].rename(
    columns={'country': 'team_name', 'snapshot_date': 'elo_date'}
)

# Merge dei datasets
df_merged = pd.merge_asof(
    results_df,
    elo_subset, 
    left_on = 'date', 
    right_on = 'elo_date', 
    left_by = 'home_team',
    right_by = 'team_name',
    direction = 'backward' 
)
df_merged = df_merged.rename(columns={'rating': 'home_elo'}).drop(columns=['team_name', 'elo_date'])

df_merged = pd.merge_asof(
    df_merged, 
    elo_subset, 
    left_on='date', 
    right_on='elo_date', 
    left_by='away_team', 
    right_by='team_name', 
    direction='backward'
)
df_merged = df_merged.rename(columns={'rating': 'away_elo'}).drop(columns=['team_name', 'elo_date'])

df_merged = df_merged.dropna(subset=['home_elo', 'away_elo'])

# Calcolo della differenza di Elo tra le due squadre
df_merged['elo_diff'] = df_merged['home_elo'] - df_merged['away_elo']

# Definizione della Variabile Target (1 = Vittoria squadra Casa, 0 = Pareggio, 2 = Vittoria squadra Trasferta)
conditions = [
    (df_merged['home_score'] > df_merged['away_score']),
    (df_merged['home_score'] == df_merged['away_score']),
    (df_merged['home_score'] < df_merged['away_score'])
]
choises = [1, 0, 2] 
df_merged['target'] = np.select(conditions, choises)

# Calcolo della media dei goal segnati nelle ultime 3 partite
home_stats = df_merged[['date', 'home_team', 'home_score']].rename(columns = {'home_team': 'team', 'home_score': 'goal_scored'})
away_stats = df_merged[['date', 'away_team', 'away_score']].rename(columns = {'away_team': 'team', 'away_score': 'goal_scored'})

team_stats = pd.concat([home_stats, away_stats]).sort_values(by = ['team', 'date'])

team_stats['avg_goals_last_3'] = team_stats.groupby('team')['goal_scored'].transform(lambda x: x.rolling(window = 3, closed = 'left').mean())
team_stats['avg_goals_last_3'] = team_stats['avg_goals_last_3'].fillna(0) 

df_merged = df_merged.merge(
    team_stats[['date', 'team', 'avg_goals_last_3']].drop_duplicates(subset = ['date', 'team']),
    left_on=['date', 'home_team'], 
    right_on=['date', 'team'], 
    how='inner'
).rename(columns = {'avg_goals_last_3': 'home_avg_goals_last_3'}).drop(columns = ['team'])

df_merged = df_merged.merge(
    team_stats[['date', 'team', 'avg_goals_last_3']].drop_duplicates(subset = ['date', 'team']),
    left_on=['date', 'away_team'], 
    right_on=['date', 'team'], 
    how='inner'
).rename(columns = {'avg_goals_last_3': 'away_avg_goals_last_3'}).drop(columns = ['team'])

# Estrazione della differenza di goal medi delle ultime 3 parite 
df_merged['goals_diff_last_3'] = df_merged['home_avg_goals_last_3'] - df_merged['away_avg_goals_last_3']

# Estrazione Elo più recente per ogni squadra
latest_elo = elo_df.sort_values('snapshot_date').groupby('country').last().reset_index()
latest_elo = latest_elo[['country', 'rating']].rename(columns={'country': 'team', 'rating': 'elo'})

# Estrazione degli ultimi 3 goal reali per ogni squadra
latest_goals_seq = team_stats.sort_values('date').groupby('team')['goal_scored'].apply(lambda x: list(x.tail(3))).reset_index()
latest_goals_seq = latest_goals_seq.rename(columns={'goal_scored': 'last_3_goals'})

# Unione dei dati e salvataggio del file per Streamlit
current_stats = pd.merge(latest_elo, latest_goals_seq, on = 'team', how = 'inner')
current_stats.to_csv('Data/dati_squadre_correnti.csv', index = False)

# =====================================================================
# --------------- FASE 2: ANALISI ESPLORATIVA (EDA) -------------------
# =====================================================================

df_plot = df_merged.copy()

# Creazione della mappa per la visualizzazione degli esiti delle partite
map_target = {
    1: 'Vittoria Casa',
    0: 'Pareggio',
    2: 'Vittoria Trasferta'
}

df_plot['esito'] = df_plot['target'].map(map_target)

# Calcolo delle percentuali di esito per il grafico a barre
esito_counts = df_plot['esito'].value_counts(normalize = True) * 100

# GRAFICO A BARRE: Percentuali di Esito 
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x = esito_counts.index, 
    y = esito_counts.values, 
    hue = esito_counts.index, 
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c'], 
    legend = False
)
plt.title('Percentuale di Vittorie storiche', fontsize = 14, pad = 15)
plt.ylabel('Percentuale (%)', fontsize = 12)
plt.xlabel('Esito Partita', fontsize = 12)

# Aggiunta delle etichette sulle barre per mostrare le percentuali
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', xytext = (0, 8), 
                textcoords='offset points', fontweight='bold')

plt.ylim(0, max(esito_counts.values) + 10) 
plt.tight_layout()
plt.show()

# ISTOGRAMMI SOVRAPPOSTI: Distribuzione dei Gol 
plt.figure(figsize=(10, 6))

# Definizione del numero massimo di gol da visualizzare per semplificare il grafico
max_goals = 8
bins_range = np.arange(0, max_goals + 2) - 0.5 

sns.histplot(df_plot['home_score'].clip(upper = max_goals), 
             color = '#1f77b4', label = 'Gol in Casa', 
             kde = False, stat = 'proportion', bins = bins_range, alpha = 0.6)

sns.histplot(df_plot['away_score'].clip(upper = max_goals), 
             color = '#ff7f0e', label='Gol in Trasferta', 
             kde = False, stat = 'proportion', bins = bins_range, alpha = 0.6)

plt.title('Distribuzione dei Gol: Casa vs Trasferta', fontsize = 14, pad = 15)
plt.xlabel('Numero di Gol Segnati', fontsize = 12)
plt.ylabel('Proporzione Partite', fontsize = 12)
plt.xticks(range(0, max_goals + 1), [str(i) if i < max_goals else f"{i}+" for i in range(max_goals + 1)])
plt.xlim(-0.5, max_goals + 0.5)
plt.legend(fontsize = 12)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 2.1 Evoluzione Storica per le Top Nazionali
# -------------------------------------------------------------------

top_teams = ['Brazil', 'Germany', 'Argentina', 'France', 'Spain', 'England', 'Portugal']
df_top = elo_df[elo_df['country'].isin(top_teams)].copy()

plt.figure(figsize=(12, 6))

sns.lineplot(data = df_top, x = 'snapshot_date', y = 'rating', hue = 'country', linewidth = 2)

plt.title('Evoluzione Storica del Punteggio Elo (Top Nazionali Storiche)', fontsize = 14, pad = 15)
plt.xlabel('Anno', fontsize = 12)
plt.ylabel('Punteggio Elo', fontsize = 12)
plt.legend(title = 'Nazionale', bbox_to_anchor = (1.05, 1), loc = 'upper left')
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 2.2: Relazione tra Gap Elo e Risultato 
# ---------------------------------------------------------------------

# Definizione della variabile target binaria: 1 = Vittoria Casa, 0 = Non Vittoria Casa
df_merged['home_win'] = (df_merged['home_score'] > df_merged['away_score']).astype(int)

# Definizione dei bin per la differenza Elo e delle etichette corrispondenti
bins = [-np.inf, -300, -200, -100, -50, 0, 50, 100, 200, 300, np.inf]
labels = ['< -300', '-300 a -200', '-200 a -100', '-100 a -50', '-50 a 0', 
          '0 a 50', '50 a 100', '100 a 200', '200 a 300', '> 300']

df_merged['elo_diff_bin'] = pd.cut(df_merged['elo_diff'], bins=bins, labels=labels)

# Calcolo della probabilità di vittoria in casa per ciascun bin di differenza Elo
prob_win_df = df_merged.groupby('elo_diff_bin', observed=True)['home_win'].mean().reset_index()
prob_win_df['home_win_pct'] = prob_win_df['home_win'] * 100

# Riordino esplicito secondo l'ordine "logico" dei bin
prob_win_df['elo_diff_bin'] = pd.Categorical(prob_win_df['elo_diff_bin'], categories=labels, ordered=True)
prob_win_df = prob_win_df.sort_values('elo_diff_bin').reset_index(drop=True)

# Posizioni x intere condivise da barre, linea e punti
x_pos = np.arange(len(labels))
valori = prob_win_df['home_win_pct'].values

fig, ax = plt.subplots(figsize=(12, 6))

# Colori presi dalla colormap 'coolwarm', uno per bin
cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, len(labels)))

# Barre disegnate manualmente su posizioni intere -> nessun dodge, nessun disallineamento
ax.bar(
    x_pos, 
    valori, 
    color = colors, 
    edgecolor = 'black', 
    linewidth = 0.5, 
    width = 0.5
)

# Linea + punti neri sulle stesse identiche posizioni x
ax.plot(
    x_pos, 
    valori, 
    color = 'black', 
    marker = 'o', 
    markersize = 8, 
    linewidth = 2.5
)

ax.set_title('Probabilità di Vittoria in base al Divario Elo', fontsize = 15, pad = 15)
ax.set_xlabel('Differenza Elo (Home - Away)', fontsize = 12)
ax.set_ylabel('Probabilità di Vittoria (%)', fontsize = 12)

# Etichette percentuali sulle stesse posizioni x
for i, valore in enumerate(valori):
    ax.text(
        x = i, 
        y = valore + 2, 
        s = f"{valore:.1f}%", 
        ha = 'center', 
        va = 'bottom', 
        fontweight = 'bold', 
        fontsize = 10
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation = 45, ha = 'right')
ax.set_ylim(0, 110)
ax.set_xlim(-0.6, len(labels) - 0.4)  
ax.grid(axis = 'y', linestyle = '--', alpha = 0.4)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 2.3: Correlazione tra le Variabili (Heatmap)
# ---------------------------------------------------------------------

# Definizione delle colonne da includere nella matrice di correlazione
colonne_per_correlazione = [
    'home_score', 
    'away_score', 
    'home_elo',          
    'away_elo',          
    'elo_diff', 
    'neutral', 
    'home_win' 
]

df_corr = df_merged[colonne_per_correlazione]

# Calcolo della matrice di correlazione di Pearson
corr_matrix = df_corr.corr()

# Creazione della Heatmap 
plt.figure(figsize = (10, 8))

sns.heatmap(
    corr_matrix, 
    annot = True,         
    fmt = '.2f',          
    cmap = 'coolwarm',    
    vmin = -1, vmax=1,            
    linewidths = .5,       
)

plt.title('Matrice di Correlazione tra le Variabili (Heatmap)', fontsize = 15, pad = 20)
plt.xticks(rotation = 45, ha = 'right') 
plt.tight_layout()
plt.show()

# =====================================================================
# ---------- FASE 3: ADDESTRAMENTO E VALUTAZIONE DEL MODELLO ----------
# =====================================================================

# Trasformazione della colonna 'neutral' in intero (0 o 1)
df_merged['neutral'] = df_merged['neutral'].astype(int)

# Scelta periodo split dataset
split_date = pd.to_datetime('2024-01-01')

train_set = df_merged[df_merged['date'] < split_date]
test_set = df_merged[df_merged['date'] >= split_date]

features = ['elo_diff', 'goals_diff_last_3', 'neutral']

X_train = train_set[features]
y_train = train_set['target']

X_test = test_set[features]
y_test = test_set['target']

print(f"Numero di partite in Training Set (Prima del 2024): {len(X_train)}")
print(f"Numero di partite in Test/Validation Set (Dal 2024 in poi): {len(X_test)}")

# Definizione dello spazio dei parametri da esplorare per Random Forest
param_dist_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1, 2, 3]
}

# Definizione dello spazio dei parametri da esplorare per XGBoost
param_dist_xgb = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],           
    'colsample_bytree': [0.66, 1.0], 
    'min_child_weight': [1, 5, 10, 20]    
}

# Definizione dello spazio dei parametri da esplorare per LightGBM
param_dist_lgb = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [7, 15, 25],  
    'min_child_samples': [10, 20, 50, 100] 
}

# Splir dei dati per Time Series
tscv = TimeSeriesSplit(n_splits=3)

# Definzione dei modelli base 
modelli = {
    'RandomForest': (RandomForestClassifier(class_weight = 'balanced', random_state = 42), param_dist_rf),
    'XGBoost': (xgb.XGBClassifier(random_state = 42, n_jobs = -1), param_dist_xgb),
    'LightGBM': (lgb.LGBMClassifier(random_state = 42, class_weight = 'balanced', n_jobs = -1, verbose = -1), param_dist_lgb)
}

# Calcolo dei pesi per il campionamento bilanciato delle classi nel training set
sample_weights_train = compute_sample_weight(class_weight = 'balanced', y = y_train)

risultati = {}
modelli_addestrati = {}

for nome_modello, (modello_base, griglia_parametri) in modelli.items():
    print(f"\n--- Addestramento {nome_modello} ---")
    start_time = time.time()
    
    # Ricerca randomizzata
    search = RandomizedSearchCV(
        estimator = modello_base,
        param_distributions = griglia_parametri,
        n_iter = 50,
        cv = tscv,
        scoring = 'neg_log_loss', 
        random_state = 42,
        n_jobs = -1
    )
    
    if nome_modello == 'XGBoost':
        search.fit(X_train, y_train, sample_weight=sample_weights_train)
    else:
        search.fit(X_train, y_train)

    tempo_esecuzione = time.time() - start_time
    
    # Valutazione sul Test Set
    best_model = search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)
    y_pred = best_model.predict(X_test)
    
    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(y_test, y_pred, average = 'macro', zero_division = 0)
    
    # Salvataggio dei risultati
    risultati[nome_modello] = {
        'Log Loss': round(loss, 4),
        'Accuracy': round(acc, 4),
        'Macro Precision': round(prec, 4),
        'Tempo (s)': round(tempo_esecuzione, 2),
        'Migliori Parametri': search.best_params_
    }
    modelli_addestrati[nome_modello] = best_model
    
    print(f"Log Loss: {loss:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Tempo: {tempo_esecuzione:.2f}s")
    print(f"\nReport di Classificazione per {nome_modello}:")
    print(classification_report(y_test, y_pred, zero_division=0))

df_risultati = pd.DataFrame(risultati).T

print("\n--- Dettaglio Completo dei Migliori Parametri ---")
for modello, info in risultati.items():
    print(f"{modello}:")
    print(info['Migliori Parametri'])
    print("-" * 50)

# Calcola le probabilità di tutti e tre i modelli sul Test Set
proba_rf = modelli_addestrati['RandomForest'].predict_proba(X_test)
proba_xgb = modelli_addestrati['XGBoost'].predict_proba(X_test)
proba_lgb = modelli_addestrati['LightGBM'].predict_proba(X_test)

# Fai la media aritmetica delle probabilità (Soft Ensemble)
proba_ensemble = (proba_rf + proba_xgb + proba_lgb) / 3

# Calcola le previsioni "secche" (la classe con la probabilità media più alta)
pred_ensemble = np.argmax(proba_ensemble, axis=1)

# Valuta l'Ensemble
loss_ensemble = log_loss(y_test, proba_ensemble)
acc_ensemble = accuracy_score(y_test, pred_ensemble)

prec_ensemble = precision_score(y_test, pred_ensemble, average = 'macro', zero_division = 0)

print(f"\nEnsemble -> Log Loss: {loss_ensemble:.4f} | Accuracy: {acc_ensemble:.4f} | Macro Precision: {prec_ensemble:.4f}")
print("\nReport di Classificazione ENSEMBLE Finale:")
print(classification_report(y_test, pred_ensemble, zero_division = 0))

# Salva il modello addestrato
joblib.dump(modelli_addestrati, 'Models/modelli_wc2026.pkl')