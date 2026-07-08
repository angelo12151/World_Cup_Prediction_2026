# Librerie standard di Python
import os
import warnings
import ast
import json
from itertools import combinations
from collections import Counter

# Manipolazione dati, matematica e serializzazione
import numpy as np
import pandas as pd
import joblib

# Sviluppo Web Application (Streamlit)
import streamlit as st

# Visualizzazione dati (Web / Interattiva)
import altair as alt

# Impostazioni di sistema e gestione warnings
os.system('cls')
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# =====================================================================
# ------------- CARICAMENTO CONFIGURAZIONI (JSON) ---------------------
# =====================================================================

try:
    with open('config_mondiale.json', 'r') as f:
        config = json.load(f)
        
    host_nations = config['host_nations']
    latest_elo = config['latest_elo']
    gironi_ufficiali = config['gironi_ufficiali']
    
    partite_reali = {}
    for key, value in config['partite_reali'].items():
        team_home, team_away = key.split('|')
        partite_reali[(team_home, team_away)] = tuple(value)

except FileNotFoundError:
    st.error("⚠️ File config_mondiale.json non trovato.")
    st.stop()

# =====================================================================
# ----------- LOGICA DI AGGIORNAMENTO LIVE E CACHING ------------------
# =====================================================================

@st.cache_data
def prepara_dati_live(df_storico, partite_giocate, nazioni_ospitanti):
    """
    Assimila i risultati reali in corso d'opera per aggiornare dinamicamente 
    lo stato di forma delle squadre.
    Restituisce le medie gol aggiornate sulle ultime tre partite disputate.
    """    
    live_elo = latest_elo.copy()

    # Inizializzazione storico gaol 
    live_goals = {}
    for _, row in df_storico.iterrows():
        try:
            live_goals[row['team']] = ast.literal_eval(row['last_3_goals'])
        except (ValueError, SyntaxError):
            live_goals[row['team']] = [1, 1, 1]
    
    # Aggiornamento con partite giocate
    for (home_team, away_team), (home_score, away_score) in partite_giocate.items():
        
        # Aggiornamento storico gol
        if home_team in live_goals:
            live_goals[home_team].append(home_score)
            live_goals[home_team] = live_goals[home_team][-3:]
            
        if away_team in live_goals:
            live_goals[away_team].append(away_score)
            live_goals[away_team] = live_goals[away_team][-3:]
            
    # Calcolo media goal attuale per ogni squadra
    live_avg_goals = {team: np.mean(goals) for team, goals in live_goals.items()}

    return live_elo, live_avg_goals

@st.cache_resource
def carica_risorse():
    """
    Carica in memoria gli oggetti predittivi (modelli di Machine Learning serializzati)
    e il dataset strutturale contenente le statistiche di partenza delle formazioni. 
    """

    modelli = joblib.load('modelli_wc2026.pkl')
    df_squadre = pd.read_csv('dati_squadre_correnti.csv') 
    
    return modelli, df_squadre

# =====================================================================
# --------------------------- MONDIALE 2026 ---------------------------
# =====================================================================

def calcola_nuovo_elo_sim(elo_home, elo_away, outocome, is_knockout = False):
    """
    Aggiorna i punteggi Elo delle due squadre in base all'esito della partita.
    Utilizza un fattore K dinamico: 45 per le partite a eliminazione diretta
    e 30 per le partite di girone, per riflettere l'importanza dell'evento.
    """
        
    K = 45 if is_knockout else 30

    expected_home = 1 / (1 + 10 ** (-(elo_home - elo_away) / 400))
    expected_away = 1 - expected_home

    if outocome == 1:
        w_home, w_away = 1, 0
    elif outocome == 2:
        w_home, w_away = 0, 1
    else:
        w_home, w_away = 0.5, 0.5

    elo_home_new = elo_home + K * (w_home - expected_home)
    elo_away_new = elo_away + K * (w_away - expected_away)

    return elo_home_new, elo_away_new

def simulate_match_ensemble(home_team, away_team, modelli_dict, current_elo_dict, live_avg_goals, is_knockout = False):
    """
    Simula l'esito di una partita tra due squadre utilizzando un ensemble di modelli. 
    Gestisce la varianza stocastica dell'Elo,
    l'eventuale scambio casa/trasferta per le nazioni ospitanti e calcola
    la distribuzione di probabilità finale per l'esito della gara.
    """

    # Gestione squadra in casa
    scambio_effettuato = False

    if away_team in host_nations and home_team not in host_nations:
        vero_home, vero_away = away_team, home_team
        scambio_effettuato = True
    else:
        vero_home, vero_away = home_team, away_team

    is_neutral = 0 if vero_home in host_nations or vero_away in host_nations else 1

    elo_h_base = current_elo_dict.get(vero_home, 1500)
    elo_a_base = current_elo_dict.get(vero_away, 1500)

    # Inserimento della varianza stocastica 
    elo_h_sim = np.random.normal(loc = elo_h_base, scale = 75)
    elo_a_sim = np.random.normal(loc = elo_a_base, scale = 75)

    elo_diff = elo_h_sim - elo_a_sim
    goals_diff = live_avg_goals.get(vero_home, 1.0) - live_avg_goals.get(vero_away, 1.0)

    X_match = pd.DataFrame([[elo_diff, goals_diff, is_neutral]], columns=['elo_diff', 'goals_diff_last_3', 'neutral'])

    # Soft Ensemble
    p_rf = modelli_dict['RandomForest'].predict_proba(X_match)[0]
    p_xgb = modelli_dict['XGBoost'].predict_proba(X_match)[0]
    p_lgb = modelli_dict['LightGBM'].predict_proba(X_match)[0]
    probs = (p_rf + p_xgb + p_lgb) / 3

    if is_knockout:
        p_A = probs[1]
        p_B = probs[2]
        probs = [0.0, p_A / (p_A + p_B), p_B / (p_A + p_B)]

    probs = np.array(probs)
    probs = probs / np.sum(probs)

    outcome = np.random.choice([0, 1, 2], p=probs)

    new_elo_h, new_elo_a = calcola_nuovo_elo_sim(elo_h_base, elo_a_base, outcome, is_knockout)

    if scambio_effettuato:
        current_elo_dict[away_team] = new_elo_h
        current_elo_dict[home_team] = new_elo_a
        final_outcome = 2 if outcome == 1 else 1 if outcome == 2 else 0
    else:
        current_elo_dict[home_team] = new_elo_h
        current_elo_dict[away_team] = new_elo_a
        final_outcome = outcome

    return final_outcome

def simulate_group(nome_girone, teams, modelli_dict, current_elo_dict, live_avg_goals):
    """
    Simula tutti gli incontri di un girone (round-robin) avvalendosi del modello ensemble.
    Assegna i punti canonici (3 per la vittoria, 1 per il pareggio, 0 per la sconfitta) 
    e restituisce la classifica finale ordinata. In caso di parità di punteggio, 
    utilizza il rating Elo corrente come criterio matematico di spareggio.
    """
     
    # Inizializzazione punti a 0 per ogni squadra
    punti = {team: 0 for team in teams}

    # Generazione delle combinazioni possibili di incontri
    matches = list(combinations(teams, 2))
    
    # Simulazione delle partite
    for team1, team2 in matches:
        
        # CORREZIONE: Qui passiamo correttamente live_avg_goals come 5° parametro!
        esito = simulate_match_ensemble(team1, team2, modelli_dict, current_elo_dict, live_avg_goals)
        
        if esito == 1:
            punti[team1] += 3       
        elif esito == 2:
            punti[team2] += 3       
        else:
            punti[team1] += 1       
            punti[team2] += 1
            
    # Ordinamento della classifica
    classifica_ordinata = sorted(punti.items(), key = lambda x: (x[1], current_elo_dict.get(x[0], 0)), reverse=True)
    
    return classifica_ordinata

def simulate_tournament_phase(modelli_dict, base_elo, live_avg_goals):
    """
    Gestisce l'intera logica di simulazione del torneo: simula la fase a gironi,
    effettua il ripescaggio delle terze classificate in base a punti ed Elo,
    costruisce il tabellone a eliminazione diretta e procede con i turni fino
    alla determinazione del campione finale.
    """
    
    current_tournament_elo = base_elo.copy()
    
    # FASE 1: GIRONI
    prime_classificate = []
    seconde_classificate = []
    terze_classificate = []
    
    # Simulazione separata dei gironi
    for group_name, teams in gironi_ufficiali.items():
        
        classifica = simulate_group(group_name, teams, modelli_dict, current_tournament_elo, live_avg_goals)
        
        squadra_prima = classifica[0][0]
        squadra_seconda = classifica[1][0]
        squadra_terza = classifica[2][0]
        punti_terza = classifica[2][1]
        
        prime_classificate.append((squadra_prima, group_name))
        seconde_classificate.append((squadra_seconda, group_name))
        
        dati_terza = {
            'team': (squadra_terza, group_name),
            'punti': punti_terza,
            'elo': current_tournament_elo.get(squadra_terza, 0)
        }
        terze_classificate.append(dati_terza)

    # Ripescaggio delle terze per punti e poi per Elo
    terze_ordinate = sorted(terze_classificate, key=lambda x: (x['punti'], x['elo']), reverse=True)
    migliori_8_terze = []
    for item in terze_ordinate[:8]:
        migliori_8_terze.append(item['team'])
    
    # FASE 2: CREAZIONE TABELLONE (SEDICESIMI)
    accoppiamenti = []
    
    prime_disponibili = list(prime_classificate)
    seconde_disponibili = list(seconde_classificate)
    terze_disponibili = list(migliori_8_terze)
    
    # Associazione delle terze con le prime 8 prime 
    for terza in terze_disponibili:
        nome_t, girone_t = terza
        for i, prima in enumerate(prime_disponibili):
            nome_p, girone_p = prima
            if girone_p != girone_t:
                accoppiamenti.append(nome_p)
                accoppiamenti.append(nome_t)
                prime_disponibili.pop(i) 
                break 
                
    # Associazione delle restanti 4 prime con 4 seconde 
    for prima in prime_disponibili:
        nome_p, girone_p = prima
        for i, seconda in enumerate(seconde_disponibili):
            nome_s, girone_s = seconda
            if girone_p != girone_s:
                accoppiamenti.append(nome_p)
                accoppiamenti.append(nome_s)
                seconde_disponibili.pop(i)
                break
                
    # Associazione delle restanti 8 seconde tra di loro
    while len(seconde_disponibili) > 1:
        squadra1 = seconde_disponibili.pop(0) 
        accoppiamento_trovato = False
        
        for i, squadra2 in enumerate(seconde_disponibili):
            if squadra1[1] != squadra2[1]: 
                accoppiamenti.append(squadra1[0])
                accoppiamenti.append(squadra2[0])
                seconde_disponibili.pop(i)
                accoppiamento_trovato = True
                break
                
        # Se non c'è scelta, vengono accoppiate le prime due rimaste
        if not accoppiamento_trovato:
            squadra2 = seconde_disponibili.pop(0)
            accoppiamenti.append(squadra1[0])
            accoppiamenti.append(squadra2[0])
            
    # FASE 3: ELIMINAZIONE DIRETTA
    
    def gioca_turno_semplice(squadre_in_gara):
        """
        Prende una lista di squadre, le accoppia a due a due e ne simula lo scontro
        diretto utilizzando il modello ensemble. Restituisce una lista contenente
        solo le squadre vincitrici di ciascun match.
        """
        
        squadre_vincenti = []

        for i in range(0, len(squadre_in_gara), 2):
            team_a = squadre_in_gara[i]
            team_b = squadre_in_gara[i+1]
            
            esito = simulate_match_ensemble(team_a, team_b, modelli_dict, current_tournament_elo, live_avg_goals, is_knockout=True)
            
            if esito == 1:
                squadre_vincenti.append(team_a)
            else:
                squadre_vincenti.append(team_b)
                
        return squadre_vincenti
    
    sedicesimi_vincenti = gioca_turno_semplice(accoppiamenti)

    ottavi_vincenti = gioca_turno_semplice(sedicesimi_vincenti)

    quarti_vincenti = gioca_turno_semplice(ottavi_vincenti)

    semifinali_vincenti = gioca_turno_semplice(quarti_vincenti)

    campione = gioca_turno_semplice(semifinali_vincenti)
    
    return campione[0]

# =====================================================================
# ----------------------- INTERFACCIA STREAMLIT -----------------------
# =====================================================================

st.set_page_config(
    page_title = "Oracolo Mondiali 2026",
    page_icon = "🏆",
    layout = "wide"
)

st.markdown("""
<style>
h1 a, h2 a, h3 a, h4 a {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

try:
    modelli_dict, df_squadre_base = carica_risorse()
    elo_pre_torneo = dict(zip(df_squadre_base['team'], df_squadre_base['elo']))

    live_elo, live_avg_goals = prepara_dati_live(
        df_storico = df_squadre_base, 
        partite_giocate = partite_reali, 
        nazioni_ospitanti = host_nations
    )
except Exception as e:
    st.error(f"⚠️ Errore nel caricamento dei file o nell'aggiornamento live: {e}")
    st.stop()

st.title("🏆 Simulatore Predittivo - FIFA World Cup 2026")
st.markdown("Motore Live aggiornato in tempo reale | Modello Ensemble (RF + XGB + LGBM)")

st.sidebar.header("Vuoi sbancare con le bet?💸\nCi pensa Angelo!😜")
modalita = st.sidebar.radio(
    "Scegli la modalità:",
    ["⚽ Previsione Singolo Match", "🏆🥇 Simulazione Torneo (Monte Carlo)"]
)

# =====================================================================
# MODALITÀ 1: SINGOLO MATCH
# =====================================================================
if modalita == "⚽ Previsione Singolo Match":
    st.header("Previsione Singolo Match")
    
    # Estrazione elenco ordinato delle squadre dai dati aggiornati live
    elenco_squadre = sorted(list(live_elo.keys()))
    
    col1, col2 = st.columns(2)

    with col1:
        squadra_A = st.selectbox("Squadra A: ", elenco_squadre, index = None, placeholder = "Scegli la prima squadra...")
    with col2:
        squadra_B = st.selectbox("Squadra B: ", elenco_squadre, index = None, placeholder = "Scegli la seconda squadra...")
        
    is_knockout = st.toggle("Partita a eliminazione diretta (Supplementari/Rigori inclusi - Esclude il pareggio)")

    if st.button("Calcola Probabilità", type="primary"):
        if not squadra_A or not squadra_B:
            st.warning("⚠️ Seleziona entrambe le squadre per procedere!")
        elif squadra_A == squadra_B:
            st.warning("⚠️ Seleziona due squadre diverse! Una squadra non può giocare contro se stessa.")
        else:
            elo_A = live_elo.get(squadra_A, 1500)
            elo_B = live_elo.get(squadra_B, 1500)
            
            goals_A = live_avg_goals.get(squadra_A, 1.0)
            goals_B = live_avg_goals.get(squadra_B, 1.0)
            
            elo_diff = elo_A - elo_B
            goals_diff = goals_A - goals_B
            
            # Gestione squadra di casa
            is_neutral = 1
            if squadra_A in host_nations or squadra_B in host_nations:
                is_neutral = 0
            
            # Costruzione del vettore di input con i nomi corretti delle colonne
            X_input = pd.DataFrame(
                [[elo_diff, goals_diff, is_neutral]], 
                columns=['elo_diff', 'goals_diff_last_3', 'neutral']
            )
            
            # Inferenza con Soft Ensemble (Media aritmetica delle probabilità dei 3 modelli)
            prob_rf = modelli_dict['RandomForest'].predict_proba(X_input)[0]
            prob_xgb = modelli_dict['XGBoost'].predict_proba(X_input)[0]
            prob_lgb = modelli_dict['LightGBM'].predict_proba(X_input)[0]
            
            prob_ensemble = (prob_rf + prob_xgb + prob_lgb) / 3
            
            p_X = prob_ensemble[0]  
            p_A = prob_ensemble[1]  
            p_B = prob_ensemble[2]  
            
            # Gestione eliminazione diretta
            if is_knockout:
                p_A_norm = p_A / (p_A + p_B)
                p_B_norm = p_B / (p_A + p_B)
                p_A, p_B = p_A_norm, p_B_norm
                p_X = 0.0

            # Visualizzazione risultati 
            neutral_text = "No (Paese ospitante in campo)" if is_neutral == 0 else "Sì"

            st.success(f"""
            📊 **Parametri calcolati dall'algoritmo:**
            - **Differenza Elo:** {elo_diff:+.1f}
            - **Differenza Media Gol (Ultime 3):** {goals_diff:+.2f}
            - **Partita in Campo Neutro:** {neutral_text}
            """)
            
            # Visualizzazione delle metriche percentuali in 3 colonne distintive
            if is_knockout:
                c1, c2 = st.columns(2)
                c1.metric(label=f"Vittoria {squadra_A}", value=f"{p_A*100:.1f}%")
                c2.metric(label=f"Vittoria {squadra_B}", value=f"{p_B*100:.1f}%")
                
                chart_data = pd.DataFrame({
                    'Esito': [f'Vittoria {squadra_A}', f'Vittoria {squadra_B}'],
                    'Probabilità': [p_A, p_B]
                })
                sort_order = [f'Vittoria {squadra_A}', f'Vittoria {squadra_B}']
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric(label=f"Vittoria {squadra_A}", value=f"{p_A*100:.1f}%")
                c2.metric(label="Pareggio", value=f"{p_X*100:.1f}%")
                c3.metric(label=f"Vittoria {squadra_B}", value=f"{p_B*100:.1f}%")
                
                chart_data = pd.DataFrame({
                    'Esito': [f'Vittoria {squadra_A}', 'Pareggio', f'Vittoria {squadra_B}'],
                    'Probabilità': [p_A, p_X, p_B]
                })
                sort_order = [f'Vittoria {squadra_A}', 'Pareggio', f'Vittoria {squadra_B}']
            
            # Generazione del grafico 
            grafico = alt.Chart(chart_data).mark_bar(color='#1f77b4').encode(
                x=alt.X(
                    'Esito', 
                    sort=sort_order, 
                    axis=alt.Axis(labelAngle=0, title=None, labelLimit=0) 
                ), 
                y=alt.Y(
                    'Probabilità', 
                    axis=alt.Axis(format='%', title='Probabilità Stimata')
                ) 
            ).properties(height=350) 
            
            st.altair_chart(grafico, use_container_width=True)

# =====================================================================
# MODALITÀ 2: SIMULAZIONE TORNEO
# =====================================================================
elif modalita == "🏆🥇 Simulazione Torneo (Monte Carlo)":
    st.header("Simulazione Torneo Globale")
    
    n_simulazioni = st.slider("Numero di Mondiali da simulare: ", min_value = 100, max_value=5000, value=1000, step=100)

    st.markdown(f"Il modello simulerà l'intero mondiale {n_simulazioni} volte, valutando probabilità, fluttuazioni di forma, scontri diretti e il formato a 48 squadre.")
    
    if st.button("Avvia Oracolo", type="primary"):

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        vittorie_mondiali = []
        
        # Simulazione Monte Carlo
        for i in range(n_simulazioni):
            if i % 10 == 0 or i == n_simulazioni - 1:
                progress_bar.progress((i + 1) / n_simulazioni)
                status_text.text(f"Simulazione {i} di {n_simulazioni} in corso...")
                
            campione = simulate_tournament_phase(modelli_dict, elo_pre_torneo, live_avg_goals)
            vittorie_mondiali.append(campione)
            
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ {n_simulazioni} mondiali simulati con successo!")
        
        # --- CALCOLO RISULTATI ---
        conteggio_vittorie = Counter(vittorie_mondiali)
        classifica_probabilita = [(squadra, (vittorie / n_simulazioni) * 100) for squadra, vittorie in conteggio_vittorie.items()]
        classifica_probabilita.sort(key=lambda x: x[1], reverse=True)
        
        # Estrazione della Top 15 per visualizzarla
        top_15 = classifica_probabilita[:15]
        
        # --- VISUALIZZAZIONE GRAFICA ---
        st.subheader("La Top 15 delle Favorite")
        
        df_risultati = pd.DataFrame(top_15, columns=['Nazionale', 'Probabilità di Vittoria (%)'])
        
        # Grafico a barre orizzontali Altair
        grafico_torneo = alt.Chart(df_risultati).mark_bar(cornerRadiusEnd = 4).encode(
            x=alt.X('Probabilità di Vittoria (%):Q', title = 'Probabilità Stimata', axis = alt.Axis(format='.1f')),
            y=alt.Y('Nazionale:N', sort = '-x', title = None),
            color=alt.Color('Probabilità di Vittoria (%):Q', scale = alt.Scale(scheme = 'viridis'), legend = None),
            tooltip=['Nazionale', alt.Tooltip('Probabilità di Vittoria (%):Q', format = '.2f')]
        ).properties(height = 500)
        
        # Aggiunta delle etichette testuali alla fine di ogni barra
        text_labels = grafico_torneo.mark_text(
            align = 'left', baseline = 'middle', dx = 3, fontWeight = 'bold', color = 'white'
        ).encode(
            text = alt.Text('Probabilità di Vittoria (%):Q', format = '.1f')
        )
        
        st.altair_chart(grafico_torneo + text_labels, use_container_width = True)
        
        