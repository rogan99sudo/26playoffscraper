import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from nba_api.stats.endpoints import leaguegamefinder

st.set_page_config(page_title="NBA Live Intelligence Engine", layout="wide")
st.title("🏀 NBA Live Intelligence Engine (Fixed Version)")

# -----------------------------
# GAME DATA
# -----------------------------
@st.cache_data
def get_games():
    g = leaguegamefinder.LeagueGameFinder(season_nullable="2025-26").get_data_frames()[0]
    return g[g["SEASON_ID"] == "22025"]

# -----------------------------
# TEAM UNIVERSE (FIXED)
# -----------------------------
def get_teams_from_games(games):
    return sorted(games["TEAM_ABBREVIATION"].dropna().unique().tolist())

# -----------------------------
# ELO SYSTEM
# -----------------------------
def init_elo(teams):
    return {t: 1500 for t in teams}

def elo_update(elo, a, b, result, k=18):
    expected_a = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
    score_a = 1 if result == "W" else 0

    elo[a] += k * (score_a - expected_a)
    elo[b] -= k * (score_a - expected_a)

# -----------------------------
# BUILD MODEL (FIXED LOGIC)
# -----------------------------
def parse_matchup(matchup):
    # "LAL vs BOS" or "LAL @ BOS"
    if " vs " in matchup:
        return matchup.split(" vs ")
    if " @ " in matchup:
        return matchup.split(" @ ")
    return None, None

def build_model(teams, games):
    elo = init_elo(teams)

    for _, row in games.iterrows():
        team = row["TEAM_ABBREVIATION"]
        matchup = row["MATCHUP"]
        result = row["WL"]

        if team not in elo:
            continue

        t1, t2 = parse_matchup(matchup)
        if not t1 or not t2:
            continue

        opp = t2 if team == t1 else t1

        if opp not in elo:
            continue

        elo_update(elo, team, opp, result)

    return elo

# -----------------------------
# WIN PROBABILITY
# -----------------------------
def live_win_prob(elo_a, elo_b, momentum=0):
    return 1 / (1 + 10 ** ((elo_b - elo_a - momentum) / 400))

# -----------------------------
# MARKET MODEL (SIMULATED)
# -----------------------------
def market_prob(team_a, team_b):
    return np.random.uniform(0.35, 0.65)

# -----------------------------
# PLAYOFF SIMULATION
# -----------------------------
def simulate_playoffs(teams, elo, sims=300):
    results = {t: 0 for t in teams}

    for _ in range(sims):
        bracket = teams.copy()
        np.random.shuffle(bracket)

        if len(bracket) % 2 == 1:
            bracket = bracket[:-1]

        while len(bracket) > 1:
            next_round = []

            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i + 1]

                p = live_win_prob(elo[a], elo[b])
                winner = a if np.random.rand() < p else b
                next_round.append(winner)

            bracket = next_round

        results[bracket[0]] += 1

    return results

# -----------------------------
# RUN APP
# -----------------------------
if st.button("Run Full Live Intelligence Model"):

    games = get_games()
    teams = get_teams_from_games(games)

    elo = build_model(teams, games)

    # -----------------------------
    # ELO TABLE
    # -----------------------------
    st.subheader("📊 Team Strength (Elo Ratings)")

    df = pd.DataFrame({
        "Team": list(elo.keys()),
        "Elo": list(elo.values())
    }).sort_values("Elo", ascending=False)

    st.dataframe(df)

    st.plotly_chart(px.bar(df, x="Team", y="Elo"))

    # -----------------------------
    # PLAYOFF SIM
    # -----------------------------
    st.subheader("🏆 Championship Simulation")

    results = simulate_playoffs(teams, elo)

    champ_df = pd.DataFrame({
        "Team": list(results.keys()),
        "Title %": np.array(list(results.values())) / 300
    }).sort_values("Title %", ascending=False)

    st.dataframe(champ_df)
    st.plotly_chart(px.bar(champ_df, x="Team", y="Title %"))

    # -----------------------------
    # MATCHUP SIM
    # -----------------------------
    st.subheader("📡 Live Matchup Simulator")

    a = st.selectbox("Team A", teams)
    b = st.selectbox("Team B", teams)
    momentum = st.slider("Momentum Swing", -10, 10, 0)

    prob = live_win_prob(elo[a], elo[b], momentum)

    st.write(f"{a} win probability: {prob:.2%}")
    st.write(f"{b} win probability: {(1 - prob):.2%}")

    market = market_prob(a, b)

    st.subheader("📊 Model vs Market")
    st.write(f"Model: {a} {prob:.2%}")
    st.write(f"Market: {a} {market:.2%}")
    st.write(f"Edge: {(prob - market):.2%}")
