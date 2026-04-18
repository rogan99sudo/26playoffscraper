import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from nba_api.stats.endpoints import leaguegamefinder, leaguestandings

st.set_page_config(page_title="NBA Live Intelligence Engine", layout="wide")

st.title("🏀 NBA Live Intelligence Engine (Version 6)")

# -----------------------------
# PLAYOFF TEAMS
# -----------------------------
@st.cache_data
def get_teams():
    standings = leaguestandings.LeagueStandings().get_data_frames()[0]

    st.write(standings.columns)
    st.write(standings.head())

    # stop function early so you can see output
    return []

# -----------------------------
# GAME DATA
# -----------------------------
@st.cache_data
def get_games():
    g = leaguegamefinder.LeagueGameFinder(season_nullable="2025-26").get_data_frames()[0]
    return g[g["SEASON_ID"] == "22025"]

# -----------------------------
# PLAYER IMPACT MODEL (simplified RAPTOR-style)
# -----------------------------
def player_adjustment(team_strength, injury_factor):
    return team_strength * (1 - injury_factor)

# -----------------------------
# ADVANCED ELO
# -----------------------------
def init_elo(teams):
    return {t: 1500 for t in teams}

def elo_update(elo, a, b, result, home_adv=0, clutch=1):
    k = 18 * clutch

    expected = 1 / (1 + 10 ** ((elo[b] - elo[a] - home_adv) / 400))
    score = 1 if result == "W" else 0

    elo[a] += k * (score - expected)

# -----------------------------
# LIVE PROBABILITY ENGINE
# -----------------------------
def live_win_prob(elo_a, elo_b, momentum=0):
    return 1 / (1 + 10 ** ((elo_b - elo_a - momentum) / 400))

# -----------------------------
# MARKET VALUE COMPARISON (SIMULATED)
# -----------------------------
def market_prob(team_a, team_b):
    return np.random.uniform(0.35, 0.65)

# -----------------------------
# MODEL BUILD
# -----------------------------
def build_model(teams, games):

    elo = init_elo(teams)

    for _, row in games.iterrows():
        team = row["TEAM_ABBREVIATION"]
        matchup = row["MATCHUP"]
        result = row["WL"]

        if team not in teams:
            continue

        injury = np.random.rand() * 0.1  # simulated injury factor
        strength = player_adjustment(elo[team], injury)

        for opp in teams:
            if opp in matchup and opp != team:

                home_adv = 2 if "vs." in matchup else -2
                clutch = np.random.rand() > 0.8

                elo_update(elo, team, opp, result, home_adv, clutch)

    return elo

# -----------------------------
# SIMULATE PLAYOFF BRACKET
# -----------------------------
def simulate_playoffs(teams, elo, sims=300):

    results = {t: 0 for t in teams}

    for _ in range(sims):
        bracket = teams.copy()
        np.random.shuffle(bracket)

        while len(bracket) > 1:
            next_round = []

            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i+1]

                p = live_win_prob(elo[a], elo[b], momentum=np.random.uniform(-5, 5))

                winner = a if np.random.rand() < p else b
                next_round.append(winner)

            bracket = next_round

        results[bracket[0]] += 1

    return results

# -----------------------------
# RUN ENGINE
# -----------------------------
if st.button("Run Full Live Intelligence Model"):

    teams = get_teams()
    games = get_games()

    elo = build_model(teams, games)

    st.subheader("📊 Adjusted Team Strength (Elo + Injuries + Clutch)")

    df = pd.DataFrame({
        "Team": list(elo.keys()),
        "Strength": list(elo.values())
    }).sort_values("Strength", ascending=False)

    st.dataframe(df)

    fig = px.bar(df, x="Team", y="Strength")
    st.plotly_chart(fig)

    # -----------------------------
    # CHAMPIONSHIP SIM
    # -----------------------------
    st.subheader("🏆 Championship Probability Engine")

    results = simulate_playoffs(teams, elo)

    champ_df = pd.DataFrame({
        "Team": list(results.keys()),
        "Title %": [v / 300 for v in results.values()]
    }).sort_values("Title %", ascending=False)

    st.dataframe(champ_df)

    fig2 = px.bar(champ_df, x="Team", y="Title %")
    st.plotly_chart(fig2)

    # -----------------------------
    # LIVE MATCHUP SIM
    # -----------------------------
    st.subheader("📡 Live Matchup Simulator")

    a = st.selectbox("Team A", teams)
    b = st.selectbox("Team B", teams)

    momentum = st.slider("Momentum Swing", -10, 10, 0)

    prob = live_win_prob(elo[a], elo[b], momentum)

    st.write(f"{a} win probability: {prob:.2%}")
    st.write(f"{b} win probability: {(1-prob):.2%}")

    market = market_prob(a, b)

    st.subheader("📊 Market vs Model")
    st.write(f"Model: {a} {prob:.2%}")
    st.write(f"Market: {a} {market:.2%}")
    st.write(f"Edge: {prob - market:.2%}")
