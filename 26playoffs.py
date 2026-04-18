import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from nba_api.stats.endpoints import leaguegamefinder

st.set_page_config(page_title="NBA Live Intelligence Engine", layout="wide")
st.title("🏀 NBA Live Intelligence Engine")

# -----------------------------
# GAME DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def get_games():
    games = leaguegamefinder.LeagueGameFinder(season_nullable="2025-26").get_data_frames()[0]
    games = games[games["SEASON_ID"] == "22025"].copy()
    return games

# -----------------------------
# TEAM UNIVERSE
# -----------------------------
def get_teams_from_games(games):
    return sorted(games["TEAM_ABBREVIATION"].dropna().unique().tolist())

# -----------------------------
# ELO SYSTEM
# -----------------------------
def init_elo(teams):
    return {t: 1500.0 for t in teams}

def elo_expected(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def elo_update_pair(elo, team_a, team_b, team_a_won, k=18):
    expected_a = elo_expected(elo[team_a], elo[team_b])
    score_a = 1.0 if team_a_won else 0.0
    delta = k * (score_a - expected_a)

    elo[team_a] += delta
    elo[team_b] -= delta

# -----------------------------
# BUILD MODEL
# Process each GAME_ID once
# -----------------------------
def build_model(teams, games):
    elo = init_elo(teams)

    # Each NBA game appears twice in leaguegamefinder: one row for each team
    game_groups = games.groupby("GAME_ID")

    for _, game_rows in game_groups:
        if len(game_rows) < 2:
            continue

        row1 = game_rows.iloc[0]
        row2 = game_rows.iloc[1]

        team_a = row1["TEAM_ABBREVIATION"]
        team_b = row2["TEAM_ABBREVIATION"]

        if team_a not in elo or team_b not in elo:
            continue

        # Determine winner from WL
        # One row should be W, the other L
        team_a_won = row1["WL"] == "W"

        elo_update_pair(elo, team_a, team_b, team_a_won)

    return elo

# -----------------------------
# WIN PROBABILITY
# positive momentum helps team A
# -----------------------------
def live_win_prob(elo_a, elo_b, momentum=0):
    adjusted_a = elo_a + momentum * 10
    return 1 / (1 + 10 ** ((elo_b - adjusted_a) / 400))

# -----------------------------
# MARKET MODEL
# stable fake market number
# -----------------------------
def market_prob(team_a, team_b):
    # deterministic pseudo-market so it does not change every rerun
    seed = abs(hash(f"{team_a}-{team_b}")) % (2**32)
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.35, 0.65))

# -----------------------------
# PLAYOFF SIMULATION
# -----------------------------
def simulate_playoffs(teams, elo, sims=300):
    results = {t: 0 for t in teams}

    for _ in range(sims):
        bracket = teams.copy()
        np.random.shuffle(bracket)

        while len(bracket) > 1:
            next_round = []

            for i in range(0, len(bracket) - 1, 2):
                a, b = bracket[i], bracket[i + 1]
                p = live_win_prob(elo[a], elo[b])
                winner = a if np.random.rand() < p else b
                next_round.append(winner)

            if len(bracket) % 2 == 1:
                next_round.append(bracket[-1])

            bracket = next_round

        if bracket:
            results[bracket[0]] += 1

    return results

# -----------------------------
# LOAD DATA + MODEL ONCE
# -----------------------------
games = get_games()
teams = get_teams_from_games(games)
elo = build_model(teams, games)

# -----------------------------
# TOP METRICS
# -----------------------------
st.subheader("League Model Overview")

elo_df = pd.DataFrame({
    "Team": list(elo.keys()),
    "Elo": list(elo.values())
}).sort_values("Elo", ascending=False).reset_index(drop=True)

col1, col2, col3 = st.columns(3)
col1.metric("Teams Modeled", len(teams))
col2.metric("Games Loaded", games["GAME_ID"].nunique())
col3.metric("Top Team", elo_df.iloc[0]["Team"] if not elo_df.empty else "N/A")

# -----------------------------
# ELO SECTION
# -----------------------------
st.subheader("📊 Team Strength (Elo Ratings)")
st.dataframe(elo_df, use_container_width=True)
st.plotly_chart(px.bar(elo_df, x="Team", y="Elo"), use_container_width=True)

# -----------------------------
# PLAYOFF SIM
# -----------------------------
st.subheader("🏆 Championship Simulation")

sim_count = st.slider("Number of playoff simulations", 100, 5000, 500, step=100)

if st.button("Run Championship Simulation"):
    results = simulate_playoffs(teams, elo, sims=sim_count)

    champ_df = pd.DataFrame({
        "Team": list(results.keys()),
        "Title %": np.array(list(results.values())) / sim_count
    }).sort_values("Title %", ascending=False).reset_index(drop=True)

    st.dataframe(champ_df, use_container_width=True)
    st.plotly_chart(px.bar(champ_df, x="Team", y="Title %"), use_container_width=True)

# -----------------------------
# LIVE MATCHUP SIMULATOR
# Keep OUTSIDE the button block
# -----------------------------
st.subheader("📡 Live Matchup Simulator")

default_a = teams[0] if teams else None
default_b = teams[1] if len(teams) > 1 else None

with st.form("matchup_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        team_a = st.selectbox("Team A", teams, index=0 if default_a else None)

    with c2:
        team_b_index = 1 if len(teams) > 1 else 0
        team_b = st.selectbox("Team B", teams, index=team_b_index)

    with c3:
        momentum = st.slider("Momentum Swing", -10, 10, 0)

    submitted = st.form_submit_button("Update Matchup")

# render continuously even after submit
if team_a and team_b:
    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        prob = live_win_prob(elo[team_a], elo[team_b], momentum)
        market = market_prob(team_a, team_b)
        edge = prob - market

        m1, m2, m3 = st.columns(3)
        m1.metric(f"{team_a} win probability", f"{prob:.2%}")
        m2.metric(f"{team_b} win probability", f"{1 - prob:.2%}")
        m3.metric(f"{team_a} edge vs market", f"{edge:.2%}")

        compare_df = pd.DataFrame({
            "Source": ["Model", "Market"],
            f"{team_a} Probability": [prob, market]
        })

        st.plotly_chart(
            px.bar(compare_df, x="Source", y=f"{team_a} Probability"),
            use_container_width=True
        )

        st.write(f"**Model:** {team_a} {prob:.2%}")
        st.write(f"**Market:** {team_a} {market:.2%}")
        st.write(f"**Edge:** {(prob - market):.2%}")
