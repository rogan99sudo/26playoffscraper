import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nba_api.stats.endpoints import leaguegamefinder

st.set_page_config(
    page_title="NBA Live Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏀 NBA Live Intelligence Engine")
st.caption("Elo ratings, title simulations, and live matchup intelligence")

# =========================================================
# DATA
# =========================================================
@st.cache_data(show_spinner=False)
def get_games():
    games = leaguegamefinder.LeagueGameFinder(season_nullable="2025-26").get_data_frames()[0]
    games = games[games["SEASON_ID"] == "22025"].copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    return games

def get_teams_from_games(games):
    return sorted(games["TEAM_ABBREVIATION"].dropna().unique().tolist())

# =========================================================
# ELO
# =========================================================
def init_elo(teams):
    return {team: 1500.0 for team in teams}

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def elo_update_pair(elo, team_a, team_b, team_a_won, k=18):
    ea = expected_score(elo[team_a], elo[team_b])
    sa = 1.0 if team_a_won else 0.0
    delta = k * (sa - ea)

    elo[team_a] += delta
    elo[team_b] -= delta

def build_model(teams, games, k_factor=18):
    elo = init_elo(teams)
    history = []

    grouped = games.groupby("GAME_ID", sort=False)

    for game_id, rows in grouped:
        if len(rows) < 2:
            continue

        rows = rows.sort_values("TEAM_ABBREVIATION").reset_index(drop=True)
        row_a = rows.iloc[0]
        row_b = rows.iloc[1]

        team_a = row_a["TEAM_ABBREVIATION"]
        team_b = row_b["TEAM_ABBREVIATION"]

        if team_a not in elo or team_b not in elo:
            continue

        pre_a = elo[team_a]
        pre_b = elo[team_b]

        team_a_won = row_a["WL"] == "W"
        elo_update_pair(elo, team_a, team_b, team_a_won, k=k_factor)

        history.append({
            "GAME_ID": game_id,
            "GAME_DATE": row_a["GAME_DATE"],
            "TEAM_A": team_a,
            "TEAM_B": team_b,
            "TEAM_A_WON": team_a_won,
            "PRE_ELO_A": pre_a,
            "PRE_ELO_B": pre_b,
            "POST_ELO_A": elo[team_a],
            "POST_ELO_B": elo[team_b],
        })

    history_df = pd.DataFrame(history)
    return elo, history_df

# =========================================================
# MODEL HELPERS
# =========================================================
def live_win_prob(elo_a, elo_b, momentum=0, home_edge=0):
    adjusted_a = elo_a + momentum * 10 + home_edge
    return 1 / (1 + 10 ** ((elo_b - adjusted_a) / 400))

def stable_market_prob(team_a, team_b):
    seed = abs(hash("market-" + "-".join(sorted([team_a, team_b])))) % (2**32)
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.38, 0.62))

def simulate_playoffs(teams, elo, sims=1000):
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

def team_form_from_history(history_df, team, last_n=10):
    if history_df.empty:
        return pd.DataFrame()

    rows = []

    for _, r in history_df.iterrows():
        if r["TEAM_A"] == team:
            rows.append({
                "GAME_DATE": r["GAME_DATE"],
                "TEAM": team,
                "PRE_ELO": r["PRE_ELO_A"],
                "POST_ELO": r["POST_ELO_A"],
                "OPP": r["TEAM_B"],
                "RESULT": "W" if r["TEAM_A_WON"] else "L"
            })
        elif r["TEAM_B"] == team:
            rows.append({
                "GAME_DATE": r["GAME_DATE"],
                "TEAM": team,
                "PRE_ELO": r["PRE_ELO_B"],
                "POST_ELO": r["POST_ELO_B"],
                "OPP": r["TEAM_A"],
                "RESULT": "L" if r["TEAM_A_WON"] else "W"
            })

    df = pd.DataFrame(rows).sort_values("GAME_DATE")
    if df.empty:
        return df
    return df.tail(last_n)

def head_to_head_summary(games, team_a, team_b):
    mask = (
        games["MATCHUP"].str.contains(team_a, na=False) &
        games["MATCHUP"].str.contains(team_b, na=False)
    )
    subset = games.loc[mask].copy()

    if subset.empty:
        return {
            "games": 0,
            "a_wins": 0,
            "b_wins": 0
        }

    # Count each game once using team_a row if possible
    team_a_rows = subset[subset["TEAM_ABBREVIATION"] == team_a]
    team_a_rows = team_a_rows.drop_duplicates(subset=["GAME_ID"])

    a_wins = (team_a_rows["WL"] == "W").sum()
    total = len(team_a_rows)

    return {
        "games": total,
        "a_wins": int(a_wins),
        "b_wins": int(total - a_wins)
    }

def win_prob_gauge(prob, team_a):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": f"{team_a} Win Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.3},
            "steps": [
                {"range": [0, 35], "color": "#f8d7da"},
                {"range": [35, 65], "color": "#fff3cd"},
                {"range": [65, 100], "color": "#d1e7dd"},
            ],
        }
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# =========================================================
# LOAD MODEL
# =========================================================
with st.spinner("Loading season data and building model..."):
    games = get_games()
    teams = get_teams_from_games(games)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Controls")

k_factor = st.sidebar.slider("Elo K-Factor", 8, 40, 18)
sim_count = st.sidebar.slider("Playoff Simulations", 100, 5000, 1000, step=100)
last_n_form = st.sidebar.slider("Form Window", 5, 20, 10)
home_edge = st.sidebar.slider("Home Court Edge", 0, 120, 50, step=5)

elo, history_df = build_model(teams, games, k_factor=k_factor)

elo_df = pd.DataFrame({
    "Team": list(elo.keys()),
    "Elo": list(elo.values())
}).sort_values("Elo", ascending=False).reset_index(drop=True)

# =========================================================
# TOP SUMMARY
# =========================================================
top_team = elo_df.iloc[0]["Team"] if not elo_df.empty else "N/A"
avg_elo = elo_df["Elo"].mean() if not elo_df.empty else 1500
games_count = games["GAME_ID"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Teams", len(teams))
c2.metric("Games Modeled", games_count)
c3.metric("Top Elo Team", top_team)
c4.metric("Average Elo", f"{avg_elo:.1f}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Power Ratings",
    "Championship Odds",
    "Live Matchup",
    "Team Form"
])

# =========================================================
# TAB 1 — POWER RATINGS
# =========================================================
with tab1:
    st.subheader("📊 Power Ratings")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.dataframe(elo_df, use_container_width=True, height=650)

    with col2:
        fig = px.bar(
            elo_df.head(15),
            x="Team",
            y="Elo",
            title="Top 15 Teams by Elo"
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 2 — CHAMPIONSHIP ODDS
# =========================================================
with tab2:
    st.subheader("🏆 Championship Simulation")

    if st.button("Run Title Simulation", use_container_width=True):
        results = simulate_playoffs(teams, elo, sims=sim_count)

        champ_df = pd.DataFrame({
            "Team": list(results.keys()),
            "Titles": list(results.values())
        })
        champ_df["Title %"] = champ_df["Titles"] / sim_count
        champ_df = champ_df.sort_values("Title %", ascending=False).reset_index(drop=True)

        left, right = st.columns([1, 1])

        with left:
            st.dataframe(
                champ_df[["Team", "Titles", "Title %"]],
                use_container_width=True,
                height=650
            )

        with right:
            fig = px.bar(
                champ_df.head(15),
                x="Team",
                y="Title %",
                title="Top 15 Championship Odds"
            )
            fig.update_layout(height=650)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3 — LIVE MATCHUP
# =========================================================
with tab3:
    st.subheader("📡 Live Matchup Simulator")

    if "team_a" not in st.session_state:
        st.session_state.team_a = teams[0]
    if "team_b" not in st.session_state:
        st.session_state.team_b = teams[1] if len(teams) > 1 else teams[0]
    if "momentum" not in st.session_state:
        st.session_state.momentum = 0
    if "is_home_a" not in st.session_state:
        st.session_state.is_home_a = True

    with st.form("matchup_controls"):
        mc1, mc2, mc3, mc4 = st.columns(4)

        with mc1:
            team_a = st.selectbox("Team A", teams, index=teams.index(st.session_state.team_a))
        with mc2:
            team_b = st.selectbox("Team B", teams, index=teams.index(st.session_state.team_b))
        with mc3:
            momentum = st.slider("Momentum Swing", -15, 15, st.session_state.momentum)
        with mc4:
            is_home_a = st.checkbox("Team A is Home", value=st.session_state.is_home_a)

        submitted = st.form_submit_button("Update Matchup", use_container_width=True)

    st.session_state.team_a = team_a
    st.session_state.team_b = team_b
    st.session_state.momentum = momentum
    st.session_state.is_home_a = is_home_a

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        a_elo = elo[team_a]
        b_elo = elo[team_b]
        applied_home_edge = home_edge if is_home_a else 0

        prob = live_win_prob(a_elo, b_elo, momentum=momentum, home_edge=applied_home_edge)
        market = stable_market_prob(team_a, team_b)
        edge = prob - market

        h2h = head_to_head_summary(games, team_a, team_b)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{team_a} Elo", f"{a_elo:.0f}")
        k2.metric(f"{team_b} Elo", f"{b_elo:.0f}")
        k3.metric(f"{team_a} Win Prob", f"{prob:.2%}")
        k4.metric("Model Edge", f"{edge:.2%}")

        left, right = st.columns([1, 1])

        with left:
            st.plotly_chart(win_prob_gauge(prob, team_a), use_container_width=True)

        with right:
            compare_df = pd.DataFrame({
                "Source": ["Model", "Market"],
                "Probability": [prob, market]
            })

            fig = px.bar(
                compare_df,
                x="Source",
                y="Probability",
                title=f"{team_a} Probability: Model vs Market",
                text_auto=".1%"
            )
            fig.update_layout(height=320, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        info1, info2, info3 = st.columns(3)
        info1.info(f"Head-to-head games found: **{h2h['games']}**")
        info2.info(f"{team_a} wins: **{h2h['a_wins']}**")
        info3.info(f"{team_b} wins: **{h2h['b_wins']}**")

        verdict = "No edge"
        if edge > 0.05:
            verdict = f"Strong model edge on {team_a}"
        elif edge > 0.02:
            verdict = f"Moderate model edge on {team_a}"
        elif edge < -0.05:
            verdict = f"Market may be overpriced on {team_a}"
        elif edge < -0.02:
            verdict = f"Slight fade signal on {team_a}"

        st.success(verdict)

# =========================================================
# TAB 4 — TEAM FORM
# =========================================================
with tab4:
    st.subheader("📈 Team Form Tracker")

    tf1, tf2 = st.columns([1, 1])

    with tf1:
        selected_team = st.selectbox("Select team", teams, key="form_team")

    form_df = team_form_from_history(history_df, selected_team, last_n=last_n_form)

    if form_df.empty:
        st.warning("No form data available for that team.")
    else:
        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            display_df = form_df.copy()
            display_df["GAME_DATE"] = display_df["GAME_DATE"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                display_df[["GAME_DATE", "OPP", "RESULT", "PRE_ELO", "POST_ELO"]],
                use_container_width=True,
                height=500
            )

        with col_right:
            fig = px.line(
                form_df,
                x="GAME_DATE",
                y="POST_ELO",
                markers=True,
                title=f"{selected_team} Elo Trend (Last {last_n_form})"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Notes: Elo is built from season game results, title odds are Monte Carlo estimates, "
    "and the market line is a stable placeholder until you connect real odds."
)
