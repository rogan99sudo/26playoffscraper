import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nba_api.stats.endpoints import leaguegamefinder

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="NBA Live Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}
.main {
    background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    letter-spacing: -0.02em;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px 16px;
    border-radius: 16px;
}
.dashboard-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.ticker {
    padding: 12px 16px;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(59,130,246,0.18), rgba(168,85,247,0.16));
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 0.95rem;
    margin-bottom: 14px;
}
.small-note {
    opacity: 0.75;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Live Intelligence Engine")
st.caption("Broadcast-style team ratings, title odds, live matchup intelligence, and trend tracking")

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
# LIGHT CONFERENCE MAP
# =========================================================
EAST = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
    "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"
}
WEST = {
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
    "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"
}

def get_conference(team):
    if team in EAST:
        return "East"
    if team in WEST:
        return "West"
    return "Unknown"

# =========================================================
# ELO MODEL
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

    return elo, pd.DataFrame(history)

# =========================================================
# HELPERS
# =========================================================
def live_win_prob(elo_a, elo_b, momentum=0, venue_adjustment=0):
    adjusted_a = elo_a + momentum * 10 + venue_adjustment
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

@st.cache_data(show_spinner=False)
def cached_playoff_sim(teams_tuple, elo_items_tuple, sims):
    teams = list(teams_tuple)
    elo = dict(elo_items_tuple)
    return simulate_playoffs(teams, elo, sims=sims)

def team_form_from_history(history_df, team, last_n=10):
    if history_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in history_df.iterrows():
        if r["TEAM_A"] == team:
            rows.append({
                "GAME_DATE": r["GAME_DATE"],
                "TEAM": team,
                "OPP": r["TEAM_B"],
                "RESULT": "W" if r["TEAM_A_WON"] else "L",
                "PRE_ELO": r["PRE_ELO_A"],
                "POST_ELO": r["POST_ELO_A"],
                "ELO_DELTA": r["POST_ELO_A"] - r["PRE_ELO_A"],
            })
        elif r["TEAM_B"] == team:
            rows.append({
                "GAME_DATE": r["GAME_DATE"],
                "TEAM": team,
                "OPP": r["TEAM_A"],
                "RESULT": "L" if r["TEAM_A_WON"] else "W",
                "PRE_ELO": r["PRE_ELO_B"],
                "POST_ELO": r["POST_ELO_B"],
                "ELO_DELTA": r["POST_ELO_B"] - r["PRE_ELO_B"],
            })

    df = pd.DataFrame(rows).sort_values("GAME_DATE")
    return df.tail(last_n)

def head_to_head_summary(games, team_a, team_b):
    mask = (
        games["MATCHUP"].str.contains(team_a, na=False) &
        games["MATCHUP"].str.contains(team_b, na=False)
    )
    subset = games.loc[mask].copy()

    if subset.empty:
        return {"games": 0, "a_wins": 0, "b_wins": 0}

    team_a_rows = subset[subset["TEAM_ABBREVIATION"] == team_a].drop_duplicates(subset=["GAME_ID"])
    total = len(team_a_rows)
    a_wins = int((team_a_rows["WL"] == "W").sum())

    return {"games": total, "a_wins": a_wins, "b_wins": total - a_wins}

def win_prob_gauge(prob, team_a):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": f"{team_a} Win Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.28},
            "steps": [
                {"range": [0, 35], "color": "rgba(239,68,68,0.35)"},
                {"range": [35, 65], "color": "rgba(245,158,11,0.35)"},
                {"range": [65, 100], "color": "rgba(34,197,94,0.35)"},
            ],
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

def make_rank_delta_table(form_map, elo_df):
    rows = []
    for _, r in elo_df.iterrows():
        team = r["Team"]
        form_df = form_map.get(team, pd.DataFrame())
        recent_delta = float(form_df["ELO_DELTA"].sum()) if not form_df.empty else 0.0
        rows.append({
            "Team": team,
            "Conference": get_conference(team),
            "Elo": r["Elo"],
            "Last Window Elo Change": recent_delta
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["Conference", "Elo"], ascending=[True, False]).reset_index(drop=True)

# =========================================================
# LOAD MODEL
# =========================================================
with st.spinner("Loading season data and building model..."):
    games = get_games()
    teams = get_teams_from_games(games)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Dashboard Controls")

k_factor = st.sidebar.slider("Elo K-Factor", 8, 40, 18)
sim_count = st.sidebar.slider("Playoff Simulations", 100, 5000, 1000, step=100)
form_window = st.sidebar.slider("Form Window", 5, 20, 10)
home_edge = st.sidebar.slider("Home Court Edge", 0, 120, 50, step=5)

conference_filter = st.sidebar.selectbox(
    "Conference Filter",
    ["All", "East", "West"]
)

show_only_top = st.sidebar.slider("Show Top Teams", 5, 30, 15)

elo, history_df = build_model(teams, games, k_factor=k_factor)

elo_df = pd.DataFrame({
    "Team": list(elo.keys()),
    "Elo": list(elo.values())
})
elo_df["Conference"] = elo_df["Team"].map(get_conference)
elo_df = elo_df.sort_values("Elo", ascending=False).reset_index(drop=True)

if conference_filter != "All":
    filtered_elo_df = elo_df[elo_df["Conference"] == conference_filter].reset_index(drop=True)
else:
    filtered_elo_df = elo_df.copy()

form_map = {
    team: team_form_from_history(history_df, team, last_n=form_window)
    for team in teams
}

trend_df = make_rank_delta_table(form_map, elo_df)

# =========================================================
# HEADLINE / TICKER
# =========================================================
top_team = elo_df.iloc[0]["Team"] if not elo_df.empty else "N/A"
top_elo = elo_df.iloc[0]["Elo"] if not elo_df.empty else 0
biggest_riser_row = trend_df.sort_values("Last Window Elo Change", ascending=False).iloc[0]
biggest_faller_row = trend_df.sort_values("Last Window Elo Change", ascending=True).iloc[0]

st.markdown(
    f"""
    <div class="ticker">
        <b>Top Power Team:</b> {top_team} ({top_elo:.1f} Elo)
        &nbsp;&nbsp;•&nbsp;&nbsp;
        <b>Hottest Team:</b> {biggest_riser_row["Team"]} ({biggest_riser_row["Last Window Elo Change"]:+.1f} last window)
        &nbsp;&nbsp;•&nbsp;&nbsp;
        <b>Coldest Team:</b> {biggest_faller_row["Team"]} ({biggest_faller_row["Last Window Elo Change"]:+.1f} last window)
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TOP METRICS
# =========================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Teams", len(filtered_elo_df))
col2.metric("Games Modeled", games["GAME_ID"].nunique())
col3.metric("Top Team", top_team)
col4.metric("Top Elo", f"{top_elo:.1f}")
col5.metric("Avg Elo", f"{filtered_elo_df['Elo'].mean():.1f}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Power Board",
    "Playoff Board",
    "Live Matchup",
    "Form Tracker",
    "Watchlist"
])

# =========================================================
# TAB 1 — POWER BOARD
# =========================================================
with tab1:
    st.subheader("📊 Power Board")

    left, right = st.columns([1.05, 1])

    with left:
        st.dataframe(
            filtered_elo_df[["Team", "Conference", "Elo"]],
            use_container_width=True,
            height=620
        )

    with right:
        chart_df = filtered_elo_df.head(show_only_top).copy()
        fig = px.bar(
            chart_df.sort_values("Elo", ascending=True),
            x="Elo",
            y="Team",
            color="Conference",
            orientation="h",
            title=f"Top {min(show_only_top, len(chart_df))} Power Ratings"
        )
        fig.update_layout(
            height=620,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text=""
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# TAB 2 — PLAYOFF BOARD
# =========================================================
with tab2:
    st.subheader("🏆 Playoff Board")

    results = cached_playoff_sim(tuple(teams), tuple(elo.items()), sim_count)

    champ_df = pd.DataFrame({
        "Team": list(results.keys()),
        "Titles": list(results.values())
    })
    champ_df["Title %"] = champ_df["Titles"] / sim_count
    champ_df["Conference"] = champ_df["Team"].map(get_conference)
    champ_df = champ_df.sort_values("Title %", ascending=False).reset_index(drop=True)

    if conference_filter != "All":
        champ_df = champ_df[champ_df["Conference"] == conference_filter].reset_index(drop=True)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Championship Odds**")
        st.dataframe(
            champ_df[["Team", "Conference", "Titles", "Title %"]],
            use_container_width=True,
            height=540
        )

    with right:
        fig = px.bar(
            champ_df.head(show_only_top).sort_values("Title %", ascending=True),
            x="Title %",
            y="Team",
            color="Conference",
            orientation="h",
            title="Best Title Chances"
        )
        fig.update_layout(
            height=540,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickformat=".0%"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Playoff Field Snapshot")
    east_board = champ_df[champ_df["Conference"] == "East"].head(8)[["Team", "Title %"]].reset_index(drop=True)
    west_board = champ_df[champ_df["Conference"] == "West"].head(8)[["Team", "Title %"]].reset_index(drop=True)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**East Top 8**")
        east_view = east_board.copy()
        east_view.index = np.arange(1, len(east_view) + 1)
        st.dataframe(east_view, use_container_width=True, height=320)

    with b2:
        st.markdown("**West Top 8**")
        west_view = west_board.copy()
        west_view.index = np.arange(1, len(west_view) + 1)
        st.dataframe(west_view, use_container_width=True, height=320)

# =========================================================
# TAB 3 — LIVE MATCHUP
# =========================================================
with tab3:
    st.subheader("📡 Live Matchup")

    available_teams = filtered_elo_df["Team"].tolist() if not filtered_elo_df.empty else teams

    if "team_a" not in st.session_state:
        st.session_state.team_a = available_teams[0]
    if "team_b" not in st.session_state:
        st.session_state.team_b = available_teams[1] if len(available_teams) > 1 else available_teams[0]
    if "momentum" not in st.session_state:
        st.session_state.momentum = 0
    if "venue" not in st.session_state:
        st.session_state.venue = "Team A Home"

    with st.form("matchup_form"):
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            team_a = st.selectbox(
                "Team A",
                available_teams,
                index=available_teams.index(st.session_state.team_a) if st.session_state.team_a in available_teams else 0
            )

        with m2:
            team_b = st.selectbox(
                "Team B",
                available_teams,
                index=1 if len(available_teams) > 1 else 0
            )

        with m3:
            momentum = st.slider("Momentum Swing", -15, 15, st.session_state.momentum)

        with m4:
            venue = st.radio("Venue", ["Neutral", "Team A Home", "Team B Home"], horizontal=False)

        st.form_submit_button("Update Matchup", use_container_width=True)

    st.session_state.team_a = team_a
    st.session_state.team_b = team_b
    st.session_state.momentum = momentum
    st.session_state.venue = venue

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        venue_adjustment = 0
        if venue == "Team A Home":
            venue_adjustment = home_edge
        elif venue == "Team B Home":
            venue_adjustment = -home_edge

        prob = live_win_prob(elo[team_a], elo[team_b], momentum=momentum, venue_adjustment=venue_adjustment)
        market = stable_market_prob(team_a, team_b)
        edge = prob - market
        h2h = head_to_head_summary(games, team_a, team_b)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{team_a} Elo", f"{elo[team_a]:.0f}")
        k2.metric(f"{team_b} Elo", f"{elo[team_b]:.0f}")
        k3.metric(f"{team_a} Win Prob", f"{prob:.2%}")
        k4.metric("Model Edge", f"{edge:.2%}")

        left, right = st.columns([1, 1])

        with left:
            st.plotly_chart(win_prob_gauge(prob, team_a), use_container_width=True, config={"displayModeBar": False})

        with right:
            compare_df = pd.DataFrame({
                "Source": ["Model", "Market", f"{team_b} Implied"],
                "Probability": [prob, market, 1 - prob]
            })

            fig = px.bar(
                compare_df,
                x="Source",
                y="Probability",
                text_auto=".1%",
                title=f"{team_a} Pricing Panel"
            )
            fig.update_layout(
                height=300,
                yaxis_tickformat=".0%",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        i1, i2, i3 = st.columns(3)
        i1.info(f"Head-to-head games: **{h2h['games']}**")
        i2.info(f"{team_a} wins: **{h2h['a_wins']}**")
        i3.info(f"{team_b} wins: **{h2h['b_wins']}**")

        if edge >= 0.07:
            st.error(f"🚨 Major discrepancy: model strongly favors {team_a}")
        elif edge >= 0.03:
            st.warning(f"⚠️ Moderate edge: model favors {team_a}")
        elif edge <= -0.07:
            st.error(f"🚨 Model is far lower than market on {team_a}")
        elif edge <= -0.03:
            st.warning(f"⚠️ Slight fade signal on {team_a}")
        else:
            st.success("No major pricing discrepancy detected")

# =========================================================
# TAB 4 — FORM TRACKER
# =========================================================
with tab4:
    st.subheader("📈 Form Tracker")

    team_for_form = st.selectbox("Choose team", available_teams, key="form_select")
    form_df = team_form_from_history(history_df, team_for_form, last_n=form_window)

    if form_df.empty:
        st.warning("No form data available.")
    else:
        left, right = st.columns([1.05, 1])

        with left:
            form_view = form_df.copy()
            form_view["GAME_DATE"] = form_view["GAME_DATE"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                form_view[["GAME_DATE", "OPP", "RESULT", "PRE_ELO", "POST_ELO", "ELO_DELTA"]],
                use_container_width=True,
                height=520
            )

        with right:
            fig = px.line(
                form_df,
                x="GAME_DATE",
                y="POST_ELO",
                markers=True,
                title=f"{team_for_form} Elo Trend"
            )
            fig.update_layout(
                height=520,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# TAB 5 — WATCHLIST
# =========================================================
with tab5:
    st.subheader("🎯 Watchlist")

    watch_df = trend_df.copy()

    if conference_filter != "All":
        watch_df = watch_df[watch_df["Conference"] == conference_filter].reset_index(drop=True)

    risers = watch_df.sort_values("Last Window Elo Change", ascending=False).head(8).reset_index(drop=True)
    fallers = watch_df.sort_values("Last Window Elo Change", ascending=True).head(8).reset_index(drop=True)
    strongest = watch_df.sort_values("Elo", ascending=False).head(8).reset_index(drop=True)

    w1, w2, w3 = st.columns(3)

    with w1:
        st.markdown("**Hottest Teams**")
        st.dataframe(risers, use_container_width=True, height=320)

    with w2:
        st.markdown("**Coldest Teams**")
        st.dataframe(fallers, use_container_width=True, height=320)

    with w3:
        st.markdown("**Strongest Teams**")
        st.dataframe(strongest, use_container_width=True, height=320)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "This dashboard uses a season-based Elo model, Monte Carlo title simulation, "
    "and a stable placeholder market line. Replace the market model with real sportsbook odds for real betting analysis."
)
