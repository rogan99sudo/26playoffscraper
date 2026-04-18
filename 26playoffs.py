import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
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
# STYLE
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
    padding-top: 1rem;
    padding-bottom: 1rem;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 12px 14px;
    border-radius: 16px;
}
.ticker {
    padding: 12px 16px;
    border-radius: 14px;
    background: linear-gradient(90deg, rgba(59,130,246,0.18), rgba(168,85,247,0.16));
    border: 1px solid rgba(255,255,255,0.08);
    font-size: 0.95rem;
    margin-bottom: 14px;
}
.bracket-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 10px;
}
.small-note {
    opacity: 0.75;
    font-size: 0.85rem;
}
.logo-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 6px 0 2px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🏀 NBA Live Intelligence Engine")
st.caption("Power ratings, playoff simulations, live matchup intelligence, team form, and watchlists")

# =========================================================
# TEAM MAPS
# =========================================================
EAST = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
    "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"
}
WEST = {
    "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
    "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"
}

TEAM_ID_MAP = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "BKN": 1610612751,
    "CHA": 1610612766,
    "CHI": 1610612741,
    "CLE": 1610612739,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "DET": 1610612765,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "IND": 1610612754,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MEM": 1610612763,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "NOP": 1610612740,
    "NYK": 1610612752,
    "OKC": 1610612760,
    "ORL": 1610612753,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "WAS": 1610612764,
}

def get_conference(team):
    if team in EAST:
        return "East"
    if team in WEST:
        return "West"
    return "Unknown"

def get_logo_url(team):
    team_id = TEAM_ID_MAP.get(team)
    if not team_id:
        return None
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

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

def market_prob(team_a, team_b, use_api=False, api_key=None):
    # Placeholder hook for real odds integration
    # Replace this block with requests to your odds provider if desired
    if use_api and api_key:
        return stable_market_prob(team_a, team_b)
    return stable_market_prob(team_a, team_b)

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

def team_last_n_record(games, team, n=10):
    team_games = games[games["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE").drop_duplicates("GAME_ID")
    last_n = team_games.tail(n)
    wins = int((last_n["WL"] == "W").sum())
    losses = int((last_n["WL"] == "L").sum())
    return wins, losses, last_n

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

def build_trend_table(form_map, elo_df):
    rows = []
    for _, r in elo_df.iterrows():
        team = r["Team"]
        form_df = form_map.get(team, pd.DataFrame())
        recent_delta = float(form_df["ELO_DELTA"].sum()) if not form_df.empty else 0.0
        recent_wins = int((form_df["RESULT"] == "W").sum()) if not form_df.empty else 0
        recent_losses = int((form_df["RESULT"] == "L").sum()) if not form_df.empty else 0
        rows.append({
            "Team": team,
            "Conference": get_conference(team),
            "Elo": r["Elo"],
            "Recent Elo Change": recent_delta,
            "Recent Record": f"{recent_wins}-{recent_losses}"
        })
    return pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)

def render_logo(team, width=90):
    logo_url = get_logo_url(team)
    if logo_url:
        st.markdown('<div class="logo-wrap">', unsafe_allow_html=True)
        st.image(logo_url, width=width)
        st.markdown('</div>', unsafe_allow_html=True)

def render_bracket_side(df, title):
    st.markdown(f"**{title}**")
    for idx, row in df.reset_index(drop=True).iterrows():
        st.markdown(
            f"""
            <div class="bracket-card">
                <b>{idx + 1}. {row["Team"]}</b><br>
                <span class="small-note">Title %: {row["Title %"]:.1%}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# OPTIONAL AUTO-REFRESH
# =========================================================
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = time.time()

# =========================================================
# LOAD DATA
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

conference_filter = st.sidebar.selectbox("Conference Filter", ["All", "East", "West"])
show_only_top = st.sidebar.slider("Show Top Teams", 5, 30, 15)

st.sidebar.markdown("---")
use_market_api = st.sidebar.checkbox("Use Real Odds Hook", value=False)
market_api_key = st.sidebar.text_input("Odds API Key", type="password")
auto_refresh = st.sidebar.checkbox("Auto Refresh Every 30s", value=False)

if st.sidebar.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# =========================================================
# MODEL
# =========================================================
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
trend_df = build_trend_table(form_map, elo_df)

# =========================================================
# TICKER
# =========================================================
top_team = elo_df.iloc[0]["Team"] if not elo_df.empty else "N/A"
top_elo = elo_df.iloc[0]["Elo"] if not elo_df.empty else 0.0
hottest = trend_df.sort_values("Recent Elo Change", ascending=False).iloc[0]
coldest = trend_df.sort_values("Recent Elo Change", ascending=True).iloc[0]

st.markdown(
    f"""
    <div class="ticker">
        <b>Top Power Team:</b> {top_team} ({top_elo:.1f} Elo)
        &nbsp;&nbsp;•&nbsp;&nbsp;
        <b>Hottest:</b> {hottest["Team"]} ({hottest["Recent Elo Change"]:+.1f})
        &nbsp;&nbsp;•&nbsp;&nbsp;
        <b>Coldest:</b> {coldest["Team"]} ({coldest["Recent Elo Change"]:+.1f})
        &nbsp;&nbsp;•&nbsp;&nbsp;
        <b>Refresh:</b> {pd.Timestamp.now().strftime("%H:%M:%S")}
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TOP METRICS
# =========================================================
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Teams", len(filtered_elo_df))
m2.metric("Games Modeled", games["GAME_ID"].nunique())
m3.metric("Top Team", top_team)
m4.metric("Top Elo", f"{top_elo:.1f}")
m5.metric("Avg Elo", f"{filtered_elo_df['Elo'].mean():.1f}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Power Board",
    "Playoff Board",
    "Live Matchup",
    "Team Profile",
    "Form Tracker",
    "Watchlist",
    "Model Guide"
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
        chart_df = filtered_elo_df.head(show_only_top).sort_values("Elo", ascending=True)
        fig = px.bar(
            chart_df,
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

    c1, c2 = st.columns([1, 1])

    with c1:
        st.dataframe(
            champ_df[["Team", "Conference", "Titles", "Title %"]],
            use_container_width=True,
            height=500
        )

    with c2:
        fig = px.bar(
            champ_df.head(show_only_top).sort_values("Title %", ascending=True),
            x="Title %",
            y="Team",
            color="Conference",
            orientation="h",
            title="Best Championship Chances"
        )
        fig.update_layout(
            height=500,
            xaxis_tickformat=".0%",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Bracket Snapshot")

    east_board = champ_df[champ_df["Conference"] == "East"].head(8)[["Team", "Title %"]]
    west_board = champ_df[champ_df["Conference"] == "West"].head(8)[["Team", "Title %"]]

    b1, b2 = st.columns(2)
    with b1:
        render_bracket_side(east_board, "East Top 8")
    with b2:
        render_bracket_side(west_board, "West Top 8")

# =========================================================
# TAB 3 — LIVE MATCHUP
# =========================================================
with tab3:
    st.subheader("📡 Live Matchup")

    available_teams = filtered_elo_df["Team"].tolist() if not filtered_elo_df.empty else teams

    if "team_a" not in st.session_state or st.session_state.team_a not in available_teams:
        st.session_state.team_a = available_teams[0]
    if "team_b" not in st.session_state or st.session_state.team_b not in available_teams:
        st.session_state.team_b = available_teams[1] if len(available_teams) > 1 else available_teams[0]
    if "momentum" not in st.session_state:
        st.session_state.momentum = 0
    if "venue" not in st.session_state:
        st.session_state.venue = "Team A Home"

    with st.form("matchup_form"):
        a1, a2, a3, a4 = st.columns(4)

        with a1:
            team_a = st.selectbox(
                "Team A",
                available_teams,
                index=available_teams.index(st.session_state.team_a)
            )

        with a2:
            default_b_index = (
                available_teams.index(st.session_state.team_b)
                if st.session_state.team_b in available_teams
                else (1 if len(available_teams) > 1 else 0)
            )
            team_b = st.selectbox(
                "Team B",
                available_teams,
                index=default_b_index
            )

        with a3:
            momentum = st.slider("Momentum Swing", -15, 15, st.session_state.momentum)

        with a4:
            venue = st.radio("Venue", ["Neutral", "Team A Home", "Team B Home"])

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
        market = market_prob(team_a, team_b, use_api=use_market_api, api_key=market_api_key)
        edge = prob - market
        h2h = head_to_head_summary(games, team_a, team_b)

        logos, metrics = st.columns([0.8, 2.4])

        with logos:
            l1, l2 = st.columns(2)
            with l1:
                render_logo(team_a, 95)
                st.markdown(f"<div style='text-align:center;'><b>{team_a}</b></div>", unsafe_allow_html=True)
            with l2:
                render_logo(team_b, 95)
                st.markdown(f"<div style='text-align:center;'><b>{team_b}</b></div>", unsafe_allow_html=True)

        with metrics:
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
                "Source": ["Model", "Market"],
                "Probability": [prob, market]
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
# TAB 4 — TEAM PROFILE
# =========================================================
with tab4:
    st.subheader("🧾 Team Profile")

    profile_team = st.selectbox("Select team", available_teams, key="profile_team")
    form_df = team_form_from_history(history_df, profile_team, last_n=form_window)
    last10_w, last10_l, last10_df = team_last_n_record(games, profile_team, n=10)

    p1, p2 = st.columns([0.8, 2.2])
    with p1:
        render_logo(profile_team, 120)
    with p2:
        st.markdown(f"### {profile_team}")
        st.write(f"**Conference:** {get_conference(profile_team)}")
        st.write(f"**Current Elo:** {elo[profile_team]:.1f}")
        st.write(f"**Last 10 Record:** {last10_w}-{last10_l}")

    if not form_df.empty:
        c1, c2 = st.columns([1, 1])

        with c1:
            profile_view = form_df.copy()
            profile_view["GAME_DATE"] = profile_view["GAME_DATE"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                profile_view[["GAME_DATE", "OPP", "RESULT", "PRE_ELO", "POST_ELO", "ELO_DELTA"]],
                use_container_width=True,
                height=420
            )

        with c2:
            fig = px.line(
                form_df,
                x="GAME_DATE",
                y="POST_ELO",
                markers=True,
                title=f"{profile_team} Elo Trend"
            )
            fig.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# TAB 5 — FORM TRACKER
# =========================================================
with tab5:
    st.subheader("📈 Form Tracker")

    team_for_form = st.selectbox("Choose team", available_teams, key="form_select")
    form_df = team_form_from_history(history_df, team_for_form, last_n=form_window)
    last10_w, last10_l, _ = team_last_n_record(games, team_for_form, n=10)

    x1, x2, x3 = st.columns(3)
    x1.metric("Current Elo", f"{elo[team_for_form]:.1f}")
    x2.metric("Last 10 Record", f"{last10_w}-{last10_l}")
    x3.metric("Recent Elo Change", f"{form_df['ELO_DELTA'].sum():+.1f}" if not form_df.empty else "+0.0")

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
                height=500
            )

        with right:
            fig = px.bar(
                form_df,
                x="GAME_DATE",
                y="ELO_DELTA",
                title=f"{team_for_form} Game-by-Game Elo Change"
            )
            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# TAB 6 — WATCHLIST
# =========================================================
with tab6:
    st.subheader("🎯 Watchlist")

    watch_df = trend_df.copy()
    if conference_filter != "All":
        watch_df = watch_df[watch_df["Conference"] == conference_filter].reset_index(drop=True)

    risers = watch_df.sort_values("Recent Elo Change", ascending=False).head(8).reset_index(drop=True)
    fallers = watch_df.sort_values("Recent Elo Change", ascending=True).head(8).reset_index(drop=True)
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
    "This dashboard uses season-based Elo, Monte Carlo title simulation, optional auto-refresh, "
    "logo display, and a placeholder market hook. Swap the market function with a real odds provider when ready."
)

# =========================================================
# AUTO REFRESH
# =========================================================
if auto_refresh:
    time.sleep(30)
    st.rerun()

# =========================================================
# TAB 7 — MODEL GUIDE
# =========================================================
with tab7:
    st.subheader("🧠 How the Elo Model Works")

    st.markdown("""
    ### 📊 What is Elo?

    Elo is a rating system that measures team strength based on game results.

    - Every team starts at **1500**
    - Teams gain points when they win
    - Teams lose points when they lose
    - Beating a strong team = bigger boost
    - Losing to a weak team = bigger penalty
    """)

    st.markdown("---")

    st.markdown("### ⚙️ Win Probability Formula")

    st.write("We calculate win probability using this formula:")

    st.latex(r"P(A) = \frac{1}{1 + 10^{(Elo_B - Elo_A)/400}}")

    st.markdown("""
    - If both teams are equal → **50%**
    - If Team A is stronger → probability increases
    - If Team A is weaker → probability decreases
    """)

    st.markdown("---")

    st.markdown("### 🔄 Rating Update Formula")

    st.latex(r"Elo_{new} = Elo_{old} + K \cdot (Actual - Expected)")

    st.markdown("""
    Where:
    - **Expected** = predicted win probability
    - **Actual** = 1 (win) or 0 (loss)
    - **K-factor** controls how fast ratings change
    """)

    st.markdown("---")

    st.markdown("### 🔥 Example")

    st.write("Team A (1600 Elo) vs Team B (1500 Elo):")

    st.latex(r"P(A) \approx 64\%")

    st.markdown("""
    If Team A wins:
    - small Elo gain (expected result)

    If Team A loses:
    - big Elo drop (upset penalty)
    """)

    st.markdown("---")

    st.markdown("### 🚀 How This App Enhances Elo")

    st.markdown("""
    We extend basic Elo with:

    **1. Momentum Adjustment**
    - Slider adds/subtracts Elo in real-time
    - Simulates hot streaks or in-game swings

    **2. Home Court Advantage**
    - Adds a fixed Elo boost for home team
    - Typically worth ~40–80 Elo

    **3. Monte Carlo Simulation**
    - Runs thousands of simulated playoffs
    - Converts ratings into **title odds**

    **4. Market Comparison**
    - Compare model probability vs sportsbook
    - Identify **edges and inefficiencies**
    """)

    st.markdown("---")

    st.markdown("### 📈 Interpreting Elo Differences")

    diff_df = pd.DataFrame({
        "Elo Difference": [0, 50, 100, 200, 300],
        "Win Probability": ["50%", "57%", "64%", "76%", "85%"]
    })

    st.dataframe(diff_df, use_container_width=True)

    st.markdown("---")

    st.markdown("### 🎯 What Makes This Useful")

    st.markdown("""
    - Objective team strength measurement  
    - Reacts to every game played  
    - Powers matchup predictions  
    - Drives simulation-based betting insights  
    """)

    st.success("💡 Think of Elo as a live power rating that constantly updates based on performance.")
