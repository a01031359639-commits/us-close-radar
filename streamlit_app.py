
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="미장 종가 레이더",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

NY = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")

DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","NFLX",
    "PLTR","COIN","MSTR","HOOD","SOFI","RIVN","LCID","INTC","MU","QCOM",
    "ARM","SMCI","CRWD","PANW","SNOW","DDOG","NET","SHOP","UBER","ABNB",
    "JPM","BAC","C","WFC","GS","MS","V","MA","PYPL","AXP",
    "XOM","CVX","COP","OXY","SLB","LLY","UNH","JNJ","PFE","MRK",
    "WMT","COST","HD","LOW","NKE","DIS","BA","CAT","GE","GM",
    "F","DAL","UAL","AAL","CCL","NCLH","T","VZ","TMUS","KO",
    "PEP","MCD","SBUX","ORCL","CRM","ADBE","IBM","NOW","APP","RDDT"
]

st.markdown("""
<style>
.block-container{
    max-width:760px;
    padding-top:0.8rem;
    padding-left:0.85rem;
    padding-right:0.85rem;
    padding-bottom:2rem;
}
h1{font-size:1.65rem !important; margin-bottom:.25rem !important;}
h2{font-size:1.25rem !important;}
div[data-testid="stMetricValue"]{font-size:1.25rem;}
div[data-testid="stMetricLabel"]{font-size:.8rem;}
.stButton > button{
    width:100%;
    min-height:3rem;
    font-size:1rem;
    font-weight:700;
    border-radius:.8rem;
}
div[data-testid="stExpander"]{
    border-radius:.8rem;
}
.radar-card{
    border:1px solid rgba(128,128,128,.25);
    border-radius:1rem;
    padding:.85rem .9rem;
    margin:.55rem 0;
}
.radar-card .ticker{font-size:1.3rem;font-weight:800;}
.radar-card .price{font-size:1.05rem;font-weight:700;}
.radar-card .reason{font-size:.88rem;opacity:.8;margin-top:.25rem;}
.badge{
    display:inline-block;
    padding:.18rem .45rem;
    border-radius:.55rem;
    border:1px solid rgba(128,128,128,.28);
    margin:.15rem .12rem .1rem 0;
    font-size:.78rem;
}
.small-note{font-size:.8rem;opacity:.72;}
</style>
""", unsafe_allow_html=True)


def market_status():
    now_ny = datetime.now(NY)
    t = now_ny.time().replace(tzinfo=None)
    if now_ny.weekday() >= 5:
        return "주말", now_ny
    if dtime(9, 30) <= t < dtime(16, 0):
        close_dt = datetime.combine(now_ny.date(), dtime(16, 0), NY)
        mins = max(0, int((close_dt - now_ny).total_seconds() // 60))
        return f"장중 · 종가까지 {mins//60}시간 {mins%60}분", now_ny
    if t < dtime(9, 30):
        return "장 시작 전", now_ny
    return "정규장 종료", now_ny


@st.cache_data(ttl=180, show_spinner=False)
def fetch_intraday(tickers):
    return yf.download(
        tickers=tickers,
        period="5d",
        interval="5m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        prepost=False,
    )


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily(tickers):
    return yf.download(
        tickers=tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )


def symbol_frame(data, symbol):
    if data is None or len(data) == 0:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if symbol in data.columns.get_level_values(0):
            out = data[symbol].copy()
        else:
            try:
                out = data.xs(symbol, axis=1, level=1).copy()
            except Exception:
                return pd.DataFrame()
    else:
        out = data.copy()
    return out.dropna(how="all")


def fnum(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def analyze(symbol, intra, daily):
    i = symbol_frame(intra, symbol)
    d = symbol_frame(daily, symbol)
    if i.empty or d.empty:
        return None

    if not {"Open","High","Low","Close","Volume"}.issubset(i.columns):
        return None
    if not {"High","Close","Volume"}.issubset(d.columns):
        return None

    idx = i.index
    try:
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC").tz_convert(NY)
        else:
            idx = idx.tz_convert(NY)
    except Exception:
        pass
    i = i.copy()
    i.index = idx

    current_date = datetime.now(NY).date()
    tday = i[i.index.date == current_date]
    if tday.empty:
        last_date = i.index[-1].date()
        tday = i[i.index.date == last_date]
    if tday.empty:
        return None

    opens = tday["Open"].dropna()
    closes = tday["Close"].dropna()
    if opens.empty or closes.empty:
        return None

    op = fnum(opens.iloc[0])
    close = fnum(closes.iloc[-1])
    high = fnum(tday["High"].max())
    volume = fnum(tday["Volume"].sum())
    if not np.isfinite(close) or close <= 0:
        return None

    change = (close / op - 1) * 100 if np.isfinite(op) and op > 0 else np.nan
    high_dist = (high - close) / high * 100 if np.isfinite(high) and high > 0 else np.nan

    typ = (tday["High"] + tday["Low"] + tday["Close"]) / 3
    vols = tday["Volume"].fillna(0)
    vwap = fnum((typ * vols).sum() / vols.sum()) if vols.sum() > 0 else np.nan
    above_vwap = bool(np.isfinite(vwap) and close >= vwap)

    d2 = d.dropna(subset=["Close"]).copy()
    if len(d2) < 22:
        return None
    hist = d2.iloc[-21:-1]
    avg_vol = fnum(hist["Volume"].mean())
    avg_price = fnum(hist["Close"].mean())
    avg_dollar = avg_vol * avg_price if np.isfinite(avg_vol) and np.isfinite(avg_price) else np.nan
    rvol = volume / avg_vol if np.isfinite(avg_vol) and avg_vol > 0 else np.nan
    high20 = fnum(hist["High"].max())
    breakout = bool(np.isfinite(high20) and close >= high20)

    last6 = tday.tail(6)
    mom30 = np.nan
    if len(last6) >= 2:
        first = fnum(last6["Close"].iloc[0])
        last = fnum(last6["Close"].iloc[-1])
        if np.isfinite(first) and first > 0 and np.isfinite(last):
            mom30 = (last / first - 1) * 100

    score = 0.0
    reasons = []

    if np.isfinite(change):
        score += np.clip(change, -5, 12) * 2
        if change >= 2:
            reasons.append(f"당일 +{change:.1f}%")
    if np.isfinite(rvol):
        score += np.clip((rvol - 1) * 12, -6, 30)
        if rvol >= 1.5:
            reasons.append(f"RVOL {rvol:.1f}배")
    if np.isfinite(high_dist):
        score += max(0, 20 - high_dist * 7)
        if high_dist <= 1.5:
            reasons.append(f"고가 {high_dist:.1f}% 이내")
    if above_vwap:
        score += 12
        reasons.append("VWAP 위")
    if breakout:
        score += 18
        reasons.append("20일 고가 돌파")
    if np.isfinite(mom30):
        score += np.clip(mom30 * 6, -8, 15)
        if mom30 >= .7:
            reasons.append(f"30분 +{mom30:.1f}%")

    return {
        "티커": symbol,
        "현재가": close,
        "등락률": change,
        "RVOL": rvol,
        "고가거리": high_dist,
        "VWAP": vwap,
        "VWAP위": above_vwap,
        "20일돌파": breakout,
        "30분": mom30,
        "평균거래대금": avg_dollar,
        "점수": float(score),
        "이유": " · ".join(reasons[:4]) if reasons else "조건 약함",
    }


status, now_ny = market_status()
now_kst = datetime.now(KST)

st.title("📡 미장 종가 레이더")
st.caption("휴대폰 전용 · 자동주문 없이 종가 후보만 찾아주는 버전")

m1, m2 = st.columns(2)
m1.metric("뉴욕", now_ny.strftime("%H:%M"))
m2.metric("한국", now_kst.strftime("%H:%M"))
st.markdown(f"**시장 상태:** {status}")

with st.expander("⚙️ 조건 설정", expanded=False):
    min_price = st.number_input("최소 주가 ($)", min_value=1.0, value=5.0, step=1.0)
    min_adv = st.number_input("최소 평균 거래대금 ($M)", min_value=1, value=20, step=5)
    min_change = st.number_input("최소 당일 상승률 (%)", value=1.0, step=.5)
    min_rvol = st.number_input("최소 RVOL", min_value=.1, value=1.2, step=.1)
    max_high_dist = st.number_input("당일 고가와 최대 거리 (%)", min_value=.1, value=3.0, step=.5)
    must_vwap = st.checkbox("VWAP 위 종목만", value=True)
    top_n = st.slider("후보 개수", 3, 12, 5)

if st.button("🔎 지금 종가 후보 찾기"):
    st.session_state["scan"] = True

if st.session_state.get("scan", False):
    with st.spinner("미국 종목을 분석하고 있어요..."):
        try:
            intra = fetch_intraday(DEFAULT_TICKERS)
            daily = fetch_daily(DEFAULT_TICKERS)
        except Exception as e:
            st.error(f"시세를 불러오지 못했어요: {e}")
            st.stop()

        rows = []
        for sym in DEFAULT_TICKERS:
            try:
                item = analyze(sym, intra, daily)
                if item:
                    rows.append(item)
            except Exception:
                pass

    if not rows:
        st.warning("분석 가능한 시세가 없어요. 잠시 뒤 다시 눌러주세요.")
        st.stop()

    df = pd.DataFrame(rows)
    mask = (
        (df["현재가"] >= min_price) &
        (df["평균거래대금"] >= min_adv * 1_000_000) &
        (df["등락률"] >= min_change) &
        (df["RVOL"] >= min_rvol) &
        (df["고가거리"] <= max_high_dist)
    )
    if must_vwap:
        mask &= df["VWAP위"]

    out = df[mask].sort_values(["점수","RVOL","등락률"], ascending=False).head(top_n)

    st.subheader("오늘의 종가 후보")

    if out.empty:
        st.info("현재 조건을 모두 통과한 종목이 없어요. 조건 설정에서 RVOL이나 상승률 기준을 조금 낮춰보세요.")
        out = df.sort_values("점수", ascending=False).head(3)
        st.caption("참고: 조건 미충족이지만 점수가 높은 종목")

    for rank, (_, r) in enumerate(out.iterrows(), start=1):
        arrow = "▲" if r["등락률"] >= 0 else "▼"
        st.markdown(
            f"""
            <div class="radar-card">
              <div class="ticker">{rank}. {r['티커']}</div>
              <div class="price">${r['현재가']:.2f} · {arrow} {r['등락률']:+.2f}%</div>
              <div>
                <span class="badge">RVOL {r['RVOL']:.2f}x</span>
                <span class="badge">고가거리 {r['고가거리']:.2f}%</span>
                <span class="badge">점수 {r['점수']:.1f}</span>
              </div>
              <div class="reason">{r['이유']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    ticker = st.selectbox("차트 확인", out["티커"].tolist())
    chart = symbol_frame(intra, ticker)
    if not chart.empty and "Close" in chart.columns:
        st.line_chart(chart[["Close"]].dropna().tail(80), height=260)

st.caption("※ Yahoo Finance 비공식 시세를 사용하므로 지연·누락 가능성이 있습니다. 실제 주문 전에는 증권사 시세를 확인하세요.")
st.caption("※ 레이더 점수는 후보를 거르는 규칙일 뿐 수익 가능성을 보장하지 않습니다.")
