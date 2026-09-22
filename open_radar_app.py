
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="미장 시가 레이더",
    page_icon="🌅",
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
h1{font-size:1.7rem !important; margin-bottom:.25rem !important;}
div[data-testid="stMetricValue"]{font-size:1.25rem;}
.stButton > button{
    width:100%;
    min-height:3rem;
    font-size:1rem;
    font-weight:700;
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
</style>
""", unsafe_allow_html=True)

def market_status():
    now_ny = datetime.now(NY)
    t = now_ny.time().replace(tzinfo=None)
    if now_ny.weekday() >= 5:
        return "주말", now_ny
    if t < dtime(4,0):
        return "프리마켓 전", now_ny
    if dtime(4,0) <= t < dtime(9,30):
        return "프리마켓", now_ny
    if dtime(9,30) <= t < dtime(10,0):
        mins = int((datetime.combine(now_ny.date(), dtime(10,0), NY) - now_ny).total_seconds() // 60)
        return f"장 시작 직후 · 첫 30분 {mins}분 남음", now_ny
    if dtime(10,0) <= t < dtime(16,0):
        return "정규장 진행 중", now_ny
    return "정규장 종료", now_ny

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(tickers):
    return yf.download(
        tickers=tickers,
        period="5d",
        interval="5m",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        prepost=True,
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
    if not {"Open","High","Low","Close","Volume"}.issubset(d.columns):
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

    now_ny = datetime.now(NY)
    current_date = now_ny.date()

    today = i[i.index.date == current_date]
    if today.empty:
        last_date = i.index[-1].date()
        today = i[i.index.date == last_date]

    d2 = d.dropna(subset=["Close"]).copy()
    if len(d2) < 22:
        return None

    # 이전 거래일 종가/고가
    prev_close = fnum(d2["Close"].iloc[-2])
    prev_high = fnum(d2["High"].iloc[-2])
    if not np.isfinite(prev_close) or prev_close <= 0:
        return None

    # 프리마켓 04:00~09:30 ET
    pre = today[(today.index.time >= dtime(4,0)) & (today.index.time < dtime(9,30))]
    reg = today[(today.index.time >= dtime(9,30)) & (today.index.time < dtime(16,0))]

    ref_frame = pre if not pre.empty else today
    if ref_frame.empty:
        return None

    last_price = fnum(ref_frame["Close"].dropna().iloc[-1])
    pre_high = fnum(pre["High"].max()) if not pre.empty else np.nan
    pre_low = fnum(pre["Low"].min()) if not pre.empty else np.nan
    pre_vol = fnum(pre["Volume"].sum()) if not pre.empty else 0.0

    gap = (last_price / prev_close - 1) * 100 if np.isfinite(last_price) else np.nan

    # 평균 거래량/거래대금
    hist20 = d2.iloc[-21:-1]
    avg_vol20 = fnum(hist20["Volume"].mean())
    avg_price20 = fnum(hist20["Close"].mean())
    avg_dollar = avg_vol20 * avg_price20 if np.isfinite(avg_vol20) and np.isfinite(avg_price20) else np.nan

    # 프리마켓 상대 거래량: 일평균의 일부와 비교한 간이 지표
    pre_rvol = pre_vol / (avg_vol20 * 0.12) if np.isfinite(avg_vol20) and avg_vol20 > 0 else np.nan

    # 정규장 시작 후 지표
    open_change = np.nan
    above_open = False
    above_vwap = False
    orb_breakout = False
    first15_high = np.nan
    reg_price = np.nan

    if not reg.empty:
        reg_price = fnum(reg["Close"].dropna().iloc[-1])
        open_price = fnum(reg["Open"].dropna().iloc[0]) if not reg["Open"].dropna().empty else np.nan
        if np.isfinite(open_price) and open_price > 0 and np.isfinite(reg_price):
            open_change = (reg_price / open_price - 1) * 100
            above_open = reg_price >= open_price

        typ = (reg["High"] + reg["Low"] + reg["Close"]) / 3
        vols = reg["Volume"].fillna(0)
        vwap = fnum((typ * vols).sum() / vols.sum()) if vols.sum() > 0 else np.nan
        above_vwap = bool(np.isfinite(vwap) and np.isfinite(reg_price) and reg_price >= vwap)

        first15 = reg.head(3)  # 5분봉 3개
        if not first15.empty:
            first15_high = fnum(first15["High"].max())
            if np.isfinite(first15_high) and np.isfinite(reg_price):
                orb_breakout = reg_price >= first15_high

        last_price = reg_price if np.isfinite(reg_price) else last_price

    score = 0.0
    reasons = []

    if np.isfinite(gap):
        score += np.clip(gap, -5, 12) * 2.0
        if gap >= 2:
            reasons.append(f"갭 +{gap:.1f}%")

    if np.isfinite(pre_rvol):
        score += np.clip((pre_rvol - 1) * 10, -5, 30)
        if pre_rvol >= 1.5:
            reasons.append(f"프리 RVOL {pre_rvol:.1f}x")

    if np.isfinite(pre_high) and np.isfinite(last_price) and pre_high > 0:
        dist_pre_high = (pre_high - last_price) / pre_high * 100
        score += max(0, 18 - dist_pre_high * 6)
        if dist_pre_high <= 1.0:
            reasons.append("프리 고점 근접")
    else:
        dist_pre_high = np.nan

    if np.isfinite(last_price) and np.isfinite(prev_high) and last_price >= prev_high:
        score += 15
        reasons.append("전일 고가 돌파")

    if above_open:
        score += 8
        reasons.append("시가 위")
    if above_vwap:
        score += 10
        reasons.append("VWAP 위")
    if orb_breakout:
        score += 16
        reasons.append("15분 고가 돌파")
    if np.isfinite(open_change):
        score += np.clip(open_change * 6, -8, 15)

    return {
        "티커": symbol,
        "현재가": last_price,
        "갭%": gap,
        "프리RVOL": pre_rvol,
        "프리고가거리%": dist_pre_high,
        "전일고가돌파": bool(np.isfinite(last_price) and np.isfinite(prev_high) and last_price >= prev_high),
        "시가위": above_open,
        "VWAP위": above_vwap,
        "15분돌파": orb_breakout,
        "시가대비%": open_change,
        "평균거래대금": avg_dollar,
        "점수": float(score),
        "이유": " · ".join(reasons[:4]) if reasons else "조건 약함",
    }

status, now_ny = market_status()
now_kst = datetime.now(KST)

st.title("🌅 미장 시가 레이더")
st.caption("프리마켓 + 장 시작 직후 강한 종목을 찾는 버전")

m1, m2 = st.columns(2)
m1.metric("뉴욕", now_ny.strftime("%H:%M"))
m2.metric("한국", now_kst.strftime("%H:%M"))
st.markdown(f"**시장 상태:** {status}")

with st.expander("⚙️ 조건 설정", expanded=False):
    min_price = st.number_input("최소 주가 ($)", min_value=0.1, value=0.5, step=0.1)
    min_adv = st.number_input("최소 평균 거래대금 ($M)", min_value=1, value=20, step=5)
    min_gap = st.number_input("최소 갭 상승률 (%)", value=1.0, step=.5)
    min_pre_rvol = st.number_input("최소 프리마켓 RVOL", min_value=.1, value=1.2, step=.1)
    max_pre_high_dist = st.number_input("프리마켓 고가와 최대 거리 (%)", min_value=.1, value=3.0, step=.5)
    top_n = st.slider("후보 개수", 3, 12, 5)

if st.button("🔎 지금 시가 후보 찾기"):
    st.session_state["scan_open"] = True

if st.session_state.get("scan_open", False):
    with st.spinner("프리마켓/시가 후보를 분석하고 있어요..."):
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
        (df["갭%"] >= min_gap) &
        (df["프리RVOL"] >= min_pre_rvol) &
        (df["프리고가거리%"] <= max_pre_high_dist)
    )

    out = df[mask].sort_values(["점수","프리RVOL","갭%"], ascending=False).head(top_n)

    st.subheader("오늘의 시가 후보")

    if out.empty:
        st.info("현재 조건을 모두 통과한 종목이 없어요. 갭/RVOL/고가거리 조건을 조금 낮춰보세요.")
        out = df.sort_values("점수", ascending=False).head(3)
        st.caption("참고: 조건 미충족이지만 점수가 높은 종목")

    for rank, (_, r) in enumerate(out.iterrows(), start=1):
        arrow = "▲" if r["갭%"] >= 0 else "▼"
        extra = []
        if r["전일고가돌파"]:
            extra.append("전일고가 돌파")
        if r["시가위"]:
            extra.append("시가 위")
        if r["VWAP위"]:
            extra.append("VWAP 위")
        if r["15분돌파"]:
            extra.append("15분 돌파")
        extra_text = " · ".join(extra) if extra else "프리마켓 조건 중심"

        st.markdown(
            f"""
            <div class="radar-card">
              <div class="ticker">{rank}. {r['티커']}</div>
              <div class="price">${r['현재가']:.2f} · {arrow} 갭 {r['갭%']:+.2f}%</div>
              <div>
                <span class="badge">프리 RVOL {r['프리RVOL']:.2f}x</span>
                <span class="badge">프리 고가거리 {r['프리고가거리%']:.2f}%</span>
                <span class="badge">점수 {r['점수']:.1f}</span>
              </div>
              <div class="reason">{r['이유']}</div>
              <div class="reason">{extra_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    ticker = st.selectbox("차트 확인", out["티커"].tolist())
    chart = symbol_frame(intra, ticker)
    if not chart.empty and "Close" in chart.columns:
        st.line_chart(chart[["Close"]].dropna().tail(100), height=260)

st.caption("※ Yahoo Finance 비공식 시세를 사용하므로 지연·누락 가능성이 있습니다. 실제 주문 전에는 증권사 시세를 확인하세요.")
st.caption("※ 시가 레이더 점수는 후보를 거르는 규칙일 뿐 수익 가능성을 보장하지 않습니다.")
