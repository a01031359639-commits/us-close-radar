
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="미장 시가 레이더 2.0",
    page_icon="🌅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

NY = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")

TICKERS = [
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
.block-container{max-width:760px;padding-top:.8rem;padding-left:.85rem;padding-right:.85rem;padding-bottom:2rem;}
h1{font-size:1.7rem !important;margin-bottom:.25rem !important;}
.stButton>button{width:100%;min-height:3rem;font-size:1rem;font-weight:700;border-radius:.8rem;}
.card{border:1px solid rgba(128,128,128,.25);border-radius:1rem;padding:.85rem .9rem;margin:.55rem 0;}
.ticker{font-size:1.3rem;font-weight:800;}
.price{font-size:1.05rem;font-weight:700;}
.reason{font-size:.88rem;opacity:.82;margin-top:.25rem;}
.badge{display:inline-block;padding:.18rem .45rem;border-radius:.55rem;border:1px solid rgba(128,128,128,.28);margin:.15rem .12rem .1rem 0;font-size:.78rem;}
</style>
""", unsafe_allow_html=True)

def session_info(now_ny):
    """뉴욕 현지시간 기준으로 세션을 자동 구분."""
    t = now_ny.time().replace(tzinfo=None)
    wd = now_ny.weekday()

    # 토요일은 휴장. 일요일 20:00 ET부터 overnight 시작 가능.
    if wd == 5:
        return "휴장", "주말"
    if wd == 6 and t < dtime(20, 0):
        return "휴장", "주말"

    if t >= dtime(20, 0) or t < dtime(4, 0):
        return "day", "데이마켓(야간/오버나이트)"
    if dtime(4, 0) <= t < dtime(9, 30):
        return "pre", "프리마켓"
    if dtime(9, 30) <= t < dtime(10, 0):
        return "open", "정규장 시작 직후"
    if dtime(10, 0) <= t < dtime(16, 0):
        return "regular", "정규장"
    return "after", "애프터마켓"

def local_session_window(mode, now_ny):
    """현재 세션에 해당하는 ET 시간창 반환."""
    d = now_ny.date()
    t = now_ny.time().replace(tzinfo=None)

    if mode == "day":
        if t >= dtime(20,0):
            start = datetime.combine(d, dtime(20,0), NY)
            end = datetime.combine(d + timedelta(days=1), dtime(4,0), NY)
        else:
            start = datetime.combine(d - timedelta(days=1), dtime(20,0), NY)
            end = datetime.combine(d, dtime(4,0), NY)
    elif mode == "pre":
        start = datetime.combine(d, dtime(4,0), NY)
        end = datetime.combine(d, dtime(9,30), NY)
    elif mode in ("open","regular"):
        start = datetime.combine(d, dtime(9,30), NY)
        end = datetime.combine(d, dtime(16,0), NY)
    elif mode == "after":
        start = datetime.combine(d, dtime(16,0), NY)
        end = datetime.combine(d, dtime(20,0), NY)
    else:
        start, end = None, None
    return start, end

@st.cache_data(ttl=90, show_spinner=False)
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

def sf(data, symbol):
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

def n(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def to_ny_index(df):
    if df.empty:
        return df
    x = df.copy()
    idx = x.index
    try:
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC").tz_convert(NY)
        else:
            idx = idx.tz_convert(NY)
    except Exception:
        return x
    x.index = idx
    return x

def last_completed_daily(daily):
    d = daily.dropna(subset=["Close"]).copy()
    if len(d) < 22:
        return None
    # intraday 중이면 마지막 일봉이 오늘일 수도 있어서, 날짜가 오늘이면 제외
    today_ny = datetime.now(NY).date()
    try:
        if d.index[-1].date() == today_ny:
            d = d.iloc[:-1]
    except Exception:
        pass
    if len(d) < 21:
        return None
    return d

def analyze(symbol, intra, daily, mode, now_ny):
    i = to_ny_index(sf(intra, symbol))
    d = last_completed_daily(sf(daily, symbol))
    if i.empty or d is None or d.empty:
        return None
    needed_i = {"Open","High","Low","Close","Volume"}
    needed_d = {"Open","High","Low","Close","Volume"}
    if not needed_i.issubset(i.columns) or not needed_d.issubset(d.columns):
        return None

    prev_close = n(d["Close"].iloc[-1])
    prev_high = n(d["High"].iloc[-1])
    if not np.isfinite(prev_close) or prev_close <= 0:
        return None

    hist20 = d.tail(20)
    avg_vol20 = n(hist20["Volume"].mean())
    avg_price20 = n(hist20["Close"].mean())
    avg_dollar = avg_vol20 * avg_price20 if np.isfinite(avg_vol20) and np.isfinite(avg_price20) else np.nan

    start, end = local_session_window(mode, now_ny)
    if start is None:
        return None

    ses = i[(i.index >= start) & (i.index < end)]
    if ses.empty:
        return None

    ses_close = ses["Close"].dropna()
    if ses_close.empty:
        return None

    last_ts = ses_close.index[-1]
    last_price = n(ses_close.iloc[-1])
    ses_high = n(ses["High"].max())
    ses_low = n(ses["Low"].min())
    ses_vol = n(ses["Volume"].fillna(0).sum())
    if not np.isfinite(last_price):
        return None

    # 데이터가 실제로 지금 세션까지 따라오고 있는지 체크.
    age_min = abs((now_ny - last_ts.to_pydatetime()).total_seconds()) / 60.0
    fresh = age_min <= 35

    gap = (last_price / prev_close - 1) * 100
    high_dist = (ses_high - last_price) / ses_high * 100 if np.isfinite(ses_high) and ses_high > 0 else np.nan

    # 세션별 평균 일거래량 대비 대략적 상대거래량 지표.
    # 데이마켓/프리마켓은 거래량 비중이 날마다 크게 달라 참고값으로만 사용.
    share = {"day":0.08, "pre":0.12, "open":0.18, "regular":0.60, "after":0.10}.get(mode, 0.12)
    rvol = ses_vol / (avg_vol20 * share) if np.isfinite(avg_vol20) and avg_vol20 > 0 else np.nan

    # 정규장 시작 이후 전용 지표
    above_open = False
    above_vwap = False
    orb15 = False
    open_change = np.nan

    reg_start = datetime.combine(now_ny.date(), dtime(9,30), NY)
    reg_end = datetime.combine(now_ny.date(), dtime(16,0), NY)
    reg = i[(i.index >= reg_start) & (i.index < reg_end)]

    if mode in ("open","regular") and not reg.empty:
        reg_close = reg["Close"].dropna()
        if not reg_close.empty:
            reg_price = n(reg_close.iloc[-1])
            open_px = n(reg["Open"].dropna().iloc[0]) if not reg["Open"].dropna().empty else np.nan
            if np.isfinite(open_px) and open_px > 0:
                open_change = (reg_price / open_px - 1) * 100
                above_open = reg_price >= open_px

            typ = (reg["High"] + reg["Low"] + reg["Close"]) / 3
            vol = reg["Volume"].fillna(0)
            vwap = n((typ * vol).sum() / vol.sum()) if vol.sum() > 0 else np.nan
            above_vwap = bool(np.isfinite(vwap) and reg_price >= vwap)

            first15 = reg.iloc[:3]
            if len(first15) >= 3:
                first15_high = n(first15["High"].max())
                orb15 = bool(np.isfinite(first15_high) and reg_price >= first15_high)

            last_price = reg_price
            gap = (last_price / prev_close - 1) * 100
            high_dist = (n(reg["High"].max()) - last_price) / n(reg["High"].max()) * 100 if n(reg["High"].max()) > 0 else np.nan

    score = 0.0
    reasons = []

    if np.isfinite(gap):
        score += np.clip(gap, -5, 12) * 2
        if gap >= 2:
            reasons.append(f"전일종가 대비 +{gap:.1f}%")

    if np.isfinite(rvol):
        score += np.clip((rvol - 1) * 10, -5, 28)
        if rvol >= 1.5:
            reasons.append(f"상대거래량 {rvol:.1f}x")

    if np.isfinite(high_dist):
        score += max(0, 18 - max(high_dist, 0) * 6)
        if 0 <= high_dist <= 1:
            reasons.append("세션 고점 근접")

    if last_price >= prev_high:
        score += 15
        reasons.append("전일 고가 돌파")

    if mode in ("open","regular"):
        if above_open:
            score += 8
            reasons.append("시가 위")
        if above_vwap:
            score += 10
            reasons.append("VWAP 위")
        if orb15:
            score += 16
            reasons.append("15분 고가 돌파")
        if np.isfinite(open_change):
            score += np.clip(open_change * 6, -8, 15)

    return {
        "티커": symbol,
        "현재가": last_price,
        "등락%": gap,
        "RVOL": rvol,
        "고가거리%": high_dist,
        "평균거래대금": avg_dollar,
        "점수": float(score),
        "전일고가돌파": bool(last_price >= prev_high),
        "시가위": above_open,
        "VWAP위": above_vwap,
        "15분돌파": orb15,
        "이유": " · ".join(reasons[:4]) if reasons else "조건 약함",
        "신선도분": age_min,
        "신선": fresh,
        "마지막시세": last_ts,
    }

now_ny = datetime.now(NY)
now_kst = datetime.now(KST)
mode, label = session_info(now_ny)

st.title("🌅 미장 시가 레이더 2.0")
st.caption("토스 데이마켓 · 프리마켓 · 정규장 시작 직후를 자동 구분")

c1, c2 = st.columns(2)
c1.metric("뉴욕", now_ny.strftime("%H:%M"))
c2.metric("한국", now_kst.strftime("%H:%M"))
st.markdown(f"**현재 세션:** {label}")

if mode == "day":
    st.info("데이마켓(오버나이트) 시간입니다. Yahoo 24/5 데이터가 앱에 실제로 들어오는 경우에만 후보를 표시합니다. 시세가 오래됐으면 결과를 막습니다.")
elif mode == "pre":
    st.info("프리마켓 후보를 찾는 시간입니다.")
elif mode == "open":
    st.success("정규장 시작 직후입니다. 시가·VWAP·15분 돌파 조건을 함께 봅니다.")
elif mode == "regular":
    st.warning("정규장 진행 중입니다. 이 앱은 개장 직후 후보 확인용으로 쓰는 편이 적합합니다.")
elif mode == "after":
    st.info("애프터마켓입니다. 다음 거래일 시가 후보 준비용으로만 참고하세요.")
else:
    st.warning("현재는 미국 주식 휴장 시간입니다.")

with st.expander("⚙️ 조건 설정", expanded=False):
    min_price = st.number_input("최소 주가 ($)", min_value=1.0, value=5.0, step=1.0)
    min_adv = st.number_input("최소 평균 거래대금 ($M)", min_value=1, value=20, step=5)
    min_move = st.number_input("최소 전일종가 대비 상승률 (%)", value=1.0, step=.5)
    min_rvol = st.number_input("최소 상대거래량 지표", min_value=.1, value=1.2, step=.1)
    max_high_dist = st.number_input("현재 세션 고가와 최대 거리 (%)", min_value=.1, value=3.0, step=.5)
    top_n = st.slider("후보 개수", 3, 12, 5)

disabled = (mode == "closed")
if st.button("🔎 지금 후보 찾기", disabled=disabled):
    st.session_state["scan_v2"] = True

if st.session_state.get("scan_v2", False):
    with st.spinner(f"{label} 시세를 분석하고 있어요..."):
        try:
            intra = fetch_intraday(TICKERS)
            daily = fetch_daily(TICKERS)
        except Exception as e:
            st.error(f"시세를 불러오지 못했어요: {e}")
            st.stop()

        rows = []
        for s in TICKERS:
            try:
                r = analyze(s, intra, daily, mode, now_ny)
                if r:
                    rows.append(r)
            except Exception:
                pass

    if not rows:
        if mode == "day":
            st.warning("현재 Yahoo API에서 데이마켓(오버나이트) 5분봉을 확인하지 못했어요. 이 시간에는 토스 앱의 실제 시세를 우선 확인해주세요.")
        else:
            st.warning("현재 세션의 분석 가능한 시세가 없어요. 잠시 뒤 다시 시도해주세요.")
        st.stop()

    df = pd.DataFrame(rows)

    # 최신 데이터가 아니면 후보 순위를 보여주지 않음.
    fresh_df = df[df["신선"] == True].copy()
    if fresh_df.empty:
        latest = df.sort_values("신선도분").iloc[0]
        ts = latest["마지막시세"]
        st.error(
            f"현재 세션의 최신 데이터가 잡히지 않았어요. 가장 최근 데이터가 약 {latest['신선도분']:.0f}분 전입니다. "
            "오래된 시세로 후보를 만들지 않았습니다."
        )
        if mode == "day":
            st.caption("Yahoo Finance는 24/5 시세를 제공하지만 yfinance 경로에서는 오버나이트 5분봉이 누락될 수 있습니다. 실제 데이마켓 매매 전에는 토스증권 시세를 확인하세요.")
        st.stop()

    mask = (
        (fresh_df["현재가"] >= min_price) &
        (fresh_df["평균거래대금"] >= min_adv * 1_000_000) &
        (fresh_df["등락%"] >= min_move) &
        (fresh_df["RVOL"] >= min_rvol) &
        (fresh_df["고가거리%"] <= max_high_dist) &
        (fresh_df["고가거리%"] >= 0)
    )
    out = fresh_df[mask].sort_values(["점수","RVOL","등락%"], ascending=False).head(top_n)

    st.subheader(f"오늘의 {label} 후보")

    if out.empty:
        st.info("현재 조건을 모두 통과한 종목이 없어요.")
        ref = fresh_df.sort_values("점수", ascending=False).head(3)
        if not ref.empty:
            st.caption("참고: 조건 미충족이지만 점수가 높은 종목")
            out = ref

    for rank, (_, r) in enumerate(out.iterrows(), 1):
        extras = []
        if r["전일고가돌파"]:
            extras.append("전일고가 돌파")
        if r["시가위"]:
            extras.append("시가 위")
        if r["VWAP위"]:
            extras.append("VWAP 위")
        if r["15분돌파"]:
            extras.append("15분 돌파")
        extras_text = " · ".join(extras) if extras else label

        st.markdown(
            f"""
            <div class="card">
              <div class="ticker">{rank}. {r['티커']}</div>
              <div class="price">${r['현재가']:.2f} · 전일종가 대비 {r['등락%']:+.2f}%</div>
              <div>
                <span class="badge">상대거래량 {r['RVOL']:.2f}x</span>
                <span class="badge">고가거리 {r['고가거리%']:.2f}%</span>
                <span class="badge">점수 {r['점수']:.1f}</span>
              </div>
              <div class="reason">{r['이유']}</div>
              <div class="reason">{extras_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if not out.empty:
        symbol = st.selectbox("차트 확인", out["티커"].tolist())
        chart = to_ny_index(sf(intra, symbol))
        if not chart.empty and "Close" in chart.columns:
            st.line_chart(chart[["Close"]].dropna().tail(120), height=260)

st.caption("※ Yahoo Finance의 24/5/확장시간 시세는 누락·지연될 수 있습니다. 앱은 최신 시세가 잡히지 않으면 후보를 표시하지 않도록 되어 있습니다.")
st.caption("※ 실제 주문 전에는 토스증권의 실시간 가격·호가를 확인하세요. 레이더 점수는 수익 가능성을 뜻하지 않습니다.")
