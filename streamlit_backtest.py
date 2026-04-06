import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="股票回測分析", layout="wide", page_icon="📈")

# ── 樣式 ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e27; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1a1f3a;
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-title { color: #adb5bd; font-size: 13px; margin-bottom: 4px; }
    .metric-value { font-size: 26px; font-weight: bold; }
    .metric-sub   { color: #adb5bd; font-size: 12px; margin-top: 4px; }
    .positive { color: #51cf66; }
    .negative { color: #ff6b6b; }
    .neutral  { color: #e9ecef; }
</style>
""", unsafe_allow_html=True)

TRIGGER_PCT = 0.20

# ══════════════════════════════════════════════════════
#  指標計算
# ══════════════════════════════════════════════════════
def calc_indicators(df, ma_short, ma_long):
    d = df.copy()
    d['MA_S'] = d['Close'].rolling(ma_short).mean()
    d['MA_L'] = d['Close'].rolling(ma_long).mean()
    d['BB_mid'] = d['Close'].rolling(20).mean()
    std = d['Close'].rolling(20).std()
    d['BB_up'] = d['BB_mid'] + 2 * std
    d['BB_dn'] = d['BB_mid'] - 2 * std
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    low9  = d['Low'].rolling(9).min()
    high9 = d['High'].rolling(9).max()
    rsv   = (d['Close'] - low9) / (high9 - low9 + 1e-9) * 100
    d['K'] = rsv.ewm(com=2, adjust=False).mean()
    d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    d['J'] = 3 * d['K'] - 2 * d['D']
    d['Vol_MA'] = d['Volume'].rolling(20).mean()
    return d


def calc_signals(df, cfg):
    n = len(df)
    buy_parts, sell_parts = [], []

    if cfg['ma']:
        mb, ms = pd.Series(False, index=df.index), pd.Series(False, index=df.index)
        for i in range(1, n):
            pms, pml = df['MA_S'].iloc[i-1], df['MA_L'].iloc[i-1]
            cms, cml = df['MA_S'].iloc[i],   df['MA_L'].iloc[i]
            if all(pd.notna(x) for x in [pms, pml, cms, cml]):
                if pms < pml and cms >= cml: mb.iloc[i] = True
                elif pms > pml and cms <= cml: ms.iloc[i] = True
        buy_parts.append(mb); sell_parts.append(ms)

    if cfg['bb']:
        bb = (df['Close'].shift(1) < df['BB_mid'].shift(1)) & (df['Close'] >= df['BB_mid'])
        bs = (df['Close'].shift(1) > df['BB_mid'].shift(1)) & (df['Close'] <  df['BB_mid'])
        buy_parts.append(bb.fillna(False)); sell_parts.append(bs.fillna(False))

    if cfg['rsi']:
        rb = (df['RSI'].shift(1) < 30) & (df['RSI'] >= 30)
        rs = (df['RSI'].shift(1) > 70) & (df['RSI'] <= 70)
        buy_parts.append(rb.fillna(False)); sell_parts.append(rs.fillna(False))

    if cfg['kdj']:
        kb = (df['K'].shift(1) < df['D'].shift(1)) & (df['K'] >= df['D'])
        ks = (df['K'].shift(1) > df['D'].shift(1)) & (df['K'] <  df['D'])
        buy_parts.append(kb.fillna(False)); sell_parts.append(ks.fillna(False))

    if cfg['vol']:
        vb = (df['Close'] > df['BB_up'].shift(1)) & (df['Volume'] > df['Vol_MA'] * 1.5)
        vs = (df['Close'] < df['BB_dn'].shift(1)) & (df['Volume'] > df['Vol_MA'] * 1.5)
        buy_parts.append(vb.fillna(False)); sell_parts.append(vs.fillna(False))

    if not buy_parts:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    if cfg['logic'] == 'AND':
        buy_sig  = buy_parts[0].copy()
        sell_sig = sell_parts[0].copy()
        for b, s in zip(buy_parts[1:], sell_parts[1:]):
            buy_sig &= b; sell_sig &= s
    else:
        buy_sig  = buy_parts[0].copy()
        sell_sig = sell_parts[0].copy()
        for b, s in zip(buy_parts[1:], sell_parts[1:]):
            buy_sig |= b; sell_sig |= s

    return buy_sig, sell_sig


# ══════════════════════════════════════════════════════
#  回測引擎
# ══════════════════════════════════════════════════════
def run_backtest(df, backtest_start, initial_cash, ma_short, ma_long, cfg):
    df = calc_indicators(df, ma_short, ma_long)
    bt = df[df.index >= backtest_start].copy().reset_index()
    bt.rename(columns={'Date': 'date'}, inplace=True)
    bt = bt.reset_index(drop=True)
    if bt.empty:
        return None, None, None, None, None

    buy_sig, sell_sig = calc_signals(bt, cfg)
    entry_price = float(bt.loc[0, 'Close'])
    shares_bh   = initial_cash / entry_price
    result_bh   = [{'date': r['date'], 'total': shares_bh * float(r['Close'])}
                   for _, r in bt.iterrows()]

    cash = 0.0; shares = initial_cash / entry_price
    sell_count = 0; last_buy = False; last_sell = False
    trades = []; result_st = []
    bt['pending'] = None

    for i, row in bt.iterrows():
        date  = row['date']
        close = float(row['Close'])
        open_ = float(row['Open']) if pd.notna(row.get('Open')) else close
        pend  = bt.loc[i, 'pending']

        if pend == 'sell' and shares > 0:
            ep = open_
            if sell_count >= 2:
                s_sold = shares; amt = s_sold * ep
                cash += amt; shares = 0.0
                label = '清倉 (第3次)'; sell_count = 3
            else:
                s_sold = shares * TRIGGER_PCT; amt = s_sold * ep
                cash += amt; shares -= s_sold; sell_count += 1
                label = f'賣出{sell_count}'
            trades.append({'日期': date.strftime('%Y-%m-%d'), '動作': label,
                           '價格': round(ep, 4), '股數': round(s_sold, 4),
                           '金額': round(amt, 2), '現金餘額': round(cash, 2),
                           '持股市值': round(shares * ep, 2),
                           '總資產': round(cash + shares * ep, 2)})

        elif pend == 'buy' and cash > 0:
            ep = open_
            buy_amt = cash * TRIGGER_PCT; buy_sh = buy_amt / ep
            cash -= buy_amt; shares += buy_sh
            sell_count = max(0, sell_count - 1)
            trades.append({'日期': date.strftime('%Y-%m-%d'), '動作': '買回',
                           '價格': round(ep, 4), '股數': round(buy_sh, 4),
                           '金額': round(buy_amt, 2), '現金餘額': round(cash, 2),
                           '持股市值': round(shares * ep, 2),
                           '總資產': round(cash + shares * ep, 2)})

        is_sell = bool(sell_sig.iloc[i])
        is_buy  = bool(buy_sig.iloc[i])
        if is_sell and not last_sell and shares > 0 and i + 1 < len(bt):
            bt.at[i + 1, 'pending'] = 'sell'
        elif is_buy and not last_buy and cash > 0 and i + 1 < len(bt):
            bt.at[i + 1, 'pending'] = 'buy'
        last_sell = is_sell; last_buy = is_buy

        result_st.append({'date': date, 'total': cash + shares * close,
                          'cash': cash, 'shares': shares})

    return (pd.DataFrame(result_bh), pd.DataFrame(result_st),
            pd.DataFrame(trades) if trades else pd.DataFrame(),
            entry_price, bt)


# ══════════════════════════════════════════════════════
#  Plotly 圖表
# ══════════════════════════════════════════════════════
def build_chart(bh_df, st_df, trades_df, bt_df, ticker, ma_short, ma_long, cfg):
    rows = 3
    row_heights = [0.5, 0.25, 0.25]
    subplot_titles = [f"{ticker} 價格走勢", "副指標", "資產對比"]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        row_heights=row_heights,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.06)

    dates = bt_df['date']

    # ── 上圖：K線 + MA + BB ──
    fig.add_trace(go.Candlestick(
        x=dates, open=bt_df['Open'], high=bt_df['High'],
        low=bt_df['Low'],  close=bt_df['Close'],
        name='K線',
        increasing_line_color='#51cf66', decreasing_line_color='#ff6b6b',
        increasing_fillcolor='#51cf66', decreasing_fillcolor='#ff6b6b',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=bt_df['MA_S'], name=f'MA{ma_short}',
                             line=dict(color='#ff6b6b', width=1.2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bt_df['MA_L'], name=f'MA{ma_long}',
                             line=dict(color='#ffd93d', width=1.2, dash='dash')), row=1, col=1)

    if cfg['bb']:
        fig.add_trace(go.Scatter(x=dates, y=bt_df['BB_up'], name='BB上軌',
                                 line=dict(color='#6366f1', width=0.8, dash='dot'),
                                 showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['BB_dn'], name='BB下軌',
                                 line=dict(color='#6366f1', width=0.8, dash='dot'),
                                 fill='tonexty', fillcolor='rgba(99,102,241,0.05)',
                                 showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['BB_mid'], name='BB中軌',
                                 line=dict(color='#6366f1', width=0.8, dash='dot'),
                                 showlegend=True), row=1, col=1)

    # 買賣標記
    if not trades_df.empty:
        buy_t  = trades_df[trades_df['動作'] == '買回']
        sell_t = trades_df[trades_df['動作'] != '買回']

        def get_price(df_t):
            prices = []
            for d in pd.to_datetime(df_t['日期']):
                r = bt_df[bt_df['date'] == d]
                prices.append(float(r['Close'].values[0]) if not r.empty else None)
            return prices

        if not buy_t.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(buy_t['日期']), y=get_price(buy_t),
                mode='markers', name='買回',
                marker=dict(symbol='triangle-up', size=12, color='#51cf66'),
            ), row=1, col=1)
        if not sell_t.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sell_t['日期']), y=get_price(sell_t),
                mode='markers', name='賣出',
                marker=dict(symbol='triangle-down', size=12, color='#ff6b6b'),
            ), row=1, col=1)

    # ── 中圖：RSI / KDJ / 成交量 ──
    if cfg['rsi']:
        fig.add_trace(go.Scatter(x=dates, y=bt_df['RSI'], name='RSI',
                                 line=dict(color='#00d4aa', width=1.2)), row=2, col=1)
        fig.add_hline(y=70, line_color='#ff6b6b', line_dash='dash', line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_color='#51cf66', line_dash='dash', line_width=0.8, row=2, col=1)
    elif cfg['kdj']:
        fig.add_trace(go.Scatter(x=dates, y=bt_df['K'], name='K',
                                 line=dict(color='#00d4aa', width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['D'], name='D',
                                 line=dict(color='#ffd93d', width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['J'], name='J',
                                 line=dict(color='#6366f1', width=1, opacity=0.7)), row=2, col=1)
    else:
        vol_colors = ['#51cf66' if float(bt_df['Close'].iloc[i]) >= float(bt_df['Close'].iloc[i-1])
                      else '#ff6b6b' for i in range(len(bt_df))]
        fig.add_trace(go.Bar(x=dates, y=bt_df['Volume'], name='成交量',
                             marker_color=vol_colors, opacity=0.7), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['Vol_MA'], name='Vol MA20',
                                 line=dict(color='#ffd93d', width=1, dash='dash')), row=2, col=1)

    # ── 下圖：資產對比 ──
    fig.add_trace(go.Scatter(x=bh_df['date'], y=bh_df['total'], name='買入持有',
                             line=dict(color='#00d4aa', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=st_df['date'], y=st_df['total'], name='動態策略',
                             line=dict(color='#6366f1', width=2)), row=3, col=1)
    fig.add_hline(y=st_df['total'].iloc[0], line_color='#adb5bd',
                  line_dash='dot', line_width=1, row=3, col=1)

    # ── 全局樣式 ──
    fig.update_layout(
        height=780,
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        font=dict(color='#e9ecef', size=11),
        legend=dict(bgcolor='#1a1f3a', bordercolor='#2d3250', borderwidth=1,
                    font=dict(size=10), orientation='h', y=1.02, x=0),
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor='#2d3250', gridwidth=0.5,
                         showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor='#2d3250', gridwidth=0.5,
                         showgrid=True, row=i, col=1)

    fig.update_yaxes(tickprefix='$', row=3, col=1)
    return fig


# ══════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════
st.title("📈 股票回測分析")

# ── 側邊欄：參數設定 ──
with st.sidebar:
    st.header("⚙️ 參數設定")

    market = st.selectbox("市場", ["美股", "台股上市 (.TW)", "台股上櫃 (.TWO)"])
    raw_ticker = st.text_input("股票代號", value="ONDS").strip().upper()
    if market == "台股上市 (.TW)":
        ticker = raw_ticker + ".TW"
    elif market == "台股上櫃 (.TWO)":
        ticker = raw_ticker + ".TWO"
    else:
        ticker = raw_ticker

    backtest_start = st.date_input("回測起始日", value=pd.to_datetime("2025-09-01"))
    initial_cash   = st.number_input("投入金額 (USD)", value=30000, step=1000, min_value=100)

    st.divider()
    st.subheader("📐 MA 設定")
    ma_short = st.number_input("短期 MA", value=5,  min_value=1, max_value=200)
    ma_long  = st.number_input("長期 MA", value=10, min_value=2, max_value=500)

    st.divider()
    st.subheader("🎯 進出場指標")
    cb_ma  = st.checkbox("MA 交叉",        value=True)
    cb_bb  = st.checkbox("布林通道中軌",    value=False)
    cb_rsi = st.checkbox("RSI (30/70)",    value=False)
    cb_kdj = st.checkbox("KDJ 交叉",       value=False)
    cb_vol = st.checkbox("放量突破",        value=False)

    st.divider()
    st.subheader("🔗 觸發邏輯")
    logic = st.radio("", ["OR（任一指標觸發）", "AND（全部指標同時觸發）"], index=0)

    run_btn = st.button("▶ 開始回測", use_container_width=True, type="primary")

# ── 主畫面 ──
if run_btn:
    if ma_short >= ma_long:
        st.error("❌ 短期MA必須小於長期MA")
        st.stop()
    if not any([cb_ma, cb_bb, cb_rsi, cb_kdj, cb_vol]):
        st.error("❌ 請至少勾選一個指標")
        st.stop()

    cfg = {
        'ma': cb_ma, 'bb': cb_bb, 'rsi': cb_rsi,
        'kdj': cb_kdj, 'vol': cb_vol,
        'logic': 'AND' if 'AND' in logic else 'OR',
    }

    with st.spinner(f"下載 {ticker} 資料中..."):
        try:
            start = (pd.to_datetime(str(backtest_start)) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            raw   = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            raw.index = pd.to_datetime(raw.index)
            raw.columns = raw.columns.get_level_values(0)
        except Exception as e:
            st.error(f"❌ 下載失敗：{e}")
            st.stop()

    with st.spinner("回測計算中..."):
        res = run_backtest(raw, str(backtest_start), initial_cash, ma_short, ma_long, cfg)
        bh_df, st_df, trades_df, entry_price, bt_df = res

    if bh_df is None:
        st.error("❌ 回測區間內無資料，請確認日期設定")
        st.stop()

    final_bh = bh_df.iloc[-1]['total']
    final_st = st_df.iloc[-1]['total']
    ret_bh   = (final_bh - initial_cash) / initial_cash * 100
    ret_st   = (final_st - initial_cash) / initial_cash * 100
    n_trades = len(trades_df) if not trades_df.empty else 0

    # ── 摘要卡片 ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("進場價格", f"${entry_price:.4f}",
                  f"{ticker} · MA{ma_short}/MA{ma_long}")
    with c2:
        st.metric("策略一（買入持有）", f"${final_bh:,.2f}",
                  f"{ret_bh:+.2f}%",
                  delta_color="normal")
    with c3:
        st.metric("策略二（動態買賣）", f"${final_st:,.2f}",
                  f"{ret_st:+.2f}%",
                  delta_color="normal")
    with c4:
        active = [k.upper() for k, v in cfg.items() if v and k != 'logic']
        st.metric("交易次數", str(n_trades),
                  f"{', '.join(active)} | {cfg['logic']}")

    st.divider()

    # ── 圖表 ──
    tab1, tab2 = st.tabs(["📊 走勢圖", f"📋 交易記錄（{n_trades} 筆）"])

    with tab1:
        fig = build_chart(bh_df, st_df, trades_df, bt_df,
                          ticker, ma_short, ma_long, cfg)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if trades_df.empty:
            st.warning("⚠️ 回測期間內未觸發任何交易")
        else:
            def style_table(row):
                if '買回' in str(row['動作']):
                    return ['color: #51cf66'] * len(row)
                else:
                    return ['color: #ff6b6b'] * len(row)

            st.dataframe(
                trades_df.style.apply(style_table, axis=1),
                use_container_width=True,
                hide_index=True,
            )

else:
    st.info("👈 請在左側設定參數後，點擊「開始回測」")
