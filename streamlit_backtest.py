import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="股票回測分析", layout="wide", page_icon="📈")
st.markdown("""
<style>
    .main { background-color: #0a0e27; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  指標計算
# ══════════════════════════════════════════════════════
def calc_indicators(df, ma_short, ma_long):
    d = df.copy()
    d['MA_S']   = d['Close'].rolling(ma_short).mean()
    d['MA_L']   = d['Close'].rolling(ma_long).mean()
    d['BB_mid'] = d['Close'].rolling(20).mean()
    std         = d['Close'].rolling(20).std()
    d['BB_up']  = d['BB_mid'] + 2 * std
    d['BB_dn']  = d['BB_mid'] - 2 * std
    delta       = d['Close'].diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    d['RSI']    = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    low9        = d['Low'].rolling(9).min()
    high9       = d['High'].rolling(9).max()
    rsv         = (d['Close'] - low9) / (high9 - low9 + 1e-9) * 100
    d['K']      = rsv.ewm(com=2, adjust=False).mean()
    d['D']      = d['K'].ewm(com=2, adjust=False).mean()
    d['J']      = 3 * d['K'] - 2 * d['D']
    d['Vol_MA'] = d['Volume'].rolling(20).mean()
    return d


def calc_signals(df, cfg):
    n = len(df)
    buy_parts, sell_parts = [], []

    if cfg['ma']:
        mb = pd.Series(False, index=df.index)
        ms = pd.Series(False, index=df.index)
        for i in range(1, n):
            pms, pml = df['MA_S'].iloc[i-1], df['MA_L'].iloc[i-1]
            cms, cml = df['MA_S'].iloc[i],   df['MA_L'].iloc[i]
            if all(pd.notna(x) for x in [pms, pml, cms, cml]):
                if float(pms) < float(pml) and float(cms) >= float(cml): mb.iloc[i] = True
                elif float(pms) > float(pml) and float(cms) <= float(cml): ms.iloc[i] = True
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
        buy_sig = buy_parts[0].copy(); sell_sig = sell_parts[0].copy()
        for b, s in zip(buy_parts[1:], sell_parts[1:]):
            buy_sig &= b; sell_sig &= s
    else:
        buy_sig = buy_parts[0].copy(); sell_sig = sell_parts[0].copy()
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

    entry_price = float(bt.loc[0, 'Close'])

    # ── 策略一：買入持有 ──
    shares_bh = initial_cash / entry_price
    result_bh = [{'date': r['date'], 'total': shares_bh * float(r['Close'])}
                 for _, r in bt.iterrows()]

    # ── 策略二：動態買賣 ──
    cash   = 0.0
    shares = initial_cash / entry_price

    peak_price     = entry_price
    peak_total     = initial_cash
    sell_triggered = [False] * 4
    sell_ratios    = [0.20, 0.30, 0.20, 0.30]

    sell_count      = 0
    in_sell_wave    = False
    trough_price    = None
    tracking_trough = False

    last_cross  = None
    buy_state   = None    # None / 'full' / 'partial'
    u_bottom    = None
    buy_step    = 0

    trades    = []
    result_st = []
    bt['pending']      = None
    bt['pending_data'] = np.empty(len(bt), dtype=object)

    for i, row in bt.iterrows():
        date  = row['date']
        close = float(row['Close'])
        open_ = float(row['Open']) if pd.notna(row.get('Open')) else close
        pend  = bt.loc[i, 'pending']
        pdata = bt.loc[i, 'pending_data']

        # ══ 執行掛單（隔日開盤）══

        if pend == 'sell' and shares > 0 and isinstance(pdata, dict):
            ep     = open_
            sidx   = pdata['sell_idx']
            ratio  = sell_ratios[sidx]
            s_sold = min((peak_total * ratio) / ep, shares)
            amt    = s_sold * ep
            cash  += amt
            shares -= s_sold
            if shares < 0.0001:
                shares = 0.0
            sell_labels = ['賣出1(死叉20%)', '賣出2(跌40%→30%)',
                           '賣出3(跌60%→20%)', '清倉(跌80%→30%)']
            trades.append({
                '日期': date.strftime('%Y-%m-%d'),
                '動作': sell_labels[sidx],
                '策略': '動態買賣',
                '價格': round(ep, 4),
                '股數': round(s_sold, 4),
                '金額': round(amt, 2),
                '現金餘額': round(cash, 2),
                '持股市值': round(shares * ep, 2),
                '總資產': round(cash + shares * ep, 2),
            })
            # 立即更新賣出狀態
            sell_count      = sidx + 1
            in_sell_wave    = True
            buy_state       = None
            buy_step        = 0
            u_bottom        = None
            tracking_trough = True
            trough_price    = close

        elif pend == 'buy' and cash > 0.01 and isinstance(pdata, dict):
            ep      = open_
            bratio  = pdata['buy_ratio']
            bstep   = pdata['buy_step']
            buy_amt = min(cash * bratio, cash)
            buy_sh  = buy_amt / ep
            cash   -= buy_amt
            shares += buy_sh

            buy_label_map = {
                (1.0, 1): '全買回(金叉)',
                (0.20, 1): '買回1-金叉(20%)',
                (0.30, 2): '買回2-漲20%(30%)',
                (1.0,  3): '買回3-漲30%(50%清)',
            }
            buy_label = buy_label_map.get((round(bratio, 2), bstep),
                                          f'買回{int(bratio*100)}%')
            trades.append({
                '日期': date.strftime('%Y-%m-%d'),
                '動作': buy_label,
                '策略': '動態買賣',
                '價格': round(ep, 4),
                '股數': round(buy_sh, 4),
                '金額': round(buy_amt, 2),
                '現金餘額': round(cash, 2),
                '持股市值': round(shares * ep, 2),
                '總資產': round(cash + shares * ep, 2),
            })
            # 全買回後重置所有狀態
            if round(bratio, 2) == 1.0 and bstep == 1:
                sell_count      = 0
                sell_triggered  = [False] * 4
                buy_state       = None
                buy_step        = 0
                u_bottom        = None
                tracking_trough = False
                trough_price    = None
                in_sell_wave    = False

        # ── 更新峰值（永不重置）──
        total_now = cash + shares * close
        if close > peak_price:
            peak_price     = close
            peak_total     = total_now
            sell_triggered = [False] * 4

        # ── 追蹤U底 ──
        if tracking_trough:
            if trough_price is None or close < trough_price:
                trough_price = close

        # ── 判斷 MA 交叉 ──
        cross = None
        if i > 0:
            pms = bt.loc[i-1, 'MA_S']; pml = bt.loc[i-1, 'MA_L']
            cms = bt.loc[i,   'MA_S']; cml = bt.loc[i,   'MA_L']
            if all(pd.notna(x) for x in [pms, pml, cms, cml]):
                pms, pml = float(pms), float(pml)
                cms, cml = float(cms), float(cml)
                if pms < pml and cms >= cml:   cross = 'golden'
                elif pms > pml and cms <= cml: cross = 'death'

        # ══ 賣出觸發判斷 ══
        if shares > 0 and i + 1 < len(bt):
            # 第1次：死叉
            if (not sell_triggered[0] and
                    cross == 'death' and cross != last_cross):
                sell_triggered[0] = True
                bt.at[i+1, 'pending']      = 'sell'
                bt.at[i+1, 'pending_data'] = {'sell_idx': 0}
                tracking_trough = True
                trough_price    = close

            # 第2次：跌到峰值 40%
            if (sell_triggered[0] and not sell_triggered[1] and
                    close <= peak_price * 0.60):
                sell_triggered[1] = True
                bt.at[i+1, 'pending']      = 'sell'
                bt.at[i+1, 'pending_data'] = {'sell_idx': 1}

            # 第3次：跌到峰值 60%
            if (sell_triggered[1] and not sell_triggered[2] and
                    close <= peak_price * 0.40):
                sell_triggered[2] = True
                bt.at[i+1, 'pending']      = 'sell'
                bt.at[i+1, 'pending_data'] = {'sell_idx': 2}

            # 第4次：跌到峰值 80%（清倉）
            if (sell_triggered[2] and not sell_triggered[3] and
                    close <= peak_price * 0.20):
                sell_triggered[3] = True
                bt.at[i+1, 'pending']      = 'sell'
                bt.at[i+1, 'pending_data'] = {'sell_idx': 3}

        # ══ 買回觸發判斷 ══
        if cash > 0.01 and i + 1 < len(bt):

            # 金叉觸發
            if cross == 'golden' and cross != last_cross:
                u_bottom        = trough_price if trough_price else close
                tracking_trough = False

                if sell_count == 1:
                    buy_state = 'full'
                    buy_step  = 1
                    bt.at[i+1, 'pending']      = 'buy'
                    bt.at[i+1, 'pending_data'] = {'buy_ratio': 1.0, 'buy_step': 1}

                elif sell_count >= 2:
                    buy_state = 'partial'
                    buy_step  = 1
                    bt.at[i+1, 'pending']      = 'buy'
                    bt.at[i+1, 'pending_data'] = {'buy_ratio': 0.20, 'buy_step': 1}

            # 分批買回後續
            elif buy_state == 'partial' and u_bottom is not None and cash > 0.01:
                rise = (close - u_bottom) / u_bottom

                if buy_step == 1 and rise >= 0.20:
                    buy_step = 2
                    bt.at[i+1, 'pending']      = 'buy'
                    bt.at[i+1, 'pending_data'] = {'buy_ratio': 0.30, 'buy_step': 2}

                elif buy_step == 2 and rise >= 0.30:
                    buy_step  = 3
                    buy_state = None
                    bt.at[i+1, 'pending']      = 'buy'
                    bt.at[i+1, 'pending_data'] = {'buy_ratio': 1.0, 'buy_step': 3}

        if cross:
            last_cross = cross

        result_st.append({'date': date, 'total': cash + shares * close})

    return (pd.DataFrame(result_bh), pd.DataFrame(result_st),
            pd.DataFrame(trades) if trades else pd.DataFrame(),
            entry_price, bt)


# ══════════════════════════════════════════════════════
#  定期定額回測
# ══════════════════════════════════════════════════════
def run_dca(bt_df, dca_amount, dca_days):
    shares = 0.0; total_invested = 0.0
    dca_trades = []; result_dca = []
    for _, row in bt_df.iterrows():
        date  = row['date']
        close = float(row['Close'])
        if pd.to_datetime(date).day in dca_days:
            buy_sh = dca_amount / close
            shares += buy_sh; total_invested += dca_amount
            dca_trades.append({
                '日期': pd.to_datetime(date).strftime('%Y-%m-%d'),
                '動作': '定期買入', '策略': '定期定額',
                '價格': round(close, 4), '股數': round(buy_sh, 4),
                '金額': round(dca_amount, 2),
                '累計投入': round(total_invested, 2),
                '持股市值': round(shares * close, 2),
                '總資產': round(shares * close, 2),
            })
        result_dca.append({'date': date, 'total': shares * close,
                           'invested': total_invested})
    return pd.DataFrame(result_dca), pd.DataFrame(dca_trades), total_invested


# ══════════════════════════════════════════════════════
#  Plotly 圖表
# ══════════════════════════════════════════════════════
def build_chart(bh_df, st_df, dca_df, trades_df, dca_trades_df,
                bt_df, ticker, company_name, ma_short, ma_long, cfg):

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.48, 0.22, 0.30],
        subplot_titles=[f"{company_name} ({ticker}) 價格走勢", "副指標", "資產對比（三策略）"],
        vertical_spacing=0.06,
    )
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=12, color='#e9ecef')

    dates = bt_df['date']

    # 上圖：收盤價 + MA + BB
    fig.add_trace(go.Scatter(x=dates, y=bt_df['Close'], name='收盤價',
                             line=dict(color='#00d4aa', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bt_df['MA_S'], name=f'MA{ma_short}',
                             line=dict(color='#ff6b6b', width=1.2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bt_df['MA_L'], name=f'MA{ma_long}',
                             line=dict(color='#ffd93d', width=1.2, dash='dash')), row=1, col=1)

    if cfg['bb']:
        fig.add_trace(go.Scatter(x=dates, y=bt_df['BB_up'], name='BB上軌',
                                 line=dict(color='#6366f1', width=0.8, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['BB_dn'], name='BB下軌',
                                 line=dict(color='#6366f1', width=0.8, dash='dot'),
                                 fill='tonexty', fillcolor='rgba(99,102,241,0.05)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['BB_mid'], name='BB中軌',
                                 line=dict(color='#6366f1', width=0.8, dash='dot')), row=1, col=1)

    # 買賣標記
    if not trades_df.empty:
        dyn    = trades_df[trades_df['策略'] == '動態買賣']
        buy_t  = dyn[dyn['動作'].str.contains('買回')]
        sell_t = dyn[~dyn['動作'].str.contains('買回')]

        def get_price(df_t):
            prices = []
            for d in pd.to_datetime(df_t['日期']):
                r = bt_df[bt_df['date'] == d]
                prices.append(float(r['Close'].values[0]) if not r.empty else None)
            return prices

        if not buy_t.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(buy_t['日期']), y=get_price(buy_t),
                mode='markers', name='動態買入',
                marker=dict(symbol='triangle-up', size=12, color='#51cf66'),
            ), row=1, col=1)
        if not sell_t.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sell_t['日期']), y=get_price(sell_t),
                mode='markers', name='動態賣出',
                marker=dict(symbol='triangle-down', size=12, color='#ff6b6b'),
            ), row=1, col=1)

    if not dca_trades_df.empty:
        dca_prices = []
        for d in pd.to_datetime(dca_trades_df['日期']):
            r = bt_df[bt_df['date'] == d]
            dca_prices.append(float(r['Close'].values[0]) if not r.empty else None)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(dca_trades_df['日期']), y=dca_prices,
            mode='markers', name='定期買入',
            marker=dict(symbol='circle', size=7, color='#ffd93d', opacity=0.8),
        ), row=1, col=1)

    # 中圖：副指標
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
                                 line=dict(color='#6366f1', width=1),
                                 opacity=0.7), row=2, col=1)
    else:
        vol_colors = ['#51cf66' if i == 0 or float(bt_df['Close'].iloc[i]) >= float(bt_df['Close'].iloc[i-1])
                      else '#ff6b6b' for i in range(len(bt_df))]
        fig.add_trace(go.Bar(x=dates, y=bt_df['Volume'], name='成交量',
                             marker_color=vol_colors, opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=bt_df['Vol_MA'], name='Vol MA20',
                                 line=dict(color='#ffd93d', width=1, dash='dash')), row=2, col=1)

    # 下圖：三策略資產對比
    fig.add_trace(go.Scatter(x=bh_df['date'], y=bh_df['total'], name='買入持有',
                             line=dict(color='#00d4aa', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=st_df['date'], y=st_df['total'], name='動態策略',
                             line=dict(color='#6366f1', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=dca_df['date'], y=dca_df['total'], name='定期定額',
                             line=dict(color='#ffd93d', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=dca_df['date'], y=dca_df['invested'], name='定額累計投入',
                             line=dict(color='#ffd93d', width=1, dash='dot'), opacity=0.5), row=3, col=1)
    fig.add_hline(y=bh_df['total'].iloc[0], line_color='#adb5bd',
                  line_dash='dot', line_width=1, row=3, col=1)

    fig.update_layout(
        height=820,
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        font=dict(color='#e9ecef', size=11),
        legend=dict(
            bgcolor='rgba(26,31,58,0.92)',
            bordercolor='#00d4aa', borderwidth=1,
            font=dict(size=11, color='#e9ecef'),
            orientation='v',
            x=1.01, y=1, xanchor='left', yanchor='top',
            itemwidth=40, tracegroupgap=4,
            title=dict(text='圖例 (可點選)', font=dict(size=11, color='#00d4aa')),
        ),
        margin=dict(l=50, r=160, t=50, b=40),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
    )
    for i in range(1, 4):
        fig.update_xaxes(gridcolor='#2d3250', gridwidth=0.5, showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor='#2d3250', gridwidth=0.5, showgrid=True, row=i, col=1)
    fig.update_yaxes(tickprefix='$', row=3, col=1)
    return fig


# ══════════════════════════════════════════════════════
#  Streamlit UI
# ══════════════════════════════════════════════════════
st.title("📈 股票回測分析")

with st.sidebar:
    st.header("⚙️ 基本參數")
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
    ma_short = st.number_input("短期 MA", value=5,  min_value=1,  max_value=200)
    ma_long  = st.number_input("長期 MA", value=10, min_value=2,  max_value=500)

    st.divider()
    st.subheader("🎯 進出場指標")
    cb_ma  = st.checkbox("MA 交叉",       value=True)
    cb_bb  = st.checkbox("布林通道中軌",   value=False)
    cb_rsi = st.checkbox("RSI (30/70)",   value=False)
    cb_kdj = st.checkbox("KDJ 交叉",      value=False)
    cb_vol = st.checkbox("放量突破",       value=False)

    st.divider()
    st.subheader("🔗 觸發邏輯")
    logic = st.radio("", ["OR（任一指標觸發）", "AND（全部指標同時觸發）"], index=0)

    st.divider()
    st.subheader("💰 定期定額設定")
    dca_on     = st.toggle("啟用定期定額", value=True)
    dca_amount = st.number_input("每次投入金額 (USD)", value=500, step=100, min_value=10)
    st.caption("選擇每月買入日（可複選）")
    dca_day5  = st.checkbox("每月 5 日",  value=True)
    dca_day15 = st.checkbox("每月 15 日", value=False)
    dca_day25 = st.checkbox("每月 25 日", value=False)
    dca_days  = [d for d, on in [(5, dca_day5), (15, dca_day15), (25, dca_day25)] if on]

    st.divider()
    run_btn = st.button("▶ 開始回測", use_container_width=True, type="primary")

# ── 主畫面 ──
if run_btn:
    if ma_short >= ma_long:
        st.error("❌ 短期MA必須小於長期MA"); st.stop()
    if not any([cb_ma, cb_bb, cb_rsi, cb_kdj, cb_vol]):
        st.error("❌ 請至少勾選一個指標"); st.stop()
    if dca_on and not dca_days:
        st.error("❌ 定期定額已啟用，請至少選一個買入日"); st.stop()

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
            try:
                info = yf.Ticker(ticker).info
                company_name = info.get('longName') or info.get('shortName') or ticker
            except Exception:
                company_name = ticker
        except Exception as e:
            st.error(f"❌ 下載失敗：{e}"); st.stop()

    with st.spinner("回測計算中..."):
        res = run_backtest(raw, str(backtest_start), initial_cash, ma_short, ma_long, cfg)
        bh_df, st_df, trades_df, entry_price, bt_df = res

    if bh_df is None:
        st.error("❌ 回測區間內無資料，請確認日期設定"); st.stop()

    dca_df        = pd.DataFrame({'date': st_df['date'], 'total': [0]*len(st_df), 'invested': [0]*len(st_df)})
    dca_trades_df = pd.DataFrame()
    dca_invested  = 0.0
    final_dca     = 0.0

    if dca_on and dca_days:
        dca_df, dca_trades_df, dca_invested = run_dca(bt_df, dca_amount, dca_days)
        final_dca = float(dca_df.iloc[-1]['total'])

    final_bh = bh_df.iloc[-1]['total']
    final_st = st_df.iloc[-1]['total']
    ret_bh   = (final_bh - initial_cash) / initial_cash * 100
    ret_st   = (final_st - initial_cash) / initial_cash * 100
    ret_dca  = (final_dca - dca_invested) / dca_invested * 100 if dca_invested > 0 else 0
    n_trades = len(trades_df) if not trades_df.empty else 0
    n_dca    = len(dca_trades_df) if not dca_trades_df.empty else 0
    days_str = "+".join([f"{d}日" for d in dca_days]) if dca_days else "—"

    st.markdown(f"### 🏢 {company_name}　`{ticker}`")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("進場價格", f"${entry_price:.4f}", ticker)
    with c2:
        st.metric("策略一 買入持有", f"${final_bh:,.2f}", f"{ret_bh:+.2f}%")
    with c3:
        st.metric("策略二 動態買賣", f"${final_st:,.2f}", f"{ret_st:+.2f}%")
    with c4:
        st.metric("策略三 定期定額", f"${final_dca:,.2f}",
                  f"{ret_dca:+.2f}%  |  投入${dca_invested:,.0f}")
    with c5:
        active = [k.upper() for k, v in cfg.items() if v and k != 'logic']
        st.metric("交易次數", f"{n_trades} + {n_dca}",
                  f"動態{n_trades}筆 / 定額{n_dca}筆")

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📊 走勢圖",
        f"📋 動態策略記錄（{n_trades}筆）",
        f"💰 定期定額記錄（{n_dca}筆）",
    ])

    with tab1:
        fig = build_chart(bh_df, st_df, dca_df, trades_df, dca_trades_df,
                          bt_df, ticker, company_name, ma_short, ma_long, cfg)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if trades_df.empty:
            st.warning("⚠️ 回測期間內未觸發任何動態策略交易")
        else:
            def style_dyn(row):
                color = '#51cf66' if '買回' in str(row['動作']) else '#ff6b6b'
                return [f'color: {color}'] * len(row)
            st.dataframe(trades_df.style.apply(style_dyn, axis=1),
                         use_container_width=True, hide_index=True)

    with tab3:
        if dca_trades_df.empty:
            st.warning("⚠️ 定期定額未啟用或無交易")
        else:
            st.dataframe(dca_trades_df.style.apply(
                lambda _: ['color: #ffd93d'] * len(dca_trades_df.columns), axis=1),
                use_container_width=True, hide_index=True)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("總投入",  f"${dca_invested:,.2f}")
            sc2.metric("現值",    f"${final_dca:,.2f}")
            sc3.metric("報酬率",  f"{ret_dca:+.2f}%",
                       f"每月{days_str}各投入${dca_amount:,.0f}")

else:
    st.info("👈 請在左側設定參數後，點擊「開始回測」")
