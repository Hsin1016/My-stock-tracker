import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="我的美股監控站", layout="wide")

# 1. 建立 Google Sheets 連線 (需替換成你的網址)
# 注意：正式部署時建議將網址放在 Streamlit Secrets 中
url = "https://docs.google.com/spreadsheets/d/1aVmTqnYP0GbWJNKUeH1ugQ8jqwrR7hk6O5RyzmP4U84/edit?gid=0#gid=0" 
conn = st.connection("gsheets", type=GSheetsConnection)

# 讀取現有資料
df = conn.read(spreadsheet=url)

st.title("📈 美股投資組合 (雲端同步版)")

# 2. 側邊欄：新增與移除功能
with st.sidebar:
    st.header("功能選單")
    
    # --- 新增股票表單 ---
    with st.form("add_form", clear_on_submit=True):
        st.subheader("➕ 新增股票")
        new_ticker = st.text_input("股票代號").upper()
        new_type = st.selectbox("分類", ["持有", "觀察"])
        new_target = st.number_input("目標價/規則價", value=0.0)
        if st.form_submit_button("儲存至雲端"):
            if new_ticker:
                new_row = pd.DataFrame([{"Ticker": new_ticker, "Type": new_type, "Target": new_target}])
                updated_df = pd.concat([df, new_row], ignore_index=True)
                conn.update(spreadsheet=url, data=updated_df)
                st.success(f"{new_ticker} 已儲存！")
                st.rerun()
    
    # --- 移除股票功能 ---
    st.subheader("🗑️ 移除股票")
    if not df.empty:
        ticker_to_remove = st.selectbox("選擇要移除的代號", df['Ticker'].unique())
        if st.button("確認移除"):
            updated_df = df[df['Ticker'] != ticker_to_remove]
            conn.update(spreadsheet=url, data=updated_df)
            st.warning(f"{ticker_to_remove} 已移除")
            st.rerun()

# 3. 主畫面：自動分析
if not df.empty:
    tickers = df['Ticker'].unique().tolist()
    with st.spinner('抓取報價中...'):
        prices = yf.download(tickers, period="1d")['Close'].iloc[-1]
        # 如果只有一支股票，yf 回傳格式會不同，需處理
        if len(tickers) == 1:
            prices = pd.Series({tickers[0]: prices})

    for cat in ["持有", "觀察"]:
        st.subheader(f"📍 {cat}清單")
        sub_df = df[df['Type'] == cat].copy()
        sub_df['最新價格'] = sub_df['Ticker'].map(prices)
        
        # 簡單分析規則
        def analyze_rule(row):
            if row['最新價格'] >= row['Target'] and row['Target'] > 0:
                return "🎯 達標"
            return "⏳ 監控中"
        
        sub_df['狀態'] = sub_df.apply(analyze_rule, axis=1)
        st.dataframe(sub_df, use_container_width=True)
else:
    st.info("目前清單是空的，請從左側新增。")
