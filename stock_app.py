import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="我的美股監控站", layout="wide")

# 1. 建立連線 (請確保 url 正確且 Google 表單已開放「知道連結的人可檢視」)
url = "你的Google表單網址"
conn = st.connection("gsheets", type=GSheetsConnection)

# 讀取現有資料並強制轉換類型，避免比較時報錯
df = conn.read(spreadsheet=url)
if not df.empty:
    df['Target'] = pd.to_numeric(df['Target'], errors='coerce').fillna(0)

st.title("📈 美股投資組合 (雲端同步版)")

# 2. 側邊欄：新增與移除
with st.sidebar:
    st.header("功能選單")
    with st.form("add_form", clear_on_submit=True):
        st.subheader("➕ 新增股票")
        new_ticker = st.text_input("股票代號").upper().strip()
        new_type = st.selectbox("分類", ["持有", "觀察"])
        new_target = st.number_input("目標價/規則價", value=0.0)
        if st.form_submit_button("儲存至雲端"):
            if new_ticker:
                new_row = pd.DataFrame([{"Ticker": new_ticker, "Type": new_type, "Target": new_target}])
                updated_df = pd.concat([df, new_row], ignore_index=True)
                conn.update(spreadsheet=url, data=updated_df)
                st.success(f"{new_ticker} 已儲存！")
                st.rerun()
    
    st.subheader("🗑️ 移除股票")
    if not df.empty:
        ticker_to_remove = st.selectbox("選擇要移除的代號", df['Ticker'].unique())
        if st.button("確認移除"):
            updated_df = df[df['Ticker'] != ticker_to_remove].reset_index(drop=True)
            conn.update(spreadsheet=url, data=updated_df)
            st.warning(f"{ticker_to_remove} 已移除")
            st.rerun()

# 3. 主畫面：抓取報價與分析
if not df.empty:
    tickers = df['Ticker'].unique().tolist()
    with st.spinner('抓取最新報價中...'):
        # 修正單一股票與多支股票的抓取邏輯
        try:
            price_df = yf.download(tickers, period="1d")['Close']
            if isinstance(price_df, pd.DataFrame):
                latest_prices = price_df.iloc[-1].to_dict()
            else:
                # 只有一支股票時返回的是 Series
                latest_prices = {tickers[0]: price_df.iloc[-1]}
        except Exception as e:
            st.error(f"抓取報價失敗: {e}")
            latest_prices = {}

    for cat in ["持有", "觀察"]:
        st.subheader(f"📍 {cat}清單")
        sub_df = df[df['Type'] == cat].copy()
        
        if not sub_df.empty:
            # 帶入價格
            sub_df['最新價格'] = sub_df['Ticker'].map(latest_prices)
            
            # 安全的分析規則：增加類型檢查
            def analyze_rule(row):
                curr = row['最新價格']
                tgt = row['Target']
                if pd.isna(curr) or curr == 0:
                    return "無報價"
                if tgt > 0 and curr >= tgt:
                    return "🎯 達標"
                return "⏳ 監控中"
            
            sub_df['狀態'] = sub_df.apply(analyze_rule, axis=1)
            # 格式化顯示價格（小數點兩位）
            st.dataframe(sub_df.style.format({"最新價格": "{:.2f}", "Target": "{:.2f}"}), use_container_width=True)
else:
    st.info("目前清單是空的，請從左側新增。")
