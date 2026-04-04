import streamlit as st
import yfinance as yf

st.title("我的美股第一站")

# 簡單輸入框
ticker = st.text_input("請輸入股票代號 (如: ONDS)", "ONDS")

if st.button("獲取報價"):
    data = yf.Ticker(ticker)
    price = data.fast_info['last_price']
    st.metric(label=f"{ticker} 最新價格", value=f"{price:.2f} USD")
    st.write(f"當前時間的 {ticker} 分析結果...")
