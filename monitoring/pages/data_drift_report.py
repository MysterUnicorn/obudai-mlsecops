import streamlit as st
import streamlit.components.v1 as components

evidently = st.container()
with evidently:
    HtmlFile = open("data_drift_report.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=2000, scrolling=True)