# ====================================================================================================
# IMPORT

import streamlit as st
import streamlit_antd_components as sac

import pandas as pd
import numpy as np

from pathlib import Path
import joblib

from naive_bayes import MultinomialNaiveBayes   # for naive_bayes model
# ====================================================================================================
# INIT

# Load Pre-trained Models
naive_bayes = joblib.load(Path('./Model/naive_bayes.sav'))
sklearn_naive_bayes = joblib.load(Path('./Model/sklearn_naive_bayes.sav'))

# Config Streamlit
st.set_page_config(page_title='CS114-GR1', page_icon=':material/mood:', layout='wide')
# ====================================================================================================
# HEADER

st.title('Ứng Dụng Phân Loại Đánh Giá Môn Học')
st.divider()
# ====================================================================================================
# SELECTION PANEL

col1, col2 = st.columns(2)
# Left - Algorithm
with col1:
    algo = sac.segmented(
        items=[
            sac.SegmentedItem(label='SVM'),
            sac.SegmentedItem(label='Naive Bayes'),
            sac.SegmentedItem(label='PhoBert'),
        ],
        label='### 📖 Chọn Phương Pháp Phân Loại', color='#F7931E', align='left', return_index=True,
    )
    
    skl = sac.switch(label='<font color="#29ABE2">**Scikit-learn**</font>', align='left', size='md', on_color='blue')

# Right - Input Type
with col2:
    input_opt = sac.segmented(
        items=[
            sac.SegmentedItem(label='File (CSV)', icon='file-earmark-excel'),
            sac.SegmentedItem(label='Sentence', icon='alphabet-uppercase'),
        ],
        label='### 🔗 Chọn Cách Nhập Dữ Liệu', color='green', align='left', return_index=True,
    )
# ====================================================================================================
# FILE

if input_opt == 0:
    uploaded_file = st.file_uploader("**Chọn tệp CSV**", type=["csv"])
    
    # Người dùng đã upload file
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # --------------------------------------------------------------------------------------------
        # Bắt đầu tính toán
        
        # Naive Bayes (Manually)
        if algo == 1 and skl == False:
            df['sentiment predict'] = naive_bayes.predict(df['sentence'])

        # Naive Bayes (Scikit-learn)
        if algo == 1 and skl == True:
            df['sentiment predict'] = sklearn_naive_bayes.predict(df['sentence'])
        # --------------------------------------------------------------------------------------------
        
        st.divider()
        st.write('### Output')
        # In ra kết quả
        st.dataframe(df)
# ====================================================================================================
# 1 Sentence

if input_opt == 1:
    with st.form("my_form"):
        text_input = st.text_area(label='**Nhập một câu đánh giá:**', height=80)
        submit = st.form_submit_button('Submit')
        
    # Người dùng đã nhập câu
    if submit and text_input:
        df = pd.DataFrame(columns=['sentence'])
        df.loc[0] = [text_input]
        # --------------------------------------------------------------------------------------------
        # Bắt đầu tính toán
        # Naive Bayes (Manually)
        if algo == 1 and skl == False:
            df['sentiment predict'] = naive_bayes.predict(df['sentence'])

        # Naive Bayes (Scikit-learn)
        if algo == 1 and skl == True:
            df['sentiment predict'] = sklearn_naive_bayes.predict(df['sentence'])
        # --------------------------------------------------------------------------------------------
        
        st.divider()
        st.write('### Output')
        # In ra kết quả
        text_res = df.loc[0]['sentiment predict']
        if text_res == 0:
            st.write("0 - Negative (Tiêu cực)")
        if text_res == 1:
            st.write('1 - Neutral (Trung lập)')
        if text_res == 2:
            st.write("2 - Positive (Tích cực)")
# ====================================================================================================