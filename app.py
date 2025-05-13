# Ứng Dụng Phân Loại Đánh Giá Môn Học
#* IMPORT ==========================================================================================
import streamlit as st
import streamlit_antd_components as sac

from pathlib import Path
import joblib

import pandas as pd
from naive_bayes import MultinomialNaiveBayes

#* INIT ============================================================================================
#? Config Streamlit
st.set_page_config(page_title='CS114-GR1', page_icon=':material/mood:', layout='wide')

#? Load pre-trained models
@st.cache_resource
def load_naive_bayes():
    return joblib.load(Path('./model/naive_bayes.sav'))

@st.cache_resource
def load_naive_bayes_sklearn():
    return joblib.load(Path('./model/naive_bayes_sklearn.sav'))

#* PAGE HEADER =====================================================================================
st.title(':orange[Ứng Dụng Phân Loại Đánh Giá Của Học Viên]')
st.divider()

#* INTRODUCTION ====================================================================================
intro_col1, intro_col2 = st.columns(2)

with intro_col1:
    st.write('### :blue[01) Giới Thiệu Ứng Dụng]')
    st.write('* **Đây là một ứng dụng máy học nhằm hỗ trợ phân loại các câu đánh giá, bình luận trong lĩnh vực giáo dục.**')
    st.write('* **Ứng dụng này được xây dựng dựa trên bộ dữ liệu [UIT-VSFC](https://www.researchgate.net/publication/329645066_UIT-VSFC_Vietnamese_Students_Feedback_Corpus_for_Sentiment_Analysis)**')
    st.write('* **Hiện tại ứng dụng chỉ hỗ trợ đầy đủ với ngôn ngữ Tiếng Việt.**')

with intro_col2:
    st.write('### :blue[02) Mô Tả Chức Năng]')
    st.write('* **Phân loại đánh giá theo cảm xúc: :green[Tích Cực], :red[Tiêu Cực], :blue[Trung Lập]**')