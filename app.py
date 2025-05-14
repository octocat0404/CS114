# Ứng Dụng Phân Loại Đánh Giá Môn Học
#* IMPORT ==========================================================================================
import streamlit as st
import streamlit_antd_components as sac

from pathlib import Path
import joblib

import re
import pandas as pd
from naive_bayes import MultinomialNaiveBayes

#* FUNCTIONS =======================================================================================
stop_words_vi = set(["là", "và", "một", "có", "những", "cho", "được", "tại", "với"])

def remove_stopwords_vi(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words_vi]
    return ' '.join(filtered_words)

def clean_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

#* INIT ============================================================================================
#? Config Streamlit
st.set_page_config(page_title='CS114-GR1', page_icon=':material/ar_stickers:', layout='wide')

#? Load pre-trained models
@st.cache_resource
def load_naive_bayes():
    return joblib.load(Path('./model/naive_bayes.sav'))

@st.cache_resource
def load_naive_bayes_sklearn():
    return joblib.load(Path('./model/naive_bayes_sklearn.sav'))

naive_bayes = load_naive_bayes()
naive_bayes_sklearn = load_naive_bayes_sklearn()

naive_bayes_result = joblib.load(Path('./eval/naive_bayes.sav'))
naive_bayes_sklearn_result = joblib.load(Path('./eval/naive_bayes_sklearn.sav'))
#* PAGE HEADER =====================================================================================
st.title(':orange[Ứng Dụng Phân Loại Đánh Giá của Học Viên 🎓]')
st.write('\n')
st.write('\n')
#* INTRODUCTION ====================================================================================
intro_col1, intro_col2 = st.columns(2)

with intro_col1:
    st.write('### :blue[Giới Thiệu Ứng Dụng]')
    st.write('* **Đây là một ứng dụng máy học nhằm hỗ trợ phân loại các câu đánh giá, bình luận trong lĩnh vực giáo dục.**')
    st.write('* **Ứng dụng được xây dựng dựa trên bộ dữ liệu [UIT-VSFC](https://www.researchgate.net/publication/329645066_UIT-VSFC_Vietnamese_Students_Feedback_Corpus_for_Sentiment_Analysis)**')
    st.write('* **Hiện tại, ứng dụng chỉ hỗ trợ đầy đủ với ngôn ngữ Tiếng Việt.**')

with intro_col2:
    st.write('### :blue[Mô Tả Chức Năng]')
    st.write('* **Phân loại đánh giá theo 3 Cảm Xúc: :green[Tích Cực], :red[Tiêu Cực], :blue[Trung Lập]**')
    st.write('* **Phân loại đánh giá theo 4 Chủ Đề: :blue[Giảng Viên], :orange[Chương Trình Đào Tạo], :green[Cơ Sở Vật Chất], :red[Khác]**')
    st.write('* **Phân loại một hoặc hàng loạt nhiều câu đánh giá, hỗ trợ xuất kết quả ra file CSV.**')

st.divider()

#* INPUT HANDLING ==================================================================================
input_col1, input_col2 = st.columns(2)

#? Các lựa chọn phương pháp phân loại
with input_col1:
    st.write('### :green[01) Chọn phương pháp phân loại]')
    
    algo = sac.segmented(
        items=[
            sac.SegmentedItem(label='SVM'),
            sac.SegmentedItem(label='Naive Bayes'),
        ],
        label='', color='#F7931E', align='left', return_index=True,
    )

    if algo == 0 or algo == 1:
        skl = sac.switch(label='<font color="#29ABE2">**Scikit-learn**</font>', align='left', size='md', on_color='blue')
    
    st.write('**Classification Report**')
    #TODO: Thêm đánh giá mô hình 
    if algo == 0:
        pass
    
    #? Naive Bayes report without Scikit-learn
    if algo == 1 and skl == False:
        st.dataframe(naive_bayes_result)
        
    #? Naive Bayes report with Scikit-learn
    if algo == 1 and skl == True:
        st.dataframe(naive_bayes_sklearn_result)

#? Các lựa chọn cách nhập dữ liệu
with input_col2:
    st.write('### :green[02) Chọn cách nhập dữ liệu]')
    df = pd.DataFrame(columns=['sentence'])
    
    input_opt = sac.segmented(
        items=[
            sac.SegmentedItem(label='File', icon='filetype-csv'),
            sac.SegmentedItem(label='Bảng', icon='table'),
            sac.SegmentedItem(label='Câu', icon='alphabet-uppercase'),
        ],
        label='', color='green', align='left', return_index=True, 
    )
    
    input_file_submit = False
    input_table_submit = False
    input_sentence_submit = False
    
    if input_opt == 0:
        st.write('* **Chọn một tệp có đuôi .csv chứa các câu đánh giá:**')
        
        input_file_col1, input_file_col2 = st.columns([0.025, 0.975])
        with input_file_col2:
            uploaded_file = st.file_uploader('', type=['csv'], label_visibility='collapsed')
            
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write('* **Preview 5 dòng trong file:**')
            
            input_file_col1, input_file_col2 = st.columns([0.025, 0.975])
            with input_file_col2:
                st.dataframe(df.sample(5))
                
            if len(df.columns) > 1:
                st.write('* **Chọn cột chứa các câu đánh giá:**')
                
                input_file_col1, input_file_col2 = st.columns([0.025, 0.975])
                with input_file_col2:
                    sentence_column = st.selectbox('', df.columns, label_visibility='collapsed')
                    input_file_submit = st.button('Submit')
        
    if input_opt == 1:
        st.write('**Nhập các câu đánh giá vào bảng:**')
        with st.form('input_table_form'):
            input_table = st.data_editor(df, num_rows='dynamic')
            df = input_table
            df = df.dropna()
            input_table_submit = st.form_submit_button('Submit')
            
    if input_opt == 2:
        st.write('**Nhập một câu đánh giá:**')
        with st.form('input_sentence_form'):
            input_sentence = st.text_area(label='', height=120, label_visibility='collapsed')
            input_sentence_submit = st.form_submit_button('Submit')

st.divider()
#* OUTPUT ==========================================================================================
if input_file_submit or input_table_submit or input_sentence_submit:
    st.write('### :green[Output]')

    if input_opt == 0:
        df['sentence (processed)'] = df[sentence_column].apply(remove_stopwords_vi)
        df['sentence (processed)'] = df[sentence_column].apply(clean_whitespace)

    if input_opt == 1:
        df['sentence (processed)'] = df['sentence'].apply(remove_stopwords_vi)
        df['sentence (processed)'] = df['sentence'].apply(clean_whitespace)


    if input_opt == 2:
        df.loc[0] = input_sentence
        df['sentence (processed)'] = df['sentence'].apply(remove_stopwords_vi)
        df['sentence (processed)'] = df['sentence'].apply(clean_whitespace)

    # Naive Bayes (Manually)
    if algo == 1 and skl == False:
        df['sentiment predict'] = naive_bayes.predict(df['sentence (processed)'])

    # Naive Bayes (Scikit-learn)
    if algo == 1 and skl == True:
        df['sentiment predict'] = naive_bayes_sklearn.predict(df['sentence (processed)'])
    
    df = df.drop(columns=['sentence (processed)'])    
    df = df.reindex(sorted(df.columns), axis=1)
    
    #? Hiển thị kết quả
    if input_opt == 0 or input_opt == 1:
        st.write('\n')
        st.dataframe(df)
        
    if input_opt == 2:
        sentiment_text_result = df['sentiment predict'].loc[0]
        
        if sentiment_text_result == 0:
            # st.write('**0 - Tiêu Cực (Negative)**')
            sac.result(label='**0**', description='**Tiêu Cực (Negative)**', status='error', icon='emoji-angry-fill')
            
        if sentiment_text_result == 1:
            # st.write('**1 - Trung Lập (Neutral)**')
            sac.result(label='**1**', description='**Trung Lập (Neutral)**', status='info', icon='emoji-neutral-fill')
            
        if sentiment_text_result == 2:
            # st.write('**2 - Tích Cực (Positive)**')
            sac.result(label='**2**', description='**Tích Cực (Positive)**', status='success', icon='emoji-heart-eyes-fill')