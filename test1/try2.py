# NLP 모델 웹에 배포하기 연습2
import streamlit as st
import requests

def query_hugging_face_api(input_text):
    API_URL = "https://api-inference.huggingface.co/models/hscrown/oliveKobart"
    headers = {"Authorization": "Bearer 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"}
    payload = {"inputs": input_text}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit 앱 UI 설정
st.title('한국어 텍스트 요약 앱')

# 사용자 입력
user_input = st.text_area("텍스트를 입력하세요:", "여기에 텍스트를 입력하면 요약문을 얻을 수 있습니다.")

# 요약 버튼
if st.button('텍스트 요약'):
    # 사용자의 입력으로 함수 호출
    output = query_hugging_face_api(user_input)
    
    # 응답 표시
    st.write("요약문:")
    if output and isinstance(output, list) and 'generated_text' in output[0]:
        st.write(output[0]['generated_text'])  # JSON의 'generated_text' 값을 표시
    else:
        st.write("응답을 처리할 수 없습니다. 응답 구조를 확인해주세요.")
 
