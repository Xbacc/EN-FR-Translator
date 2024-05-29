from model import *
from inference import *
import streamlit as st

# Streamlit web app
def main():
    st.title("Translate My Text")

    # Input text
    text = st.text_area("Enter the text to translate") + '.'

    # Translate button to translate
    if st.button("Translate"):
        translation = translate(model, text)
        st.markdown(f'<p style="color: green; font-size: 25px;"> \
                    Result: {translation} </p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
