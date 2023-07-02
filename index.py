import streamlit as st
from streamlit_option_menu import option_menu
from Streamlit_Project.KNN import knn_classifier_page
from Streamlit_Project.Naive_Bayes import naive_bayes_page
from Streamlit_Project.Spam_detection import Spam_detection
from Streamlit_Project.Kmeans import kmeans_page
from Streamlit_Project.RegressionMultiple import regression_Multiple


st.set_page_config(page_icon=":bar_chart", page_title="DataMining",  layout="wide")
st.sidebar.header('Midvi  - `DataMining`')

# with open('Strealit_Project/Style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def k_means_page():
    st.write("This is the K-means page")

def main():
    with st.sidebar:
        selected = option_menu(
            menu_title="Machine Learning Project", #required
            options=["KNN Classifier","K-means","Logistic Regression","Naive Bayes","Regression  Multiple"], #required
            # icons = ["house","book","envelope"], #optional icon's name like bootstrap
            default_index=0,
            # orientation= "horizontal",
        )

    if selected == "KNN Classifier":
        knn_classifier_page()
    elif selected == "K-means":
        kmeans_page()
    elif selected == "Logistic Regression":
        Spam_detection()
    elif selected == "Naive Bayes":
        naive_bayes_page()
    elif selected == "Regression  Multiple":
        regression_Multiple()

if __name__ == "__main__":
    main()
