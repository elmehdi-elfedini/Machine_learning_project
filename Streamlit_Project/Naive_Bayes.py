#############################################################
##  Systeme pour la classification le genre  des fleures  ##
#         en appliquant  Naive Bayes Classifier          #
#########################################################
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score
import time

def naive_bayes_page():
    ################ la partie dessin de l'application #################"
    st.title("Machine Learning: **:blue[Naive Bayes Classifier]** ") 
    st.image("images/dataset-cover.png")
    st.header("Iris DataSet")
    ######################### fin de la partie dessin ###############

    #loading data from csv to a pandas Dataframe
    iris_data = pd.read_csv("data/Iris.csv")
    iris_data = iris_data.iloc[:,1:] #iloc[nombre des ligne , nombre des columns]
    expander_data = st.expander("Voir le DataSet")
    expander_data.dataframe(iris_data)
    #l'étape 1 :  Collecte et prétraitement des données
    iris_data_non_null = iris_data.where((pd.notnull(iris_data)),'') #ce code pour remplacer les valeurs nulles par des String  nulle
    #l'étape 2 : l'étiqutage des donnée
    #on donne {Iris-setosa:0 ; Iris-virginica : 1 ; Iris-versicolor :2}
    iris_data_non_null.loc[iris_data_non_null['Species']=='Iris-setosa','Species',] =0  #se code permet de rechercher sur la column Category et si = spam en remplace par 0
    iris_data_non_null.loc[iris_data_non_null['Species']=='Iris-virginica','Species',] =1  #se code permet de rechercher sur la column Category et si = spam en remplace par 0
    iris_data_non_null.loc[iris_data_non_null['Species']=='Iris-versicolor','Species',] =2  #se code permet de rechercher sur la column Category et si = spam en remplace par 0
    X = iris_data_non_null.drop('Species', axis=1)
    Y = iris_data_non_null['Species']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3) #80% train et 20% test
    Y_train = Y_train.astype('int') #pour convertir type objet vers entier
    Y_test = Y_test.astype('int')
    model = GaussianNB() #l'instanciation du modèle 
    model.fit(X_train,Y_train) #l'entrainement du modèle
    #l'étape 6 : la prédiction sur les données d'entraînement
    prediction_on_trainnig_data = model.predict(X_train)
    accuracy_on_training_data = accuracy_score(Y_train,prediction_on_trainnig_data)
    #l'étape 7 : la prédiction sur les données de Test

    prediction_on_testing_data = model.predict(X_test)
    accuracy_on_testing_data = accuracy_score(Y_test,prediction_on_testing_data)
    ############# Formulaire #################
    SepalLengthCm = st.number_input('Insérer la Longueur du sépale en Cm')
    st.write(SepalLengthCm)
    SepalWidthCm = st.number_input('Insérer la Largeur du Sépale en Cm')
    st.write(SepalWidthCm)
    PetalLengthCm = st.number_input('Insérer la Longueur du Petal en Cm')
    st.write(PetalLengthCm)
    PetalWidthCm = st.number_input('Insérer la Largeur du Petal en Cm')
    st.write(PetalWidthCm)
    button = st.button("Classifier")
    if button:
        input_data = [[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]]
        prediction = model.predict(input_data)
        prediction_int = prediction.item()  # Extraire la valeur entière du ndarray
        prediction_dict = {0: 'Iris-setosa', 1: 'Iris-virginica', 2: 'Iris-versicolor'}
        predicted_class = prediction_dict[prediction_int]
        st.code("La Class prédicté est :")
        c1, c2  = st.columns((2,2))
            
        with c1:
            expander = st.expander(" ",expanded=True)
            if prediction_int == 0:
                expander.image("images/setosa.jpg",width=380)
                expander.code(predicted_class)
            elif prediction_int ==1:
                expander.image("images/virginica.jpg",width=380)
                expander.code(predicted_class)
            else:
                expander.image("images/versicolor.jpg",width=380)  
                expander.code(predicted_class)

        with c2:
            st.write('#### Accuracy Training =  ',accuracy_on_training_data)
            time.sleep(0.3)
            st.write('#### Accuracy Testing =  ',accuracy_on_testing_data)
            time.sleep(0.3)
            st.write("\n")
            st.write('#### Longueur du sépale en cm =  ',SepalLengthCm)
            time.sleep(0.3)
            st.write('#### Largeur du sépale en cm =  ',SepalWidthCm)
            time.sleep(0.3)
            st.write('#### Longueur du petal en cm =  ',PetalLengthCm)
            time.sleep(0.3)
            st.write('#### Largeur du petal en cm =  ',PetalWidthCm)
            time.sleep(0.3)

            st.code("Bon Model")



            
            





