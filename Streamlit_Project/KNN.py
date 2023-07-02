#############################################################
##  Systeme pour la classification le genre  des fleures  ##
#         en appliquant  KNN ALgorithme  Classifier      #
#########################################################

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
################ la partie dessin de l'application #################"

def knn_classifier_page():
    st.title("Machine Learning: **:blue[KNN Classifier]** ") 
    st.header("Iris DataSet")
    #convertir data de csv en  a pandas Dataframe
    iris_data = pd.read_csv("data/Iris.csv")
    iris_data = iris_data.iloc[:,1:] #iloc[nombre des ligne , nombre des columns]
    expander_data = st.expander("Voir le DataSet")
    expander_data.table(iris_data)
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
    k = 5
    model = KNeighborsClassifier(n_neighbors=k) #l'instanciation du modèle 
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
            st.write('#### Accuracy Testing =  ',accuracy_on_testing_data)
            st.write("\n")
            st.write('#### Longueur du sépale en cm =  ',SepalLengthCm)
            st.write('#### Largeur du sépale en cm =  ',SepalWidthCm)
            st.write('#### Longueur du petal en cm =  ',PetalLengthCm)
            st.write('#### Largeur du petal en cm =  ',PetalWidthCm)
            st.code("Bon Model")






















                
        code = '''
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score


        #Data Collection & Pre-Proccessing

        #loading data from csv to a pandas Dataframe

        mail_data = pd.read_csv('../data/mail_data.csv')
        # print(mail_data)

        #replace the null values with a null String 
        mail_data_not_null = mail_data.where((pd.notnull(mail_data)),'')
        #cheaking the numbers of rows and columns in the dataframe
        # print(mail_data_not_null.shape)
        #label encoding 

        #Ham mail ghan3tiweh 1 et Spam ghan3tiweh 0
        mail_data_not_null.loc[mail_data_not_null['Category']=='spam','Category',] =0  #ghanmchiw l columns Categori ga3 les spam ghanremplaciwhom b 0
        mail_data_not_null.loc[mail_data_not_null['Category']=='ham','Category',]  =1  #ghanghemplaciw ham par 1
        print(mail_data_not_null)
        #séparation des données
        X = mail_data_not_null['Message']
        Y = mail_data_not_null['Category']
        print(mail_data_not_null)
        #split the data into test and train

        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3) #80% train et 20% test
        print(X.shape)
        print(X_test.shape)
        print(X_train.shape)

        #########"Feature Extraction "

        #transform the text data to feature vector that can be used as input to the Logistic regression 
        feature_extraction = TfidfVectorizer(min_df = 1,stop_words='english')
        #On veut le convertir X_train into numéticale data
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test) #bla makandir fit f test
        #daba hna 3dna la valeur dial Y_train et Y_test le type dilahom object bghina nhawlohom l des entier
        Y_train = Y_train.astype('int')
        Y_test = Y_test.astype('int')
        print(X_train_features)

        #training the model 
        model = LogisticRegression()
        #training the logistic Regression model with the trainning data
        model.fit(X_train_features,Y_train)

        # #prediction en trainning data

        prediction_on_trainnig_data = model.predict(X_train_features)
        accuracy_on_training_data = accuracy_score(Y_train,prediction_on_trainnig_data)
        print("accuracy in training data ",accuracy_on_training_data)

        # #prediction en testin data

        prediction_on_testing_data = model.predict(X_test_features)
        accuracy_on_testing_data = accuracy_score(Y_test,prediction_on_testing_data)
        print("accuracy in testing data ",accuracy_on_testing_data)

        # ##### building a predict system 
        input_mail = ["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
        #convert this input into feature extraction 
        input_mail_feature = feature_extraction.transform(input_mail)
        #### make prediction
        prediction = model.predict(input_mail_feature)
        print(prediction)
        if prediction == 0:
            print("Spam")
        else:
            print("Ham")
            '''
        expander_data = st.expander("Voir le Code Source")
        expander_data.code(code)

 
