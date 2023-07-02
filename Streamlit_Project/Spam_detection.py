#############################################################
##  Systeme pour la détéction si un email est Spam ou non ##
#         en appliquant  la regression Logistique        #
#########################################################

import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def Spam_detection():
    ## C'est just la partie dessin de l'application 
    st.title("Machine Learning \: **:blue[Régression logistique]**  ") 
    st.subheader('Email / message **:blue[Spam]**  classificateur')
    st.image("images/logistique.png")
    ## fin de la partie 1 dessin



    input_msg = [st.text_area("Entrer un message")]
    predict = st.button("prédiction")
    if predict:
        if input_msg == [""]:
            st.warning("Veuiller Entrer un text svp !!")
        else:
            #l'étape 1 :  Collecte et prétraitement des données
            mail_data = pd.read_csv('data/mail_data.csv')   ## chargement de données de csv dans un pandas Dataframe
            mail_data_not_null = mail_data.where((pd.notnull(mail_data)),'') #ce code pour remplacer les valeurs nulles par des String  nulle
            #l'étape 2 : l'étiqutage des donnée
            mail_data_not_null.loc[mail_data_not_null['Category']=='spam','Category',] =0  #se code permet de rechercher sur la column Category et si = spam en remplace par 0
            mail_data_not_null.loc[mail_data_not_null['Category']=='ham','Category',]  =1  #se code permet de rechercher sur la column Category et si = ham en remplace par 1
            #l'étape 3 : la séparation des données
            X = mail_data_not_null['Message']    
            Y = mail_data_not_null['Category']
            #l'étape 4 : en divise les données en test et train
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3) #80% train et 20% test

            #l'étape 5 : c'est de l'extraction des caractéristiques en utilisant TF-IDF
            #l'objectif de cette étape  : transformer les données textuelles en vecteur de caractéristiques pouvant être utilisé comme entrée pour la régression logistique
            feature_extraction = TfidfVectorizer(min_df = 1,stop_words='english') #l'instanciation du la classe TfidfVectorizer
            X_train_features = feature_extraction.fit_transform(X_train) #On veut le convertir X_train en données numériques
            X_test_features = feature_extraction.transform(X_test) #bla makandir fit f test
            #daba hna 3dna la valeur dial Y_train et Y_test le type dilahom object bghina nhawlohom l des entier
            Y_train = Y_train.astype('int') #pour convertir type objet vers entier
            Y_test = Y_test.astype('int')
            model = LogisticRegression() #l'instanciation du modèle 
            model.fit(X_train_features,Y_train) #l'entrainement du modèle

            #l'étape 6 : la prédiction sur les données d'entraînement
            prediction_on_trainnig_data = model.predict(X_train_features)
            accuracy_on_training_data = accuracy_score(Y_train,prediction_on_trainnig_data)


            #l'étape 7 : la prédiction sur les données de Test

            prediction_on_testing_data = model.predict(X_test_features)
            accuracy_on_testing_data = accuracy_score(Y_test,prediction_on_testing_data)
            input_mail_feature = feature_extraction.transform(input_msg) #convertir cette entrée en extraction de caractéristiques
            prediction = model.predict(input_mail_feature)
            if prediction == 0:
                st.warning("Spam")
            else:
                st.success("Not Spam")
                st.balloons()


















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



        




