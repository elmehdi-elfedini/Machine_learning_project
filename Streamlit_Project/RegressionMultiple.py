#############################################################
##  Systeme pour la classification le genre  des fleures  ##
#         en appliquant  Naive Bayes Classifier          #
########################################################
import streamlit as st
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler as SC
from sklearn.metrics import r2_score




def regression_Multiple():
        ################ la partie dessin de l'application #################"
    st.title("Machine Learning: **:blue[House price prediction]** ") 
    st.header("USA_Housing DataSet")
    ######################### fin de la partie dessin ###############
    data = pd.read_csv("data/USA_Housing.csv")
    expander_data = st.expander("Voir le DataSet")
    expander_data.dataframe(data)
    X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
    Y = data[['Price']]
    # Y =(Y-np.min(Y))/(np.max(Y)-np.min(Y))
    ### Standarisation des donnés
    scaler=SC()
    Y=scaler.fit_transform(Y)
    X=scaler.fit_transform(X)
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3) #80% train et 20% test
    model = LinearRegression()
    model.fit(X_train,Y_train)
    y_predict = model.predict(X_test)
    ###Calcule R2
    R2=r2_score(Y,model.predict(X))
    print(R2)
    n=len(X)
    p=len(X[0,:])
    R2_ajuste=1-(n-1)/(n-1-p)*(1-R2)
    print(R2_ajuste)

    # print(model.score(X_train,Y_train))
    # ######### Lasso
    # lasso_reg = linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)
    # lasso_reg.fit(X_train,Y_train)
    # print(lasso_reg.score(X_test,Y_test))
    # print(lasso_reg.score(X_train,Y_train))
    # ######### Ridge
    # Ridge_reg = linear_model.Ridge(alpha=50,max_iter=100,tol=0.1)
    # Ridge_reg.fit(X_train,Y_train)
    # print(Ridge_reg.score(X_test,Y_test))
    # print(Ridge_reg.score(X_train,Y_train))
    c3, c4 = st.columns((7,3))
    with c3:
        tabl_val = []
        st.markdown('### Faire une prédiction')

        for i in data:
            tabl_val.append(i)
        for i in range(len(tabl_val)):
            if tabl_val[i] =='Address':
                st.write("")
            elif tabl_val[i] =='Price':
                st.write("")
            else:
                tabl_val[i] =st.number_input("Entrer {}".format(tabl_val[i]))

        submit = st.button("submit")
        with c4:
             if submit:
                st.markdown('### Le prix prédicté est : ')
                st.code(str(model.predict([[int(tabl_val[0]),int(tabl_val[1]),int(tabl_val[2]),int(tabl_val[3]),int(tabl_val[4])]]))+" Dh")
                st.write("R2_ajusté = ",R2_ajuste)
                st.write("R2 Score = ",R2)






