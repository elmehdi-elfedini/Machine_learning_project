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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
plt.style.use("fivethirtyeight")


#loading data from csv file
################ la partie dessin de l'application #################"

def kmeans_page():
    st.title("Machine Learning: **:blue[K-means Clustring]** ") 
    st.header("Mall Customer Segmentation Data")

    #Data Collection & Pre-Proccessing
    data = pd.read_csv("data/Mall_Customers.csv")

    if data:
        #convertir data de csv en  a pandas Dataframe
        customers_data = pd.read_csv(data)
        X = customers_data.iloc[:,[3,4]].values #ghadi nakhdo les colonnes 3 et 4 
        expander_data = st.expander("Voir le DataSet")
        expander_data.table(customers_data)
        #l'étape 1 :  Collecte et prétraitement des données
        st.subheader("Metric Variables : ")
        Salary = st.checkbox('Spending Score (1-100)')
        Age = st.checkbox('Annual Income (k$)')
        st.subheader("Calculate : ")
        algorithme = st.radio("",('K-means Clustring', 'Hierarchical Clustering'))
        if algorithme =='K-means Clustring':
            st.subheader("K-means")
            st.markdown("---")
            # Assuming customers_data is your original DataFrame
            X = customers_data.iloc[:, [3, 4]].values

            # Create a DataFrame using X
            chart_data = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

            # Plot the data using vega_lite_chart
            st.vega_lite_chart(chart_data, {
                'mark': {'type': 'circle', 'tooltip': True},
                'encoding': {
                    'x': {'field': 'Annual Income (k$)', 'type': 'quantitative'},
                    'y': {'field': 'Spending Score (1-100)', 'type': 'quantitative'},
                    'size': {'field': 'Annual Income (k$)', 'type': 'quantitative'},
                    'color': {'field': 'Spending Score (1-100)', 'type': 'quantitative'},
                },
            },use_container_width=True)
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            st.subheader("Elbow Method")
            st.markdown("---")
            chart_data = pd.DataFrame({'Number of Clusters': range(1, 11), 'WCSS': wcss})
            st.line_chart(chart_data)
            Number_cluster = st.number_input("Nombre de Cluster : ",value=4)
            # Assuming X contains your data for clustering
            kmeans = KMeans(n_clusters=int(Number_cluster), init='k-means++', random_state=42)
            y_kmeans = kmeans.fit_predict(X)


            st.subheader("Clusters Custommers")
            st.markdown("---")
            # Create a DataFrame from X and y_kmeans
            df = pd.DataFrame({'Spending Score (1-100)': X[:, 0], 'Annual Income (k$)': X[:, 1], 'Custommers': y_kmeans})

            # Define color and size encodings based on the Custommers labels
            color_scale = alt.Scale(domain=[0, 1, 2, 3], range=['red', 'blue', 'yellow', 'green'])
            size_scale = alt.Scale(domain=[0, 1, 2, 3], range=[80, 80, 80, 80])

            # Create the Vega-Lite chart
            chart = alt.Chart(df).mark_circle().encode(
                x='Spending Score (1-100):Q',
                y='Annual Income (k$):Q',
                color=alt.Color('Custommers:N', scale=color_scale),
                size=alt.Size('Custommers:N', scale=size_scale)
            ).properties(width=600, height=400)

            # Display the chart in Streamlit
            st.altair_chart(chart, use_container_width=True)
        else:
            st.title("Hierarchical Clustering Dendrogram")

            # Define your data for clustering (replace with your actual data)

            # Perform hierarchical clustering
            Z = linkage(X, method='ward')

            # Plot the dendrogram
            fig, ax = plt.subplots(figsize=(10, 6))
            dendrogram(Z)
            plt.xlabel('Data Points',fontsize=14)
            plt.ylabel('Distance', fontsize=14)      
            st.pyplot(fig)

# processus d'initialisation de K-means++ :

# Sélectionnez le premier centroïde au hasard parmi les points de données.

# Pour chaque point de données restant, calculez la distance au carré minimale (distance euclidienne) par rapport à n'importe lequel des centroïdes existants. Cette distance représente la distance du point par rapport au centroïde le plus proche.

# Choisissez le prochain centroïde parmi les points de données restants avec une probabilité proportionnelle à la distance au carré calculée à l'étape précédente. Cela signifie que les points qui sont plus éloignés des centroïdes existants ont une probabilité plus élevée d'être sélectionnés comme prochain centroïde.

# Répétez les étapes 2 et 3 jusqu'à ce que tous les K centroïdes soient sélectionnés.

# En utilisant cette technique d'initialisation, K-means++ a tendance à choisir des centroïdes bien espacés et représentatifs de différentes régions des données. Cela contribue à améliorer la qualité du regroupement et à obtenir des résultats plus fiables.
