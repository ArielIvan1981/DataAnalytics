# -*- coding: utf-8 -*-
"""
Ejemplo de EDA: Analisis Exploratorio de Datos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#importo archivo csv desde una url
url = 'https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv'

#Creo un DF
df = pd.read_csv(url, sep=";")#index_col=0

print(df.head(5))#imprimo primero 5 lineas

#Conocer informacion basica
print('Cantidad de Filas y columnas:',df.shape)
print('Nombre columnas:',df.columns)

#Columnas, nulos y tipos de datos
df.info()
#estadisticas basicas de feature numerica
df.describe()
print(df.describe())
#Matriz de Correlacion
#Verifiquemos si hay correlacion entre los datos
corr = df.set_index('alpha_3').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()#En este caso vemos baja correlación entre las variables. Dependiendo del algoritmo que utilicemos
#podría ser una buena decisión eliminar features que tuvieran alta correlación

#cargamos una segunda fuente de datos
url = 'https://raw.githubusercontent.com/DrueStaples/Population_Growth/master/countries.csv'
df_pop = pd.read_csv(url)
print(df_pop.head(5))
print('Cantidad de Filas y columnas:',df_pop.shape)
#Aqui vemos la población año tras año de España
df_pop_es = df_pop[df_pop["country"] == 'Spain' ]
print(df_pop_es.head())
df_pop_es.shape

#Visualicemos datos
df_pop_es.drop(['country'],axis=1)['population'].plot(kind='bar')

#Gráfica comparativa de crecimiento poblacional entre España y Argentina entre los años 1952 al
#2007
df_pop_ar = df_pop[(df_pop["country"] == 'Argentina')]
df_pop_ar.head()
df_pop_ar.shape
df_pop_ar.set_index('year').plot(kind='bar')

#Comparativa entre dos paises
anios = df_pop_es['year'].unique()
pop_ar = df_pop_ar['population'].values
pop_es = df_pop_es['population'].values
df_plot = pd.DataFrame({'Argentina': pop_ar,'Spain': pop_es},index=anios)
df_plot.plot(kind='bar')

#Filtracion por paises hispanohablantes
df_espanol = df.replace(np.nan, '', regex=True)
df_espanol = df_espanol[ df_espanol['languages'].str.contains('es') ]
print(df_espanol)

df_espanol.shape

#Visualicemos por población
df_espanol.set_index('alpha_3')[['population','area']].plot(kind='bar',rot=65,figsize=(20,10))

#Detección de Outliers
#Vamos a hacer detección de Outliers²², (con fines educativos) en este caso definimos como limite
#superior (e inferior) la media más (menos) “2 veces la desviación estándar” que muchas veces es
#tomada como máximos de tolerancia
anomalies = []
# Funcion ejemplo para detección de outliers
def find_anomalies(data):
    # Set upper and lower limit to 2 standard deviation
    data_std = data.std()
    data_mean = data.mean()
    anomaly_cut_off = data_std * 2
    lower_limit = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    print(lower_limit.iloc[0])
    print(upper_limit.iloc[0])
    
    # Generate outliers
    for index, row in data.iterrows():
        outlier = row # # obtener primer columna
        # print(outlier)
        if (outlier.iloc[0] > upper_limit.iloc[0]) or (outlier.iloc[0] < lower_limit.iloc[0]):
            anomalies.append(index)
    return anomalies

find_anomalies(df_espanol.set_index('alpha_3')[['population']])

# Quitemos BRA y USA por ser outlies y volvamos a graficar:
df_espanol.drop([30,233], inplace=True)
df_espanol.set_index('alpha_3')[['population','area']].plot(kind='bar',rot=65,figsize=(20,10))

#Graficamos ordenando por tamaño Población
df_espanol.set_index('alpha_3')[['population','area']].sort_values(["population"]).plot(kind='bar',rot=65,figsize=(20,10))

#Grafica sin outliers
#Visualizacion por area
df_espanol.set_index('alpha_3')[['area']].sort_values(["area"]).plot(kind='bar',rot=65,figsize=(20,10))

# En este caso, podriamos quitar por "lo bajo", area menor a 110.000 km2:
df_2 = df_espanol.set_index('alpha_3')
df_2 = df_2[df_2['area'] > 110000]
df_2
df_2[['area']].sort_values(["area"]).plot(kind='bar',rot=65,figsize=(20,10))

"""En pocos minutos hemos podido responder: cuántos datos tenemos, si hay nulos, los tipos de datos
(entero, float, string), la correlación, hicimos visualizaciones, comparativas, manipulación de datos,
detección de ouliers y volver a graficar."""