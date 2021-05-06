import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
x = df.iloc[:, [2, 3, 4, 8, 9]]
x_cualitavivos = df.iloc[:, [1, 5, 6, 7, 10]]
x_numericos = df.iloc[:, [2, 3, 4, 8, 9]]
y = df.iloc[:, [11]]
print(df.head(5))
print(df.info(5))
print(df.describe())
print(df.shape)
print(f"y: {y.head(5)}")
print(f"x: {x.head(5)}")
print(x.isna().sum())
imputer = KNNImputer(n_neighbors=5)
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)
print(x.head(5))
print(x.isna().sum())

measuresOfCentralTendency = x.describe()
print(f"measuresOfCentralTendency: \n {measuresOfCentralTendency}")
print(f"medina de AGE: {x['age'].median()}")
print(f"moda de AGE: {x['age'].mode()}")
print(f"medina de Hypertension: {x['hypertension'].median()}")
print(f"moda de Hypertension: {x['hypertension'].mode()}")
print(f"medina de heart_disease: {x['heart_disease'].median()}")
print(f"moda de heart_disease: {x['heart_disease'].mode()}")
print(f"medina de avg_glucose_level: {x['avg_glucose_level'].median()}")
print(f"moda de avg_glucose_level: {x['avg_glucose_level'].mode()}")
print(f"medina de bmi: {x['bmi'].median()}")
print(f"moda de bmi: {x['bmi'].mode()}")

print(x.shape)
x = x[y["stroke"] == 1]
print(x.shape)

plt.scatter(x["age"], x["avg_glucose_level"])
plt.show()

# K-Means Clustering

# Carga del conjunto de datos
X = x.iloc[:, [0, 3]].values  # se analiza una matriz formada por las columnas 3 y 4
# print(X)

# Metodo del Codo para encontrar el numero optimo de clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # la inercia que vendría siendo la suma
    # total del cuadrado dentro del clúster.

# Grafica de la suma de las distancias
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Creando el k-Means para los 5 grupos encontrados
kmeans = KMeans(n_clusters=2, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualizacion grafica de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c="red", label="Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c="blue", label="Cluster 2")
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c="green", label="Cluster 3")
plt.show()

"""fig, axs = plt.subplots(2)
fig.suptitle("AGE")
axs[0].boxplot(x["age"])
axs[1].hist(x["age"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig2, axs2 = plt.subplots(2)
fig2.suptitle("Hypertension")
axs2[0].boxplot(x["hypertension"])
axs2[1].hist(x["hypertension"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig3, axs3 = plt.subplots(2)
fig3.suptitle("heart_disease")
axs3[0].boxplot(x["heart_disease"])
axs3[1].hist(x["heart_disease"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig4, axs4 = plt.subplots(2)
fig4.suptitle("avg_glucose_level")
axs4[0].boxplot(x["avg_glucose_level"])
axs4[1].hist(x["avg_glucose_level"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig5, axs5 = plt.subplots(2)
fig5.suptitle("bmi")
axs5[0].boxplot(x["bmi"])
axs5[1].hist(x["bmi"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig6 = plt.figure()
plt.title("heatMap")
sns.heatmap(x.head(20), cmap="RdYlGn", linewidths=0.30, annot=True)

plt.show()"""
"""
        1. ¿Hay alguna variable que no aporta información?
            work_type,
            ever_married,
            Residence_type
        2. Si tuvieras que eliminar variables, ¿cuáles quitarías y por qué?
            work_type: no agreaga valor mas aya de factores de riesgo externos
            ever_married: no agrega valor, ya que no aparenta tener una correlacion fuerte
        3. ¿Existen variables que tengan datos extraños?
            No
        4. Si comparas las variables, ¿todas están en rangos similares? ¿Crees que esto afecte?

        5. ¿Puedes encontrar grupos qué se parezcan? ¿Qué grupos son estos?
 """
