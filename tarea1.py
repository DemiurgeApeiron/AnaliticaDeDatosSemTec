import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import tensorflow as tf
from sklearn.impute import KNNImputer

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
x = df.iloc[:, [2, 3, 4, 8, 9]]
y = df.iloc[:, [11]]
print(df.head(5))
print(df.info(5))
print(df.describe())
print(df.shape)
print(f"y: {y.head(5)}")
print(f"x: {x.head(5)}")
print(x.isna().sum())
imputer = KNNImputer(n_neighbors=5)
x_knnImputed = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)
print(x_knnImputed.head(5))
print(x_knnImputed.isna().sum())


fig, axs = plt.subplots(2)
fig.suptitle("AGE")
axs[0].boxplot(x_knnImputed["age"])
axs[1].hist(x_knnImputed["age"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig2, axs2 = plt.subplots(2)
fig2.suptitle("Hypertension")
axs2[0].boxplot(x_knnImputed["hypertension"])
axs2[1].hist(x_knnImputed["hypertension"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig3, axs3 = plt.subplots(2)
fig3.suptitle("heart_disease")
axs3[0].boxplot(x_knnImputed["heart_disease"])
axs3[1].hist(x_knnImputed["heart_disease"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig4, axs4 = plt.subplots(2)
fig4.suptitle("avg_glucose_level")
axs4[0].boxplot(x_knnImputed["avg_glucose_level"])
axs4[1].hist(x_knnImputed["avg_glucose_level"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig5, axs5 = plt.subplots(2)
fig5.suptitle("bmi")
axs5[0].boxplot(x_knnImputed["bmi"])
axs5[1].hist(x_knnImputed["bmi"], bins=range(0, 90, 5), color="#ff1c1c1c", ec="yellow")

fig6 = plt.figure()
plt.title("heatMap")
sns.heatmap(x_knnImputed.head(20), cmap="RdYlGn", linewidths=0.30, annot=True)

plt.show()

measuresOfCentralTendency = x_knnImputed.describe()
print(f"measuresOfCentralTendency: \n {measuresOfCentralTendency}")


print(f"medina de AGE: {x_knnImputed['age'].median()}")
print(f"moda de AGE: {x_knnImputed['age'].mode()}")
print(f"medina de Hypertension: {x_knnImputed['hypertension'].median()}")
print(f"moda de Hypertension: {x_knnImputed['hypertension'].mode()}")
print(f"medina de heart_disease: {x_knnImputed['heart_disease'].median()}")
print(f"moda de heart_disease: {x_knnImputed['heart_disease'].mode()}")
print(f"medina de avg_glucose_level: {x_knnImputed['avg_glucose_level'].median()}")
print(f"moda de avg_glucose_level: {x_knnImputed['avg_glucose_level'].mode()}")
print(f"medina de bmi: {x_knnImputed['bmi'].median()}")
print(f"moda de bmi: {x_knnImputed['bmi'].mode()}")

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
