import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import datetime

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

# Deep learning
# imprimimos los valores para entrenar a el modelo
print(f"x_cualitavivos: {x_cualitavivos.head(5)}")
print(f"x_numericos: {x_numericos.head(5)}")

# De nuevo rellenamos los valores NaN
print(x_cualitavivos.isna().sum())
print(x_numericos.isna().sum())
imputer = KNNImputer(n_neighbors=5)
x_numericos = pd.DataFrame(imputer.fit_transform(x_numericos), columns=x_numericos.columns)
print(x_numericos.isna().sum())

# vamos a hacer One-Hot encoding a los valores cualitavivos para poder alimentarlos al modelo

X = pd.concat([x_numericos, pd.get_dummies(x_cualitavivos)], axis=1)
print(X.head(5))
print(X.info(5))

# Now we divide the problem as a train set and a test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# now we need to standarize the values of the input
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
N, D = X_train.shape

# We initilalize the Neural network

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(D,)))
model.add(tf.keras.layers.Dense(5, activation="elu"))
model.add(tf.keras.layers.Dense(100, activation="elu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# We define an optimizer with a learning rate schededuler
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
# learning rate schededuler (variable lr)
def schedule(epoch, lr):
    if epoch >= 200:
        print(lr)
        return lr - lr / 100
    return 0.01


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# entreanamos el modelo
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, callbacks=[scheduler, tensorboard_callback])

# imprimimos las perdidas para ver el avance del aprendizaje

print(model.evaluate(X_test, y_test))
print(f"Train score: {model.evaluate(X_train, y_train)}")
print(f"Test score: {model.evaluate(X_test, y_test)}")

fig, axs = plt.subplots(2, 2)
axs[0, 0].set_title("loss")
axs[0, 0].plot(r.history["loss"], label="loss")
axs[0, 1].set_title("val loss")
axs[0, 1].plot(r.history["val_loss"], label="val_loss")
axs[1, 0].set_title("accuracy")
axs[1, 0].plot(r.history["accuracy"], label="acc")
axs[1, 1].set_title("val accuracy")
axs[1, 1].plot(r.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.tight_layout()
plt.show()

print(model.layers[0].get_weights())
