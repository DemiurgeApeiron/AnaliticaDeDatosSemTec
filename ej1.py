import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print(df.head(5))
print(df.info(5))
print(df.shape)
df.boxplot(column=["age"])
df["age"].hist(bins=range(0, 90, 5), grid=False, color="#1c1c1c", ec="yellow")
plt.xlabel("Edades")
plt.ylabel("Pacientes")
plt.title("")

plt.show()
