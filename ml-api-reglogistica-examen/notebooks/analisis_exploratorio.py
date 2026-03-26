import pandas as pd
import matplotlib.pyplot as plt
from src.data.load_data import load_data
from src.data.preprocess import preprocess

df = load_data()
print(df.head())

X, y = preprocess(df)

plt.scatter(X, y)
plt.xlabel("Horas")
plt.ylabel("Aprobado (1=Sí, 0=No)")
plt.show()