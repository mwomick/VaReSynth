import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("loss.csv")

x = data.iloc[:,1].to_list()
y = data.iloc[:,2].to_list()

plt.plot(x, y)
plt.show()