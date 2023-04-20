import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("loss.csv")

y = data.iloc[:,2].to_list()

smooth_y = []
for i in range(5, len(y)):
    smooth_y.append((y[i-5] + y[i-4] + y[i-3] + y[i-2] + y[i-1] + y[i])/6)

x = np.linspace(0, 100, len(smooth_y))

plt.plot(x, smooth_y)
plt.show()