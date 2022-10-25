import numpy as np
import pandas as pd

a = np.random.random(100).reshape(-1,1)
asq = a * a
data = np.concatenate((a, asq), 1)
df = pd.DataFrame(data, columns=["x", "y"])
df.to_csv("out.csv", index=False)
print(data)