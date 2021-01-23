import numpy as np
import pandas as pd
import seaborn as sns

start = 1.0
end = 0.01
decay = 0.985

Y = []

y = start
while y > end:
    Y.append(y)
    y = y * decay
Y.append(end)

df = pd.DataFrame({
    "Episodes": list(range(0, len(Y))),
    "Epsilon": Y
})

print(len(Y))

sns_plot = sns.lineplot(data=df, x="Episodes", y="Epsilon")
fig = sns_plot.get_figure()
fig.savefig("epsilon-decay.png")
