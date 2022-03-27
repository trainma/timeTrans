import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('transformer_pred2.csv')
print(df['truth'],df['predict'])
#%%
plt.figure(figsize=(20,16))
plt.plot(df['predict'])
plt.plot(df['truth'])
