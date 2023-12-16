import pandas as pd


df = pd.read_csv('AudioDec_Denoise/output.csv')
print()
print("AudioDec Denoise")
print(df.mean(numeric_only=True))
df = pd.read_csv('Ours/output.csv')
print()
print("Ours")
print(df.mean(numeric_only=True))



