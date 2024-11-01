import os
import pandas as pd


file_0 = "data/ti_training/0/X_test_original.csv"
file_1 = "data/ti_training/1/X_test_original.csv"
# path_0=os.path.join(os.path.dirname(os.getcwd()), file_0)
# path_1=os.path.join(os.path.dirname(os.getcwd()), file_1)

combined_df = pd.concat([pd.read_csv(file_0), pd.read_csv(file_1)], ignore_index=True, join='outer')
combined_df.fillna(0, inplace=True)
combined_df.to_csv("data/ti_training/X_test.csv", index=False)

df_0 = combined_df.sample(frac=0.5, random_state=42)
df_1 = combined_df.drop(df_0.index)

df_0 = df_0.to_csv("data/ti_training/0/X_test.csv", index=False)
df_1 = df_1.to_csv("data/ti_training/1/X_test.csv", index=False)