import pandas as pd

# HOW TO READ FILES
# df = pd.read_csv("sales_data_sample.csv",encoding="latin1")
# df = pd.read_excel("SampleSuperstore.xlsx",engine="xlrd")
# df = pd.read_json("sample_Data.json")
# print(df)

# HOW TO SAVE DATA

# data = {
#     "Name" : ["Esther","Riya","Hema"],
#     "Age" : [30,20,10],
#     "City": ["Delhi","Pune","Noida"]
# }

# df = pd.DataFrame(data)
# print(df)

# df.to_csv("output.csv",index=False)
# df.to_excel("output.xlsx",index=False)
# df.to_json("output.json",index=False)

# Printing rows

df = pd.read_csv("sales_data_sample.csv",encoding="latin1")
print(df.head())
print(df.tail())

# Understanding

print(df.info())

