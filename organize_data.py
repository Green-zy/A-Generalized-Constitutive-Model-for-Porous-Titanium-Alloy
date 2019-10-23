import os
os.chdir('G:\\data') # change the current working directory
data_origin = pd.read_csv('data.csv')
print(data_origin.tail())

data_completion = pd.concat([data_supplement,data_origin], ignore_index=True)
print(data_completion.tail())