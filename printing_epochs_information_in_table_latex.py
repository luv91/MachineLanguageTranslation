import pandas as pd
import re

selected_files=['512_256_4_2_0.1.csv',
'512_128_4_2_0.1.csv',
'512_256_4_4_0.1.csv',
'512_512_4_2_0.1.csv',
'512_64_4_4_0.csv',
'512_128_4_4_0.csv',
'512_128_4_4_0.1new2.csv',
'512_128_4_4_0.1.csv',
'512_128_4_4_0.1new.csv']

data = {
    'model size': [],
    'number of heads': [],
    'number of layers': [],
    'dropout': [],
    # 'epochs': [],
    'time': [],
    'Train Loss': [],
    # 'Train Accuracy': [],
    # 'Train Perplexity': [],
    'Validation Loss': [],
    # 'Validation Accuracy': [],
    'Validation Perplexity': [],
    'Epoch Duration': []
}
def human_readable_format(num):
    if num >= 1e6:
        return "{:.1f}M".format(num / 1e6)
    elif num >= 1e3:
        return "{:.1f}K".format(num / 1e3)
    else:
        return str(num)
    
def custom_format(num):
    if isinstance(num, float):
        if num >= 1e3:
            return "{:.1e}".format(num)
        elif num != int(num):
            return "{:.4f}".format(num).rstrip('0').rstrip('.')
        else:
            return str(num)
    return num

for file in selected_files:
    parts = file.split("_")

    model_size = int(parts[2])
    num_heads = int(parts[3])
    num_layers = int(parts[4])
    dropout = float(re.findall("\d+\.\d+", parts[5])[0])  # Extracts float number from the string

    # Read the CSV file
    csv_data = pd.read_csv(file)
    
    # Accumulate the time taken for the first 100 steps (or as many rows as are available)
    time_taken = csv_data['Epoch Duration'][:100].sum() // 60

    # Assuming you want the data from the 100th epoch (row 99 in zero-based index)
    epoch_100 = csv_data.iloc[-1] if csv_data.shape[0] > 99 else None

    data['model size'].append(model_size)
    data['number of heads'].append(num_heads)
    data['number of layers'].append(num_layers)
    data['dropout'].append(dropout)
    # data['epochs'].append(100)
    data['time'].append(time_taken)
    data['Train Loss'].append(epoch_100['Train Loss'] if epoch_100 is not None else None)
    # data['Train Accuracy'].append(epoch_100['Train Accuracy'] if epoch_100 is not None else None)
    # data['Train Perplexity'].append(epoch_100['Train Perplexity'] if epoch_100 is not None else None)
    data['Validation Loss'].append(epoch_100['Validation Loss'] if epoch_100 is not None else None)
    # data['Validation Accuracy'].append(epoch_100['Validation Accuracy'] if epoch_100 is not None else None)
    data['Validation Perplexity'].append(epoch_100['Validation Perplexity'] if epoch_100 is not None else None)
    data['Epoch Duration'].append(epoch_100['Epoch Duration'] if epoch_100 is not None else None)

df = pd.DataFrame(data)
print(df)

df = df.dropna(axis=0)


# Apply the human_readable_format function to relevant columns
# columns_to_format = ['Train Loss', 'Train Accuracy', 'Train Perplexity', 'Validation Loss', 'Validation Accuracy', 'Validation Perplexity', 'Epoch Duration']
# for col in columns_to_format:
#     df[col] = df[col].apply(human_readable_format)
# Apply custom_format function to all numeric columns
# numeric_columns = ['dropout', 'time', 'Train Loss', 'Train Accuracy', 'Train Perplexity', 'Validation Loss', 'Validation Accuracy', 'Validation Perplexity', 'Epoch Duration']
numeric_columns = ['dropout', 'time', 'Train Loss', 'Validation Loss', 'Validation Perplexity', 'Epoch Duration']


for col in numeric_columns:
    df[col] = df[col].apply(custom_format)


df.to_csv('epoch_time_taken_information_table.csv')


def bold_min_time(row):
    if row['time'] == min_time_by_model_size[row['model size']]:
        return ["\\textbf{" + str(x) + "}" for x in row]
    return row

# Find the minimum time for each unique model size
min_time_by_model_size = df.groupby('model size')['time'].min()

# Apply the bolding function to each row, using the DataFrame's `apply` method
# Apply the bold_min_time function to each row, using the DataFrame's apply method
df_bolded = df.apply(bold_min_time, axis=1)

# Convert the modified DataFrame to LaTeX with float formatting for 4 decimal places
latex_code = df_bolded.to_latex(index=False)
latex_code = df.to_latex(index=False)
# , float_format="{:0.4f}".format
print(latex_code)

# latex_code = df.to_latex(index=False)
# print(latex_code)