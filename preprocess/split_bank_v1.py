import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your CSV file
path = '../data/bank-full.csv'

# Read the CSV file with semicolon delimiter
df = pd.read_csv(path, sep=';')

# Display the shape and columns of the dataframe
print(df.shape)
print(df.columns)

# Stratified split: Split the data into train and test sets, ensuring the distribution of 'y' is maintained
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['y'], random_state=42)

# Display the shapes of the train and test sets
print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

# Optionally, check the distribution of 'y' in both train and test sets
print("Training set 'y' distribution:\n", train_df['y'].value_counts())
print("Testing set 'y' distribution:\n", test_df['y'].value_counts())

train_df.to_csv('../data/bank_train_v1.csv', index=False)
test_df.to_csv('../data/bank_test_v1.csv', index=False)
