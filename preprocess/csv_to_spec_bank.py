import pandas as pd
import argparse

# ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 
# 'campaign', 'pdays', 'previous', 'poutcome', 'y']
# determine if each feature is 'continuous' or 'discrete'
# col2char = {
#     'age': 'continuous',
#     'job': 'discrete',
#     'marital': 'discrete',
#     'education': 'discrete',
#     'default': 'discrete',
#     'balance': 'continuous',
#     'housing': 'discrete',
#     'loan': 'discrete',
#     'contact': 'discrete',
#     'day': 'continuous',
#     'month': 'discrete',
#     'duration': 'continuous',
#     'campaign': 'continuous',
#     'pdays': 'continuous',
#     'previous': 'continuous',
#     'poutcome': 'discrete',
#     'y': 'discrete',
# }

discrete_cols = [
    'contact',
    'default',
    'education',
    'housing',
    'job',
    'loan',
    'marital',
    'month',
    'poutcome',
    'y'
]

continuous_cols = [
    'age',
    'balance',
    'campaign',
    'day',
    'duration',
    'pdays',
    'previous',
]

col2char = {}
for col in discrete_cols:
    col2char[col] = 'discrete'
for col in continuous_cols:
    col2char[col] = 'continuous'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--label_pos', type=int, default=-1)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cols = df.columns.tolist()
    print(f'columns: {cols}')

    data_path = args.input.replace('.csv', '.data')
    print(f'*.data: {data_path}')
    df.to_csv(data_path, index=False, header=False)

    print(f'label_pos: {args.label_pos}')

    info_path = args.input.replace('.csv', '.info')
    print(f'*.info: {info_path}')
    with open(info_path, 'w') as f:
        for col in cols:
            f.write(f'{col} {col2char[col]}\n')
        f.write(f'LABEL_POS {args.label_pos}\n')