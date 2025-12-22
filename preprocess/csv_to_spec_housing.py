import pandas as pd
import argparse


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
            f.write(f'{col} continuous\n')
        f.write(f'LABEL_POS {args.label_pos}\n')