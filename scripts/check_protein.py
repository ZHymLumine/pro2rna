import pandas as pd

def check_protein_sequences(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 检查 'True Protein Sequence' 是否所有值以 'M' 开头
    all_start_with_M = df['True Protein Sequence'].apply(lambda seq: str(seq).startswith('M')).all()

    # 输出结果
    if all_start_with_M:
        print("All True Protein Sequences start with 'M'.")
    else:
        print("Not all True Protein Sequences start with 'M'.")

    # 返回不符合条件的行
    invalid_rows = df[~df['True Protein Sequence'].apply(lambda seq: str(seq).startswith('M'))]
    if not invalid_rows.empty:
        print("The following rows do not start with 'M':")
        print(invalid_rows)

    # 检查 True Protein Sequence 和 Predicted Protein Sequence 的长度是否一致
    df['True Length'] = df['True Protein Sequence'].apply(len)
    df['Predicted Length'] = df['Predicted Protein Sequence'].apply(len)

    # 找出长度不匹配的行
    mismatched_rows = df[df['True Length'] != df['Predicted Length']]

    # 输出结果
    if mismatched_rows.empty:
        print("All True and Predicted Protein Sequences have matching lengths.")
    else:
        print("The following rows have mismatched lengths:")
        print(mismatched_rows[['True Protein Sequence', 'Predicted Protein Sequence', 'True Length', 'Predicted Length']])

# 使用示例
file_path = "mismatch_results_label.csv"  # 替换为你的 CSV 文件路径
check_protein_sequences(file_path)
