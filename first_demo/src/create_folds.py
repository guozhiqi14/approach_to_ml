import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 假设训练数据存储在'../input/train.csv'，并且目标列名为'target'
if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv("../input/mnist_train.csv")

    # 创建一个新列'kfold'，并初始化为-1
    df['kfold'] = -1

    # 随机打乱数据
    df = df.sample(frac=1).reset_index(drop=True)

    # 获取目标列的值
    y = df['label'].values

    # 初始化StratifiedKFold
    kf = StratifiedKFold(n_splits=5)

    # 遍历折数，分配每个样本到一个折中
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_idx, 'kfold'] = fold

    # 保存新的CSV文件，包含'kfold'列
    df.to_csv("../input/mnist_train_folds.csv", index=False)