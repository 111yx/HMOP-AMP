import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import argparse
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# Parse command-line arguments only when running directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP Classification")
    parser.add_argument("--testPath", type=str, default="final_main_population.csv", help="Path to the test dataset")
    parser.add_argument("--savePath", type=str, default="classification_results.csv", help="Path to save the results")
    args = parser.parse_args()
    
    testPath = args.testPath
    savePath = args.savePath
else:
    # Default values for when imported as module
    testPath = "final_main_population.csv"
    savePath = "classification_results.csv"

# Data paths
current_dir = os.path.dirname(os.path.abspath(__file__))
trainPath = os.path.join(current_dir, "data", "classification", "train.csv")
validatePath = os.path.join(current_dir, "data", "classification", "test.csv")

# Configuration parameters
batch_size = 256
embedding_size = 20
num_tokens = 100
num_classes = 2
num_heads = 4

# Model paths
model_list = {
    "CNN": os.path.join(current_dir, "model", "classification", "CNN.pth"),
    "Transformer": os.path.join(current_dir, "model", "classification", "Transformer.pth"),
    "Attention": os.path.join(current_dir, "model", "classification", "Attention.pth"),
    "LSTM": os.path.join(current_dir, "model", "classification", "LSTM.pth"),
}
nameList = model_list.keys()

# Sequence to numerical mapping
mydict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12,
          "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}

softmax = nn.functional.softmax

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dataProcessPipeline(seq):
    """
    Process a sequence into a padded one-hot encoded tensor and a mask.

    Parameters:
    seq (str): The input sequence to process.

    Returns:
    tuple: A tuple containing the padded one-hot encoded tensor and the mask tensor.
    """
    # 确保序列是字符串
    if not isinstance(seq, str):
        seq = str(seq)

    seq = seq.upper().strip()

    # 过滤非标准氨基酸字符
    valid_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    filtered_seq = ''.join([c for c in seq if c in valid_amino_acids])

    # 如果序列为空，使用默认值
    if len(filtered_seq) == 0:
        filtered_seq = 'A'

    # 转换氨基酸序列为数字索引
    num_seq = []
    for character in filtered_seq:
        if character.upper() in mydict:
            num_seq.append(mydict[character.upper()])
        else:
            num_seq.append(mydict['A'])  # 默认值

    seq_array = np.array(num_seq, dtype=int)
    seq_len = seq_array.shape[0]

    # 关键：确保使用 torch.long 类型
    torch_seq = torch.tensor(seq_array, dtype=torch.long)

    # 进行one-hot编码
    onehotSeq = torch.nn.functional.one_hot(torch_seq, num_classes=20)
    onehotSeq = onehotSeq.float()  # 转换为float

    # Pad the sequence to a length of 100
    pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100 - seq_len))
    mask = np.zeros(100, dtype=int)
    if seq_len < 100:
        mask[seq_len:] = 1
    mask = torch.tensor(mask, dtype=torch.bool)  # 使用bool类型

    pad_seq = pad(onehotSeq)

    return pad_seq, mask


# 自定义测试数据集
class CSVTestDataset(Dataset):
    def __init__(self, csv_path):
        """
        从CSV文件读取测试数据

        Parameters:
        csv_path (str): CSV文件路径
        """
        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 找到序列列（不区分大小写）
        seq_column = None
        for col in df.columns:
            if 'sequence' in col.lower():
                seq_column = col
                break

        if seq_column is None:
            # 如果没有找到Sequence列，尝试第一列
            print(f"警告：未找到'Sequence'列，使用第一列 '{df.columns[0]}' 作为序列")
            seq_column = df.columns[0]

        # 清理序列
        def clean_seq(s):
            if pd.isna(s):
                return ""
            s = str(s).strip().upper()
            # 只保留标准氨基酸字符
            valid_chars = 'ACDEFGHIKLMNPQRSTVWY'
            return ''.join([c for c in s if c in valid_chars])

        self.seqs = list(df[seq_column].apply(clean_seq))

        # 确保没有空序列
        self.seqs = [seq if seq else "A" for seq in self.seqs]

        # 保存原始序列用于输出
        self.original_seqs = list(df[seq_column])

        print(f"成功加载 {len(self.seqs)} 个序列")

    def __getitem__(self, index):
        seq = self.seqs[index]
        original_seq = self.original_seqs[index]
        num_seq, mask = dataProcessPipeline(seq)
        return num_seq, mask, original_seq

    def __len__(self):
        return len(self.seqs)


# 训练数据集（保持不变）
class TrainDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path, header=0)
        df = df[df["Length"] <= 100]

        # 清理序列
        def clean_seq(s):
            if pd.isna(s):
                return ""
            s = str(s).strip().upper()
            return ''.join([c for c in s if c in 'ACDEFGHIKLMNPQRSTVWY'])

        self.seqs = list(df["Sequence"].apply(clean_seq))
        self.seqs = [seq if seq else "A" for seq in self.seqs]
        self.labels = list(df["label"])

    def __getitem__(self, index):
        seq = self.seqs[index]
        num_seq, mask = dataProcessPipeline(seq)
        label = self.labels[index]
        return num_seq, mask, label

    def __len__(self):
        return len(self.seqs)


class PositionalEncoding(nn.Module):
    def __init__(self, length, d_model=20):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe
        return x


class AttentionNetwork(nn.Module):
    def __init__(self, batch_size=128, embedding_size=20, num_tokens=100, num_classes=2, num_heads=4):
        super(AttentionNetwork, self).__init__()
        self.pe = PositionalEncoding(length=num_tokens, d_model=embedding_size)
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.hidden1 = 20
        self.hidden2 = 60
        self.hidden3 = 20
        self.dropout = 0.5
        self.relu = nn.ReLU()
        self.LN = nn.LayerNorm(normalized_shape=self.hidden1)
        self.fc1 = nn.Linear(self.embedding_size, self.hidden1)
        self.multihead_att = nn.MultiheadAttention(embed_dim=self.hidden1, num_heads=self.num_heads, batch_first=1,
                                                   dropout=self.dropout)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(self.hidden1 * self.num_tokens, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.fc4 = nn.Linear(self.hidden3, self.num_classes)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask):
        x = self.pe(x)
        x = self.fc1(x)
        mask = mask.to(torch.bool)
        x, _ = self.multihead_att.forward(x, x, x, key_padding_mask=mask)
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def main():
    print(f"正在处理文件: {testPath}")

    # 加载数据
    trainData = TrainDataset(data_path=trainPath)
    validateData = TrainDataset(data_path=validatePath)
    testData = CSVTestDataset(csv_path=testPath)

    # 创建数据加载器（禁用多进程以避免Windows问题）
    train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=testData, batch_size=batch_size, shuffle=False, num_workers=0)
    validate_loader = DataLoader(dataset=validateData, batch_size=batch_size, shuffle=False, num_workers=0)

    result_df = pd.DataFrame({
        'Sequence': testData.original_seqs,
        'Length': [len(seq) for seq in testData.seqs]
    })

    model_out = {}

    # 使用所有模型处理数据
    for modelName in nameList:
        modelPath = model_list[modelName]
        model_out[modelName] = []

        print(f"\n正在加载模型: {modelName}")
        t_model = torch.load(modelPath, map_location=device)
        t_model = t_model.to(device)
        t_model.eval()

        # 在验证集上评估模型
        def evaluate_model(test_loader):
            t_model.eval()
            epi = 0.000001
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            total = 0
            count = 0

            for data in test_loader:
                inputs, masks, labels = data
                inputs = inputs.float().to(device)
                masks = masks.float().to(device)

                if modelName != "Attention":
                    out = t_model(inputs)
                else:
                    out = t_model(inputs, masks)

                out = torch.squeeze(out)
                out = torch.argmax(out, -1)
                out = out.cpu()

                for i, pre in enumerate(out):
                    total += 1
                    if pre == labels[i]:
                        count += 1
                        if pre == 0:
                            tn += 1
                        else:
                            tp += 1
                    else:
                        if pre == 0:
                            fn += 1
                        else:
                            fp += 1

            print(f"\n{modelName} 在验证集上的表现:")
            print("精确度 (Precision):", np.round(tp / (tp + fp + epi), 3))
            print("召回率 (Recall):", np.round(tp / (tp + fn + epi), 3))
            print("特异性 (Specificity):", np.round(tn / (tn + fp + epi), 3))
            print("F1分数:", np.round(2 * tp / (2 * tp + fp + fn + epi), 3))
            print("准确率:", np.round(count / total, 3))

        # 评估模型
        print(f"\n评估 {modelName} 模型...")
        evaluate_model(validate_loader)

        # 使用模型预测测试数据
        print(f"使用 {modelName} 预测测试数据...")
        all_predictions = []

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, masks, seqs = data
                inputs = inputs.float().to(device)
                masks = masks.float().to(device)

                if modelName != "Attention":
                    out = t_model(inputs)
                else:
                    out = t_model(inputs, masks)

                out = out.cpu()
                if "LSTM" in modelName:
                    out = out.unsqueeze(0)

                out_ori = torch.squeeze(out)
                out_soft = softmax(out_ori, -1)
                out_soft_AMP = out_soft[:, 1]

                out_soft_numpy = list(out_soft_AMP.detach().numpy())
                out_soft_numpy = [round(v, 3) for v in out_soft_numpy]
                all_predictions.extend(out_soft_numpy)

                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f"  已处理批次: {i + 1}/{len(test_loader)}")

        model_out[modelName] = all_predictions

    # 汇总结果
    for k, v in model_out.items():
        result_df[k] = v

    # 使用Ensemble方法：如果所有模型的预测都大于0.5，则认为是AMP
    y = (result_df["CNN"] > 0.5) & (result_df["Transformer"] > 0.5) & (result_df["LSTM"] > 0.5) & (
                result_df["Attention"] > 0.5)
    result_df["Ensemble_Prediction"] = y.astype(int)

    # 计算平均概率
    result_df["Average_Probability"] = result_df[["CNN", "Transformer", "Attention", "LSTM"]].mean(axis=1).round(3)

    # 重新排列列顺序
    result_df = result_df[["Sequence", "Length", "Ensemble_Prediction", "Average_Probability",
                           "CNN", "Transformer", "Attention", "LSTM"]]

    # 保存结果
    result_df.to_csv(savePath, index=False)

    print(f"\n{'=' * 60}")
    print("分类结果统计:")
    print(f"{'=' * 60}")
    print(f"总序列数: {len(result_df)}")
    print(f"预测为AMP的序列数: {result_df['Ensemble_Prediction'].sum()}")
    print(f"AMP比例: {result_df['Ensemble_Prediction'].sum() / len(result_df) * 100:.2f}%")
    print(f"\n结果已保存到: {savePath}")
    print(f"{'=' * 60}")

    # 显示前10个结果
    print("\n前10个序列的预测结果:")
    print(result_df.head(10).to_string(index=False))


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()
    main()