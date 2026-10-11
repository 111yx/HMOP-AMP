import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
import warnings
import argparse
import os
from typing import List, Tuple

# Import AttentionNetwork classes and make them available under their original names
# This is needed for PyTorch to correctly load the saved models
try:
    from AMP_classification import AttentionNetwork as ClassificationAttentionNetwork
    from AMP_classification import PositionalEncoding
    from AMP_regression import AttentionNetwork as RegressionAttentionNetwork
    from AMP_regression import PositionalEncoding as RegressionPositionalEncoding
except ImportError:
    # If direct import fails, try adding the current directory to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from AMP_classification import AttentionNetwork as ClassificationAttentionNetwork
    from AMP_classification import PositionalEncoding
    from AMP_regression import AttentionNetwork as RegressionAttentionNetwork
    from AMP_regression import PositionalEncoding as RegressionPositionalEncoding

# Make the AttentionNetwork class available under its original name for PyTorch model loading
# This is needed because the saved models reference this exact class name
# We'll dynamically use the correct class based on the model type
AttentionNetwork = ClassificationAttentionNetwork

# 忽略警告
warnings.filterwarnings('ignore')

class AMPPredictor:
    """抗菌肽预测器，整合分类和回归模型"""
    
    def __init__(self):
        """初始化预测器"""
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 分类模型配置
        self.classification_batch_size = 256
        self.classification_model_list = {
            "CNN": os.path.join(current_dir, "model", "classification", "CNN.pth"),
            "Transformer": os.path.join(current_dir, "model", "classification", "Transformer.pth"),
            "Attention": os.path.join(current_dir, "model", "classification", "Attention.pth"),
            "LSTM": os.path.join(current_dir, "model", "classification", "LSTM.pth"),
        }
        
        # 回归模型配置
        self.regression_batch_size = 256
        self.regression_model_list = {
            "CNN": os.path.join(current_dir, "model", "regression", "CNN.pth"),
            "Transformer": os.path.join(current_dir, "model", "regression", "Transformer.pth"),
            "Attention": os.path.join(current_dir, "model", "regression", "Attention.pth"),
            "LSTM": os.path.join(current_dir, "model", "regression", "LSTM.pth"),
        }
        self.regression_weight = {"CNN": 0.25000594, "Transformer": 0.2500046, "Attention": 0.25000825, "LSTM": 0.24998219}
        
        # 氨基酸映射
        self.mydict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12,
                       "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}
        
        self.softmax = nn.functional.softmax
        
        # 设备选择
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.classification_models = self._load_classification_models()
        self.regression_models = self._load_regression_models()
    
    def _load_classification_models(self):
        """加载分类模型"""
        models = {}
        
        for modelName, modelPath in self.classification_model_list.items():
            try:
                # For classification models, we need to ensure ClassificationAttentionNetwork is available
                # Temporarily make it available in the global namespace for model loading
                import sys
                if 'Attention' in modelName:
                    sys.modules['__main__'].AttentionNetwork = ClassificationAttentionNetwork
                    sys.modules['__main__'].PositionalEncoding = PositionalEncoding
                
                # Import necessary classes for other models
                try:
                    import otherModels5_2Output
                    sys.modules['otherModels5_2Output'] = otherModels5_2Output
                except ImportError:
                    pass
                    
                # Fix for PyTorch 2.6+ weights_only restriction
                try:
                    # Try with weights_only parameter (PyTorch 2.6+)
                    model = torch.load(modelPath, map_location=self.device, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions
                    model = torch.load(modelPath, map_location=self.device)
                model = model.to(self.device)
                model.eval()
                models[modelName] = model
                print(f"成功加载分类模型: {modelName}")
            except Exception as e:
                print(f"加载分类模型 {modelName} 失败: {e}")
        return models
    
    def _load_regression_models(self):
        """加载回归模型"""
        models = {}
        
        for modelName, modelPath in self.regression_model_list.items():
            try:
                # For regression models, we need to ensure RegressionAttentionNetwork is available
                # Temporarily make it available in the global namespace for model loading
                import sys
                if 'Attention' in modelName:
                    sys.modules['__main__'].AttentionNetwork = RegressionAttentionNetwork
                    sys.modules['__main__'].PositionalEncoding = RegressionPositionalEncoding
                
                # Import necessary classes for other models
                try:
                    import otherModelsRegression
                    sys.modules['otherModelsRegression'] = otherModelsRegression
                except ImportError:
                    pass
                    
                # Fix for PyTorch 2.6+ weights_only restriction
                try:
                    # Try with weights_only parameter (PyTorch 2.6+)
                    model = torch.load(modelPath, map_location=self.device, weights_only=False)
                except TypeError:
                    # Fallback for older PyTorch versions
                    model = torch.load(modelPath, map_location=self.device)
                model = model.to(self.device)
                model.eval()
                models[modelName] = model
                print(f"成功加载回归模型: {modelName}")
            except Exception as e:
                print(f"加载回归模型 {modelName} 失败: {e}")
        return models
    
    def _data_process_pipeline(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理序列数据，转换为模型输入格式
        
        Parameters:
        seq (str): 输入序列
        
        Returns:
        tuple: 包含填充后的one-hot编码张量和掩码张量的元组
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
            if character.upper() in self.mydict:
                num_seq.append(self.mydict[character.upper()])
            else:
                num_seq.append(self.mydict['A'])  # 默认值
        
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
    
    def _predict_amp_classification(self, sequences: List[str]) -> List[bool]:
        """
        使用分类模型预测序列是否为AMP
        
        Parameters:
        sequences (List[str]): 肽序列列表
        
        Returns:
        List[bool]: 是否为AMP的预测结果列表
        """
        if not sequences:
            return []
        
        # 准备数据
        processed_data = []
        for seq in sequences:
            pad_seq, mask = self._data_process_pipeline(seq)
            processed_data.append((pad_seq, mask, seq))
        
        # 存储各模型的预测结果
        model_predictions = {}
        
        # 使用所有分类模型进行预测
        for modelName, model in self.classification_models.items():
            predictions = []
            
            with torch.no_grad():
                for pad_seq, mask, seq in processed_data:
                    inputs = pad_seq.unsqueeze(0).float().to(self.device)
                    masks = mask.unsqueeze(0).float().to(self.device)
                    
                    if modelName != "Attention":
                        out = model(inputs)
                    else:
                        out = model(inputs, masks)
                    
                    out = out.cpu()
                    if "LSTM" in modelName:
                        out = out.unsqueeze(0)
                    
                    out_ori = torch.squeeze(out)
                    out_soft = self.softmax(out_ori, -1)
                    
                    # 处理不同模型的输出维度
                    if out_soft.dim() == 1:
                        # 1D tensor: [class1_prob, class2_prob]
                        amp_prob = out_soft[1]  # AMP类的概率是索引1
                    else:
                        # 2D tensor: [[class1_prob, class2_prob]]
                        amp_prob = out_soft[:, 1]
                    
                    prob = amp_prob.item()
                    predictions.append(prob)
            
            model_predictions[modelName] = predictions
        
        # 集成预测结果：如果所有模型的预测概率都大于0.5，则认为是AMP
        is_amp_list = []
        for i in range(len(sequences)):
            all_high_prob = True
            for modelName in self.classification_models.keys():
                if model_predictions[modelName][i] <= 0.5:
                    all_high_prob = False
                    break
            is_amp_list.append(all_high_prob)
        
        return is_amp_list
    
    def _predict_mic_regression(self, sequences: List[str]) -> List[float]:
        """
        使用回归模型预测MIC值
        
        Parameters:
        sequences (List[str]): 肽序列列表
        
        Returns:
        List[float]: MIC值预测结果列表
        """
        if not sequences:
            return []
        
        # 准备数据
        processed_data = []
        for seq in sequences:
            pad_seq, mask = self._data_process_pipeline(seq)
            processed_data.append((pad_seq, mask, seq))
        
        # 存储各模型的预测结果
        model_predictions = {}
        
        # 使用所有回归模型进行预测
        for modelName, model in self.regression_models.items():
            predictions = []
            
            with torch.no_grad():
                for pad_seq, mask, seq in processed_data:
                    inputs = pad_seq.unsqueeze(0).float().to(self.device)
                    masks = mask.unsqueeze(0).float().to(self.device)
                    
                    if modelName != "Attention":
                        out = model(inputs)
                    else:
                        out = model(inputs, masks)
                    
                    out = out.cpu()
                    out_ori = torch.squeeze(out)
                    
                    # 处理不同模型的输出维度
                    if out_ori.dim() == 1:
                        # 1D tensor with one element: [value]
                        mic_scalar = out_ori[0]
                    else:
                        # 0D scalar tensor: value
                        mic_scalar = out_ori
                    
                    mic_value = mic_scalar.item()
                    predictions.append(mic_value)
            
            model_predictions[modelName] = predictions
        
        # 集成预测结果：加权平均
        ensemble_predictions = []
        for i in range(len(sequences)):
            weighted_sum = 0.0
            for modelName, weight in self.regression_weight.items():
                weighted_sum += model_predictions[modelName][i] * weight
            ensemble_predictions.append(weighted_sum)
        
        return ensemble_predictions
    
    def predict(self, sequences: List[str]) -> List[Tuple[bool, float]]:
        """
        预测序列是否为AMP以及对应的MIC值
        
        Parameters:
        sequences (List[str]): 肽序列列表
        
        Returns:
        List[Tuple[bool, float]]: 每个序列的(是否为AMP, MIC值)元组列表
        """
        if not sequences:
            return []
        
        # 第一步：判断是否为AMP
        is_amp_list = self._predict_amp_classification(sequences)
        
        # 第二步：对于是AMP的序列，预测MIC值；对于不是AMP的序列，MIC值设为3.9133
        mic_values = [3.9133] * len(sequences)  # 默认值
        
        # 收集是AMP的序列索引
        amp_sequences = []
        amp_indices = []
        for i, is_amp in enumerate(is_amp_list):
            if is_amp:
                amp_sequences.append(sequences[i])
                amp_indices.append(i)
        
        # 对于是AMP的序列，预测MIC值
        if amp_sequences:
            amp_mic_values = self._predict_mic_regression(amp_sequences)
            for idx, mic_value in zip(amp_indices, amp_mic_values):
                mic_values[idx] = mic_value
        
        # 返回结果
        results = []
        for i in range(len(sequences)):
            results.append((is_amp_list[i], mic_values[i]))
        
        return results


def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="抗菌肽预测器")
    parser.add_argument("--input", type=str, required=True, help="输入CSV文件路径，必须包含Sequence列")
    parser.add_argument("--output", type=str, required=True, help="输出CSV文件路径")
    args = parser.parse_args()
    
    # 读取输入文件
    df = pd.read_csv(args.input)
    
    # 检查是否存在Sequence列
    if 'Sequence' not in df.columns:
        raise ValueError("输入文件必须包含'Sequence'列")
    
    sequences = df['Sequence'].tolist()
    
    # 创建预测器并进行预测
    predictor = AMPPredictor()
    results = predictor.predict(sequences)
    
    # 将结果添加到DataFrame
    is_amp_list = [result[0] for result in results]
    mic_list = [result[1] for result in results]
    
    df['Is_AMP'] = is_amp_list
    df['MIC_Value'] = mic_list
    
    # 保存结果
    df.to_csv(args.output, index=False)
    print(f"预测结果已保存到: {args.output}")


if __name__ == "__main__":
    main()