import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import random
import numpy as np
import os

class ESMMutationOperator:
    """
    使用 ESM-2 预训练模型引导的智能变异算子
    """
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls, model_name=None):
        if cls._instance is None:
            # 默认使用本地路径
            if model_name is None:
                project_root = os.path.dirname(os.path.abspath(__file__))
                model_name = os.path.join(project_root, "model", "esm2_t6_8M_UR50D")
            
            cls._instance = super(ESMMutationOperator, cls).__new__(cls)
            cls._initialize(model_name)
        return cls._instance

    @classmethod
    def _initialize(cls, model_path):
        print(f"正在从本地加载 ESM-2 模型: {model_path}...")
        # 自动选择设备
        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {cls._device}")
        
        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            cls._model = EsmForMaskedLM.from_pretrained(model_path, local_files_only=True)
            cls._model.to(cls._device)
            cls._model.eval()
            print("ESM-2 本地模型加载完成。")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("请确保已将模型文件下载并放置在正确目录中。")
            raise e
        
        # 氨基酸字母表（ESM-2 标准）
        cls.AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    def suggest_mutation(self, sequence, pos):
        """
        针对序列的特定位置，使用 ESM-2 建议下一个可能的氨基酸
        """
        # 1. 准备带掩码的序列
        seq_list = list(sequence)
        original_aa = seq_list[pos]
        seq_list[pos] = "<mask>"
        masked_seq = "".join(seq_list)

        # 2. 编码输入
        inputs = self._tokenizer(masked_seq, return_tensors="pt").to(self._device)
        
        # 3. 模型推理
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # 4. 获取掩码位置的预测结果
        # 注意：ESM tokenizer 会添加 <cls> 和 <sep> 标记，所以位置需要对应
        # 简单处理：找到 <mask> 所在的 token 索引
        tokens = self._tokenizer.tokenize(masked_seq)
        mask_token_index = tokens.index("<mask>") + 1 # +1 是因为起始有 <cls>

        mask_token_logits = logits[0, mask_token_index]
        
        # 5. 提取 20 种标准氨基酸的概率
        aa_logits = []
        for aa in self.AMINO_ACIDS:
            aa_token_id = self._tokenizer.convert_tokens_to_ids(aa)
            aa_logits.append(mask_token_logits[aa_token_id].item())
        
        # 6. Softmax 转换为概率分布
        probs = self._softmax(np.array(aa_logits))
        
        # 7. 根据概率分布进行采样
        # 为了保持一定的多样性，使用加权随机选择，而不是直接选最大的
        suggested_aa = np.random.choice(self.AMINO_ACIDS, p=probs)
        
        return suggested_aa

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

# 单例模式导出，避免重复加载模型
esm_mutation_operator = None

def get_esm_mutation_operator():
    global esm_mutation_operator
    if esm_mutation_operator is None:
        esm_mutation_operator = ESMMutationOperator()
    return esm_mutation_operator
