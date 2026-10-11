import random
import numpy as np
from typing import List, Dict, Any

class AdaptiveOperatorSelector:
    """
    自适应算子选择器（多臂老虎机 Multi-Armed Bandit）
    用于在进化过程中动态调整变异/交叉算子的被选中概率。
    基于算子生成的子代进入 Pareto 前沿或存活到下一代的质量来给予奖励。
    """
    def __init__(self, operators: List[str], alpha: float = 0.1, min_prob: float = 0.05):
        """
        Args:
            operators: 支持的算子名称列表
            alpha: 学习率，决定历史奖励权重的衰减速度
            min_prob: 每个算子的最小被选中概率，确保探索性 (Exploration)
        """
        self.operators = operators
        self.alpha = alpha
        self.min_prob = min_prob
        
        # 初始概率均匀分布
        self.probabilities = {op: 1.0 / len(operators) for op in operators}
        self.scores = {op: 0.0 for op in operators}
        self.uses = {op: 0 for op in operators}
        
    def select_operator(self) -> str:
        """
        根据当前概率分布选择一个算子
        """
        ops = list(self.probabilities.keys())
        probs = list(self.probabilities.values())
        return np.random.choice(ops, p=probs)
        
    def update(self, operator: str, reward: float):
        """
        使用得到的奖励更新算子的分数和概率
        
        Args:
            operator: 使用的算子名称
            reward: 获得的奖励值
        """
        if operator not in self.operators:
            return
            
        self.uses[operator] += 1
        # 指数加权移动平均更新分数
        self.scores[operator] = (1 - self.alpha) * self.scores[operator] + self.alpha * reward
        self._recalculate_probabilities()
        
    def _recalculate_probabilities(self):
        """
        重新计算各个算子的选择概率
        使用轮盘赌策略，并保证每个算子有最低探索概率
        """
        total_score = sum(self.scores.values())
        if total_score <= 1e-6:
            # 如果分数都很低，恢复均匀分布
            self.probabilities = {op: 1.0 / len(self.operators) for op in self.operators}
            return
            
        num_ops = len(self.operators)
        # 剩余可分配概率空间
        remaining_prob = 1.0 - num_ops * self.min_prob
        
        for op in self.operators:
            self.probabilities[op] = self.min_prob + remaining_prob * (self.scores[op] / total_score)
            
    def get_stats(self) -> Dict[str, Any]:
        """返回当前算子的统计信息"""
        return {
            "probabilities": self.probabilities.copy(),
            "scores": self.scores.copy(),
            "uses": self.uses.copy()
        }
