import random
import string
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

# 尝试导入 ESM 变异算子
try:
    from esm_mutation_operator import get_esm_mutation_operator
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: esm_mutation_operator module not found. Using random mutation.")

class Peptide:
    """抗菌肽序列类"""
    
    # 氨基酸字母表
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    
    def __init__(self, sequence: Optional[str] = None, length_range: tuple = (5, 30), fixed_length: Optional[int] = None):
        """
        初始化肽序列
        
        Args:
            sequence: 肽序列字符串，如果为None则随机生成
            length_range: 肽序列长度范围 (min, max)
            fixed_length: 固定长度，如果指定则使用固定长度生成序列
        """
        self.length_range = length_range
        self.fixed_length = fixed_length
        if sequence is None:
            self.sequence = self._generate_random_peptide()
        else:
            self.sequence = sequence
            
        # 预测的目标值
        self.antimicrobial_activity: Optional[float] = None
        self.base_antimicrobial_activity: Optional[float] = None
        self.hemolytic_activity: Optional[float] = None
        # self.cell_activity: Optional[float] = None
        
        # 结构预测相关
        self.structure: Optional[str] = None
        self.helicity: float = 0.0
        self.helicity_epsilon: Optional[float] = None
        self.constraint_violation: float = 0.0
        self.is_feasible: bool = True
        
        # 支撑值（用于非支配排序等）
        self.crowding_distance: float = 0.0
        self.domination_count: int = 0
        self.dominated_solutions: List['Peptide'] = []
        
        # 小生境相关属性
        self.niche_count: int = 0  # 小生境计数
        self.niche_factor: float = 1.0  # 小生境因子
        
    def _generate_random_peptide(self) -> str:
        """生成随机肽序列"""
        if self.fixed_length is not None:
            length = self.fixed_length
        else:
            length = random.randint(*self.length_range)
        return ''.join(random.choices(self.AMINO_ACIDS, k=length))
        
    def mutate(self, mutation_rate: float = 0.1, operator_type: Optional[str] = None) -> 'Peptide':
        """
        对肽序列进行变异操作
        
        Args:
            mutation_rate: 变异率
            operator_type: 指定的变异算子类型。如果为None，则随机选择或使用默认。
            
        Returns:
            变异后的肽序列
        """
        mutated_sequence = list(self.sequence)
        i = 0
        
        # 预定义的两亲性/抗菌基序
        AMPHIPATHIC_MOTIFS = ["KWK", "KLLK", "RLLR", "KKLL", "LLKK", "FKLF"]
        
        # 获取 ESM 变异算子（如果可用）
        esm_operator = None
        if ESM_AVAILABLE:
            try:
                esm_operator = get_esm_mutation_operator()
            except Exception as e:
                pass

        # 决定当前序列可用的操作类型
        available_ops = ['random_replace']
        if esm_operator:
            available_ops.append('esm_replace')
            
        if self.fixed_length is None:
            available_ops.extend(['insert', 'delete', 'motif_insert'])
        else:
            # 固定长度可以进行基序替换 (motif_swap)
            available_ops.append('motif_swap')
            
        while i < len(mutated_sequence):
            if random.random() < mutation_rate:
                # 决定本次变异的具体操作
                current_op = operator_type if operator_type in available_ops else random.choice(available_ops)
                
                if current_op == 'esm_replace' and esm_operator:
                    try:
                        current_seq_str = "".join(mutated_sequence)
                        suggested_aa = esm_operator.suggest_mutation(current_seq_str, i)
                        mutated_sequence[i] = suggested_aa
                        i += 1
                    except Exception:
                        mutated_sequence[i] = random.choice(self.AMINO_ACIDS)
                        i += 1
                elif current_op == 'random_replace':
                    mutated_sequence[i] = random.choice(self.AMINO_ACIDS)
                    i += 1
                elif current_op == 'insert' and len(mutated_sequence) < self.length_range[1]:
                    mutated_sequence.insert(i, random.choice(self.AMINO_ACIDS))
                    i += 1
                elif current_op == 'delete' and len(mutated_sequence) > self.length_range[0]:
                    mutated_sequence.pop(i)
                    # 不增加 i，因为下一个字符移到了当前位置
                elif current_op == 'motif_insert' and len(mutated_sequence) + 2 <= self.length_range[1]:
                    motif = random.choice(AMPHIPATHIC_MOTIFS)
                    # 确保插入后不超过最大长度
                    space_left = self.length_range[1] - len(mutated_sequence)
                    motif = motif[:space_left]
                    for char in reversed(motif):
                        mutated_sequence.insert(i, char)
                    i += len(motif)
                elif current_op == 'motif_swap':
                    motif = random.choice(AMPHIPATHIC_MOTIFS)
                    # 替换掉接下来的 len(motif) 个字符（不超过序列长度）
                    swap_len = min(len(motif), len(mutated_sequence) - i)
                    for j in range(swap_len):
                        mutated_sequence[i+j] = motif[j]
                    i += swap_len
                else:
                    # 默认降级为随机替换
                    mutated_sequence[i] = random.choice(self.AMINO_ACIDS)
                    i += 1
            else:
                i += 1  # 没有变异，移动到下一个位置
        
        # 确保序列长度在允许范围内
        if self.fixed_length is not None:
            if len(mutated_sequence) != self.fixed_length:
                mutated_sequence = mutated_sequence[:self.fixed_length]
                while len(mutated_sequence) < self.fixed_length:
                    mutated_sequence.append(random.choice(self.AMINO_ACIDS))
        else:
            if len(mutated_sequence) < self.length_range[0]:
                while len(mutated_sequence) < self.length_range[0]:
                    mutated_sequence.append(random.choice(self.AMINO_ACIDS))
            elif len(mutated_sequence) > self.length_range[1]:
                mutated_sequence = mutated_sequence[:self.length_range[1]]
        
        new_peptide = Peptide(''.join(mutated_sequence), length_range=self.length_range, fixed_length=self.fixed_length)
        # 记录变异算子历史
        new_peptide.last_operator = operator_type
        return new_peptide
        
    def crossover(self, other: 'Peptide') -> Tuple['Peptide', 'Peptide']:
        """
        与另一个肽序列进行结构感知交叉操作 (Structure-Aware Crossover)
        
        Args:
            other: 另一个肽序列
            
        Returns:
            交叉后的两个子代肽序列
        """
        # 确保序列长度至少为2才能进行交叉
        if len(self.sequence) < 2 or len(other.sequence) < 2:
            return self.clone(), other.clone()
            
        def get_valid_crossover_points(peptide: 'Peptide') -> List[int]:
            # 获取结构中为 'C' (无规则卷曲) 的位置，避免切断 'H' (Alpha螺旋)
            if hasattr(peptide, 'structure') and peptide.structure is not None:
                # 排除首尾端点，确保至少有片段交换
                points = [i for i, ss in enumerate(peptide.structure) if ss == 'C' and 0 < i < len(peptide.sequence)]
                if points:
                    return points
            return list(range(1, len(peptide.sequence)))

        valid_points_self = get_valid_crossover_points(self)
        valid_points_other = get_valid_crossover_points(other)
            
        # 随机选择交叉点
        if self.fixed_length is not None:
            # 固定长度模式下，交叉点在序列内部且必须相同
            # 尽量寻找两人都是 'C' 的交集点
            common_points = list(set(valid_points_self) & set(valid_points_other))
            if common_points:
                crossover_point1 = random.choice(common_points)
            elif valid_points_self:
                crossover_point1 = random.choice(valid_points_self)
            else:
                crossover_point1 = random.randint(1, len(self.sequence) - 1)
            crossover_point2 = crossover_point1
        else:
            # 可变长度模式下，交叉点可以在各自的 'C' 区域任意位置
            crossover_point1 = random.choice(valid_points_self) if valid_points_self else random.randint(1, len(self.sequence) - 1)
            crossover_point2 = random.choice(valid_points_other) if valid_points_other else random.randint(1, len(other.sequence) - 1)
        
        # 生成子代序列
        child1_seq = self.sequence[:crossover_point1] + other.sequence[crossover_point2:]
        child2_seq = other.sequence[:crossover_point2] + self.sequence[crossover_point1:]
        
        # 确保子代序列长度在允许范围内
        if self.fixed_length is not None:
            # 固定长度模式下确保长度不变
            if len(child1_seq) != self.fixed_length:
                # 如果长度不正确，重新生成
                child1_seq = self.sequence[:self.fixed_length//2] + other.sequence[other.fixed_length//2:self.fixed_length]
            if len(child2_seq) != self.fixed_length:
                # 如果长度不正确，重新生成
                child2_seq = other.sequence[:self.fixed_length//2] + self.sequence[self.fixed_length//2:self.fixed_length]
        else:
            # 可变长度模式下确保长度在范围内
            if len(child1_seq) < self.length_range[0]:
                # 如果太短，添加氨基酸
                while len(child1_seq) < self.length_range[0]:
                    child1_seq += random.choice(self.AMINO_ACIDS)
            elif len(child1_seq) > self.length_range[1]:
                # 如果太长，截断序列
                child1_seq = child1_seq[:self.length_range[1]]
                
            if len(child2_seq) < self.length_range[0]:
                # 如果太短，添加氨基酸
                while len(child2_seq) < self.length_range[0]:
                    child2_seq += random.choice(self.AMINO_ACIDS)
            elif len(child2_seq) > self.length_range[1]:
                # 如果太长，截断序列
                child2_seq = child2_seq[:self.length_range[1]]
        
        child1 = Peptide(child1_seq, length_range=self.length_range, fixed_length=self.fixed_length)
        child2 = Peptide(child2_seq, length_range=self.length_range, fixed_length=self.fixed_length)
        
        return child1, child2
        
    def clone(self) -> 'Peptide':
        """
        克隆当前肽序列
        
        Returns:
            克隆的肽序列
        """
        clone_peptide = Peptide(self.sequence, length_range=self.length_range, fixed_length=self.fixed_length)
        clone_peptide.antimicrobial_activity = self.antimicrobial_activity
        clone_peptide.base_antimicrobial_activity = self.base_antimicrobial_activity
        clone_peptide.hemolytic_activity = self.hemolytic_activity
        clone_peptide.structure = self.structure
        clone_peptide.helicity = self.helicity
        clone_peptide.helicity_epsilon = self.helicity_epsilon
        clone_peptide.constraint_violation = self.constraint_violation
        clone_peptide.is_feasible = self.is_feasible
        # clone_peptide.cell_activity = self.cell_activity
        return clone_peptide
        
    @staticmethod
    def sequence_similarity(seq1: str, seq2: str) -> float:
        """
        计算两个肽序列之间的相似性（使用序列匹配器）
        
        Args:
            seq1: 第一个肽序列
            seq2: 第二个肽序列
            
        Returns:
            序列相似性值 (0-1)
        """
        return SequenceMatcher(None, seq1, seq2).ratio()
        
    def dominates(self, other: 'Peptide') -> bool:
        """
        判断当前解是否支配另一个解
        支配关系：所有目标都不差，至少有一个目标更好
        
        Args:
            other: 另一个肽序列
            
        Returns:
            是否支配另一个解
        """
        self_feasible = getattr(self, 'is_feasible', True)
        other_feasible = getattr(other, 'is_feasible', True)
        self_violation = float(getattr(self, 'constraint_violation', 0.0))
        other_violation = float(getattr(other, 'constraint_violation', 0.0))

        if self_feasible and not other_feasible:
            return True
        if not self_feasible and other_feasible:
            return False
        if not self_feasible and not other_feasible:
            if self_violation < other_violation:
                return True
            if self_violation > other_violation:
                return False
            if self.antimicrobial_activity is None or self.hemolytic_activity is None or \
               other.antimicrobial_activity is None or other.hemolytic_activity is None:
                return False
            not_worse = (self.antimicrobial_activity <= other.antimicrobial_activity and
                         self.hemolytic_activity <= other.hemolytic_activity)
            better = (self.antimicrobial_activity < other.antimicrobial_activity or
                      self.hemolytic_activity < other.hemolytic_activity)
            return not_worse and better

        if self.antimicrobial_activity is None or self.hemolytic_activity is None or \
           other.antimicrobial_activity is None or other.hemolytic_activity is None:
            return False
            
        # 检查是否所有目标都不差
        # 抗菌活性（MIC值）：值越小越好
        # 溶血活性（HC50的log值）：用户通常希望溶血活性低，即 HC50 高。
        # 如果 HC50 越高越好，那么 -HC50 越小越好。
        # 如果用户说“优化目标”，通常是指最小化。
        # 溶血性预测值 HC50，HC50 越大表示毒性越小（需要更高浓度才溶血）。
        # 所以目标应该是最大化 HC50，或者最小化 -HC50，或者最小化 1/HC50。
        # 通常在进化算法中，我们会最小化目标。
        # 如果我们取 log(HC50)，那么 HC50 越大，log(HC50) 越大。
        # 为了最小化，我们可以使用 -log(HC50)。
        
        # 假设我们要最小化溶血性，即最大化 HC50。
        # 所以目标 2 是 -log(HC50)。
        
        not_worse = (self.antimicrobial_activity <= other.antimicrobial_activity and 
                     self.hemolytic_activity <= other.hemolytic_activity)
                     
        # 检查是否至少有一个目标更好
        better = (self.antimicrobial_activity < other.antimicrobial_activity or 
                  self.hemolytic_activity < other.hemolytic_activity)
                  
        return not_worse and better
        
    def __str__(self) -> str:
        return f"Peptide(sequence='{self.sequence}', length={len(self.sequence)}, " \
               f"antimicrobial_activity={self.antimicrobial_activity}, " \
               f"hemolytic_activity={self.hemolytic_activity})"
               
    def __repr__(self) -> str:
        return self.__str__()
