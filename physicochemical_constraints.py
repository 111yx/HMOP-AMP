import numpy as np
from peptide import Peptide

class PhysicochemicalConstraints:
    """物理化学性质约束检查类"""
    
    @staticmethod
    def calculate_net_charge(sequence: str) -> float:
        """
        计算肽序列的净电荷
        
        Args:
            sequence: 肽序列
            
        Returns:
            净电荷值
        """
        # 带正电荷的氨基酸
        positive_aa = set("KRH")  # 赖氨酸(K), 精氨酸(R), 组氨酸(H)
        # 带负电荷的氨基酸
        negative_aa = set("DE")   # 天冬氨酸(D), 谷氨酸(E)
        
        positive_count = sum(1 for aa in sequence if aa in positive_aa)
        negative_count = sum(1 for aa in sequence if aa in negative_aa)
        
        return positive_count - negative_count
    
    @staticmethod
    def calculate_hydrophobicity(sequence: str) -> float:
        """
        计算肽序列的疏水性 (GRAVY值)
        
        Args:
            sequence: 肽序列
            
        Returns:
            疏水性值
        """
        # 每个氨基酸的疏水性值 (根据Kyte-Doolittle scale)
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        if not sequence:
            return 0.0
            
        total = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence)
        return total / len(sequence)
    
    @staticmethod
    def check_constraints(peptide: Peptide) -> bool:
        """
        检查肽序列是否满足物理化学约束条件
        
        Args:
            peptide: 肽序列对象
            
        Returns:
            是否满足约束条件
        """
        # 计算净电荷 (+2 到 +9)
        net_charge = PhysicochemicalConstraints.calculate_net_charge(peptide.sequence)
        if not (2 <= net_charge <= 9):
            return False
            
        # 计算疏水性 (-0.5 到 +1.0)
        hydrophobicity = PhysicochemicalConstraints.calculate_hydrophobicity(peptide.sequence)
        if not (-0.5 <= hydrophobicity <= 1.0):
            return False
            
        # 检查长度 (5到40)
        if not (5 <= len(peptide.sequence) <= 40):
            return False
            
        return True
    
    @staticmethod
    def calculate_all_properties(peptide: Peptide) -> dict:
        """
        计算肽的所有物理化学性质
        
        Args:
            peptide: 肽序列对象
            
        Returns:
            包含所有性质的字典
        """
        return {
            "net_charge": PhysicochemicalConstraints.calculate_net_charge(peptide.sequence),
            "hydrophobicity": PhysicochemicalConstraints.calculate_hydrophobicity(peptide.sequence),
            "length": len(peptide.sequence)
        }