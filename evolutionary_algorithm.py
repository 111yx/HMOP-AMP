import random
from typing import List, Callable, Optional
import numpy as np
import pandas as pd
import pickle
import math
import tempfile
import shutil
from peptide import Peptide
from population import Population
from physicochemical_constraints import PhysicochemicalConstraints
from membrane_system import MembraneSystem
import sys
import os
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI初始化问题
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统：指定默认字体为"黑体"（支持中文）
plt.rcParams['axes.unicode_minus'] = False 

# 添加AMP预测器导入
from amp_predictor import AMPPredictor
# 添加结构预测器导入
from structure_predictor import StructurePredictor
from adaptive_operator import AdaptiveOperatorSelector

# 尝试导入 ESM 变异算子进行初始化
try:
    from esm_mutation_operator import get_esm_mutation_operator
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: esm_mutation_operator module not found in evolutionary_algorithm.")

class NicheManager:
    """小生境管理器类"""
    
    def __init__(self, niche_radius: float = 0.3):
        """
        初始化小生境管理器
        
        Args:
            niche_radius: 小生境半径，用于确定序列相似性的阈值
        """
        self.niche_radius = niche_radius
        
    def calculate_niche_count(self, individual: Peptide, population: List[Peptide]) -> int:
        """
        计算给定个体的小生境计数（在指定半径内相似个体的数量）
        
        Args:
            individual: 目标个体
            population: 种群列表
            
        Returns:
            小生境计数
        """
        count = 0
        for other in population:
            # 计算序列相似性
            similarity = Peptide.sequence_similarity(individual.sequence, other.sequence)
            # 如果相似性高于阈值，则认为在同一小生境内
            if similarity > (1 - self.niche_radius):
                count += 1
        return count
    
    def apply_niche_effect(self, population: List[Peptide]):
        """
        对种群应用小生境效应，更新每个个体的小生境计数和因子
        
        Args:
            population: 种群列表
        """
        for individual in population:
            individual.niche_count = self.calculate_niche_count(individual, population)
            # 小生境因子与小生境计数成反比，避免过度拥挤
            if individual.niche_count > 0:
                individual.niche_factor = 1.0 / individual.niche_count
            else:
                individual.niche_factor = 1.0

class EvolutionaryAlgorithm:
    """多目标多种群进化算法类"""
    
    def __init__(self, 
                 membrane1_size: int = 50,  # 物理化学探索膜
                 membrane2_size: int = 50,  # 结构约束膜
                 membrane3_size: int = 50,  # 溶血安全膜
                 max_generations: int = 3,  # main函数中没有设置时的默认值
                 length_range: tuple = (5, 30),
                 fixed_length: Optional[int] = None,
                 helicity_eps_start: float = 0.3,
                 helicity_eps_end: float = 0.45,
                 helicity_eps_fixed: Optional[float] = None,
                 helicity_penalty_weight: float = 0.0,
                 skin_helicity_eps: float = 0.3,
                 stagnation_threshold: float = 0.01,
                 stagnation_window: int = 10,
                 ablation_mode: str = 'full',
                 fixed_comm_interval: int = 5,
                 base_migration_size: int = 5,
                 base_p_out: float = 0.3,
                 interval_min: int = 3,
                 interval_max: int = 10,
                 # 向后兼容的参数
                 auxiliary_population_size: Optional[int] = None,
                 main_population_size: Optional[int] = None):
        """
        初始化进化算法
        
        Args:
            membrane1_size: 膜1（物理化学探索膜）的种群大小
            membrane2_size: 膜2（结构约束膜）的种群大小
            membrane3_size: 膜3（溶血安全膜）的种群大小
            max_generations: 最大进化代数
            length_range: 肽序列长度范围
            fixed_length: 固定长度，如果指定则使用固定长度生成序列
            auxiliary_population_size: 辅助种群大小（向后兼容）
            main_population_size: 主种群大小（向后兼容）
        """
        # 处理向后兼容的参数
        if auxiliary_population_size is not None:
            membrane1_size = auxiliary_population_size
        if main_population_size is not None:
            membrane2_size = main_population_size
            membrane3_size = main_population_size
        
        self.membrane1_size = membrane1_size
        self.membrane2_size = membrane2_size
        self.membrane3_size = membrane3_size
        self.max_generations = max_generations
        self.length_range = length_range    # 肽序列长度
        self.fixed_length = fixed_length    # 固定长度
        self.helicity_eps_start = helicity_eps_start
        self.helicity_eps_end = helicity_eps_end
        self.helicity_eps_fixed = helicity_eps_fixed
        self.helicity_penalty_weight = helicity_penalty_weight
        self.skin_helicity_eps = skin_helicity_eps
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_window = stagnation_window
        self.ablation_mode = ablation_mode
        self.fixed_comm_interval = fixed_comm_interval
        self.base_migration_size = base_migration_size
        self.base_p_out = base_p_out
        self.interval_min = interval_min
        self.interval_max = interval_max
        
        # 初始化日志系统
        self._setup_logging()
        
        # 创建膜计算系统
        self.membrane_system = MembraneSystem(
            membrane1_size=membrane1_size,
            membrane2_size=membrane2_size,
            membrane3_size=membrane3_size,
            length_range=length_range,
            fixed_length=fixed_length,
            helicity_eps_start=helicity_eps_start,
            helicity_eps_end=helicity_eps_end,
            skin_helicity_eps=skin_helicity_eps,
            stagnation_threshold=stagnation_threshold,
            stagnation_window=stagnation_window,
            ablation_mode=ablation_mode,
            fixed_comm_interval=fixed_comm_interval,
            base_migration_size=base_migration_size,
            base_p_out=base_p_out,
            interval_min=interval_min,
            interval_max=interval_max,
            logger=self.logger
        )
        
        # 初始化AMP预测器
        try:
            self.amp_predictor = AMPPredictor()
            self.logger.info("AMP预测器初始化成功")
        except Exception as e:
            self.logger.warning(f"无法初始化AMP预测器: {e}")
            self.logger.info("将使用随机值作为抗菌活性预测结果")
            self.amp_predictor = None
            
        # 初始化结构预测器 (S4PRED)
        try:
            # 假设权重文件在 data 目录或当前目录下，如果不存在将使用启发式模式
            # 使用单模型模式 (只加载 weights_1.pt)
            self.structure_predictor = StructurePredictor(weights_dir="s4pred_weights", single_model_mode=True)
            self.logger.info("结构预测器初始化成功 (单模型模式)")
        except Exception as e:
            self.logger.warning(f"无法初始化结构预测器: {e}")
            self.structure_predictor = None
            
        # 初始化 ESM 变异算子
        if ESM_AVAILABLE:
            try:
                self.logger.info("正在初始化 ESM 变异算子...")
                get_esm_mutation_operator()
                self.logger.info("ESM 变异算子初始化成功")
            except Exception as e:
                self.logger.warning(f"无法初始化 ESM 变异算子: {e}")
        
        # 添加toxinpred3-main到Python路径
        toxinpred_path = os.path.join(os.path.dirname(__file__), 'toxinpred3-main')
        if toxinpred_path not in sys.path:
            sys.path.append(toxinpred_path)
        
        # 知识迁移间隔（每10代进行一次知识迁移）
        self.knowledge_transfer_interval = 10
        
        # 初始化小生境管理器
        self.niche_manager = NicheManager(niche_radius=0.3)
        
        # 初始化自适应算子选择器（全局备用，主要使用膜级选择器）
        available_ops = ['random_replace']
        if ESM_AVAILABLE:
            available_ops.append('esm_replace')
        if self.fixed_length is None:
            available_ops.extend(['insert', 'delete', 'motif_insert'])
        else:
            available_ops.append('motif_swap')
        self.operator_selector = AdaptiveOperatorSelector(available_ops)

        # 将算子列表传给膜计算系统，令每个膜创建独立算子选择器
        self.membrane_system.set_operator_lists(available_ops)

        # 兼容主种群/辅助种群属性
        self.main_population = self.membrane_system.membrane2
        self.auxiliary_population = self.membrane_system.membrane1
    
    def _setup_logging(self):
        """
        初始化日志系统
        """
        # 创建日志目录
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"创建日志目录: {log_dir}")
        else:
            print(f"日志目录已存在: {log_dir}")
        
        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'evolution_log_{timestamp}.log')
        print(f"日志文件路径: {log_file}")
        
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # 创建logger
        self.logger = logging.getLogger('EvolutionaryAlgorithm')
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        self.logger.handlers.clear()
        
        # 文件handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        print("添加文件handler成功")
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        print("添加控制台handler成功")
        
        # 记录初始信息
        self.logger.info("=" * 80)
        self.logger.info("抗菌肽进化算法启动")
        self.logger.info("=" * 80)
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info(f"膜1大小: {self.membrane1_size}")
        self.logger.info(f"膜2大小: {self.membrane2_size}")
        self.logger.info(f"膜3大小: {self.membrane3_size}")
        self.logger.info(f"最大进化代数: {self.max_generations}")
        self.logger.info(f"序列长度范围: {self.length_range}")
        if self.fixed_length:
            self.logger.info(f"固定序列长度: {self.fixed_length}")
        self.logger.info("=" * 80)
        print("日志系统初始化成功")
    
    def _get_helicity_epsilon(self, generation: int) -> float:
        if self.helicity_eps_fixed is not None:
            return float(self.helicity_eps_fixed)
        if self.max_generations <= 1:
            return float(self.helicity_eps_end)
        progress = max(0.0, min(1.0, generation / float(self.max_generations - 1)))
        return float(self.helicity_eps_start + (self.helicity_eps_end - self.helicity_eps_start) * progress)
        
    def _predict_hemolytic_activity(self, peptides: List[Peptide]) -> List[float]:
        """
        预测肽序列的溶血活性（使用 HemoPI2.0）
        
        Args:
            peptides: 肽序列列表
            
        Returns:
            溶血活性预测值列表（取 log 值）
        """
        if not peptides:
            return []
            
        sequences = [peptide.sequence for peptide in peptides]
        
        # 创建临时工作目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 准备输入文件
            input_file = os.path.join(temp_dir, 'input.seq')
            with open(input_file, 'w') as f:
                for seq in sequences:
                    f.write(f"{seq}\n")
            
            # 2. 设置路径
            hemopi_dir = os.path.join(os.path.dirname(__file__), 'hemopi2-main')
            calc_script = os.path.join(hemopi_dir, 'model', 'composition_calculate_hemopi2_2.py')
            model_path = os.path.join(hemopi_dir, 'model', 'HemoPI2_reg.sav')
            out_features = os.path.join(temp_dir, 'features.csv')
            
            # 3. 计算特征
            import subprocess
            current_dir = os.getcwd()
            try:
                # 切换到 hemopi2-main 目录运行，因为脚本内部使用了相对路径
                os.chdir(hemopi_dir)
                # 执行特征计算脚本
                # python Model/composition_calculate_hemopi2_2.py {input_file} {wd} {out_features}
                # 使用 sys.executable 确保使用当前的 Python 解释器
                cmd = [sys.executable, calc_script, input_file, temp_dir, out_features]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"特征计算脚本执行失败 (退出码 {result.returncode}):")
                    print(f"标准输出: {result.stdout}")
                    print(f"标准错误: {result.stderr}")
                
                # 4. 加载模型并预测
                if os.path.exists(out_features):
                    df = pd.read_csv(out_features)
                    with open(model_path, 'rb') as f:
                        clf = pickle.load(f)
                    
                    # 预测原始值 (raw_pred)
                    raw_pred = clf.predict(df)
                    
                    # 根据 hemopi2_regression.py:62, HC50 = exp(-raw_pred)
                    # 用户要求输出 -lg(HC50):
                    # -lg(HC50) = -ln(HC50) / ln(10)
                    # 因为 raw_pred = -ln(HC50)
                    # 所以 -lg(HC50) = raw_pred / ln(10)
                    log10_hc50 = raw_pred / np.log(10)
                    
                    return log10_hc50.tolist()
                else:
                    print(f"警告: 溶血活性特征文件未生成: {out_features}")
                    return [0.0] * len(peptides)
            except Exception as e:
                print(f"溶血活性预测出错: {e}")
                return [0.0] * len(peptides)
            finally:
                os.chdir(current_dir)

    def _predict_cell_activity(self, peptides: List[Peptide]) -> List[float]:
        """
        预测肽序列的细胞活性（使用ToxinPred3）
        
        Args:
            peptides: 肽序列列表
            
        Returns:
            细胞活性预测值列表（使用PPV值作为毒性值）
        """
        # 暂时注释掉细胞毒性预测
        return [0.0 for _ in peptides]
        
        # if not peptides:
        #     return []
        #     
        # try:
        #     # 导入修改后的toxinpred3模块
        #     import toxinpred3
        #     
        #     # 准备序列数据
        #     sequences = [peptide.sequence for peptide in peptides]
        #     
        #     # 直接调用预测函数
        #     result = toxinpred3.predict_toxicity(sequences, model=2, threshold=0.38, display=2)
        #     
        #     # 提取PPV值作为细胞毒性值
        #     if 'PPV' in result.columns:
        #         ppv_values = result['PPV'].tolist()
        #         # 确保返回正确的数量值
        #         while len(ppv_values) < len(peptides):
        #             ppv_values.append(0.0)
        #         # 如果结果数量超过，截断到正确长度
        #         ppv_values = ppv_values[:len(peptides)]
        #         return ppv_values
        #     else:
        #         print("ToxinPred3结果中未找到PPV列")
        #         # 返回默认值
        #         return [0.0 for _ in peptides]
        #             
        # except Exception as e:
        #     print(f"细胞活性预测出错: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     # 返回默认值
        #     return [0.0 for _ in peptides]
    
    def _evaluate_population_list(self, individuals: List[Peptide], generation: int = 0):
        """
        评估单个肽序列列表（不依赖 Population 对象）。
        供膜计算系统内部评估子代个体时调用。
        """
        class _FakePop:
            pass
        fake = _FakePop()
        fake.individuals = individuals
        self._evaluate_population(fake, generation)  # type: ignore

    def _evaluate_population(self, population: Population, generation: int = 0):
        """
        评估种群中所有个体的目标值
        
        Args:
            population: 要评估的种群
            generation: 当前进化代数，用于调整长度惩罚
        """
        peptides_need_mic = [p for p in population.individuals if getattr(p, 'base_antimicrobial_activity', None) is None]
        peptides_need_hemo = [p for p in population.individuals if p.hemolytic_activity is None]
        
        if peptides_need_mic:
            sequences = [p.sequence for p in peptides_need_mic]
            if self.amp_predictor is not None:
                try:
                    amp_results = self.amp_predictor.predict(sequences)
                    mic_values = [result[1] for result in amp_results]
                except Exception as e:
                    print(f"抗菌活性预测出错，使用随机值: {e}")
                    mic_values = [random.random() for _ in sequences]
            else:
                mic_values = [random.random() for _ in sequences]
            
            for peptide, mic in zip(peptides_need_mic, mic_values):
                peptide.base_antimicrobial_activity = mic
        
        if peptides_need_hemo:
            hemo_values = self._predict_hemolytic_activity(peptides_need_hemo)
            for peptide, hemo in zip(peptides_need_hemo, hemo_values):
                peptide.hemolytic_activity = hemo
        
        helicity_epsilon = self._get_helicity_epsilon(generation) if self.structure_predictor else None
        
        for peptide in population.individuals:
            base_mic = getattr(peptide, 'base_antimicrobial_activity', None)
            if base_mic is not None:
                peptide.antimicrobial_activity = base_mic
            elif peptide.antimicrobial_activity is None:
                peptide.antimicrobial_activity = 0.0
            
            penalty = self._calculate_length_penalty(peptide.sequence, generation)
            peptide.antimicrobial_activity += penalty
            
            if self.structure_predictor and peptide.structure is None:
                structure, helicity = self.structure_predictor.predict(peptide.sequence)
                peptide.structure = structure
                peptide.helicity = helicity
            
            peptide.helicity_epsilon = helicity_epsilon
            if helicity_epsilon is None:
                peptide.constraint_violation = 0.0
                peptide.is_feasible = True
            else:
                violation = max(0.0, helicity_epsilon - float(getattr(peptide, 'helicity', 0.0)))
                peptide.constraint_violation = violation
                peptide.is_feasible = violation <= 0.0
                if self.helicity_penalty_weight > 0.0 and violation > 0.0:
                    peptide.antimicrobial_activity += violation * float(self.helicity_penalty_weight)
        
        self.niche_manager.apply_niche_effect(population.individuals)
    
    def _calculate_length_penalty(self, sequence: str, generation: int) -> float:
        """
        计算序列的长度惩罚
        
        Args:
            sequence: 肽序列
            generation: 当前进化代数
            
        Returns:
            长度惩罚值
        """
        min_len, max_len = self.length_range
        
        # 惩罚阈值：超过最大长度的75%才施加惩罚
        penalty_threshold = max_len * 0.75
        
        sequence_len = len(sequence)
        if sequence_len <= penalty_threshold:
            # 低于阈值，无惩罚
            return 0.0
        
        # 基础惩罚权重
        base_weight = 0.02
        
        # 惩罚随进化代数递增（每代增加1%）
        generation_factor = 1.0 + (generation / self.max_generations) * 2.0  # 最大3倍惩罚
        
        # 惩罚强度与超过阈值的程度成正比
        excess_ratio = (sequence_len - penalty_threshold) / (max_len - penalty_threshold)
        
        # 计算最终惩罚
        penalty = base_weight * generation_factor * excess_ratio
        
        return penalty
    
    def _perform_crossover_and_mutation(self, population: Population, 
                                      crossover_rate: float = 0.8, 
                                      mutation_rate: float = 0.1) -> List[Peptide]:
        """
        对种群执行交叉和变异操作
        
        Args:
            population: 要操作的种群
            crossover_rate: 交叉率
            mutation_rate: 变异率
            
        Returns:
            新生成的个体列表
        """
        new_individuals = []
        
        # 生成新一代个体
        while len(new_individuals) < population.size:
            # 选择父代
            parent1, parent2 = population.select_parents()
            
            # 交叉操作
            if random.random() < crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
                
            # 选择变异算子
            op_type1 = self.operator_selector.select_operator()
            op_type2 = self.operator_selector.select_operator()
            
            # 变异操作
            child1 = child1.mutate(mutation_rate, operator_type=op_type1)
            child2 = child2.mutate(mutation_rate, operator_type=op_type2)
            
            # 如果是受约束的种群，确保子代满足约束条件
            if population.is_constrained:
                from physicochemical_constraints import PhysicochemicalConstraints
                # 通过重新变异直到满足约束，或者直接丢弃不满足约束的个体
                # 这里采用重新变异的方式，最多尝试10次
                max_attempts = 10
                attempts = 0
                while not PhysicochemicalConstraints.check_constraints(child1) and attempts < max_attempts:
                    child1 = child1.mutate(mutation_rate, operator_type=op_type1)
                    attempts += 1
                    
                attempts = 0
                while not PhysicochemicalConstraints.check_constraints(child2) and attempts < max_attempts:
                    child2 = child2.mutate(mutation_rate, operator_type=op_type2)
                    attempts += 1
            
            new_individuals.append(child1)
            if len(new_individuals) < population.size:
                new_individuals.append(child2)
                
        # 如果是受约束的种群，只返回满足约束条件的个体
        if population.is_constrained:
            from physicochemical_constraints import PhysicochemicalConstraints
            filtered_individuals = []
            for individual in new_individuals:
                if PhysicochemicalConstraints.check_constraints(individual):
                    filtered_individuals.append(individual)
            return filtered_individuals[:population.size]
        else:
            return new_individuals[:population.size]
    
    def _knowledge_transfer(self):
        """
        从辅助种群向主种群进行知识迁移
        """
        # 确保辅助种群中的个体已被评估
        # 使用当前进化代数的最大值作为惩罚系数
        self._evaluate_population(self.auxiliary_population, self.max_generations)
        
        # 从辅助种群选择最好的个体
        best_auxiliary = self.auxiliary_population.get_best_individuals(
            n=max(1, self.main_population_size // 10)
        )
        
        # 获取主种群中已存在的序列，避免重复
        existing_sequences = {individual.sequence for individual in self.main_population.individuals}
        
        # 将不重复的个体添加到主种群
        added_count = 0
        for peptide in best_auxiliary:
            if peptide.sequence not in existing_sequences:
                # 克隆个体以避免引用问题
                cloned_peptide = peptide.clone()
                self.main_population.add_individual(cloned_peptide)
                existing_sequences.add(peptide.sequence)
                added_count += 1
                
        print(f"知识迁移: 从辅助种群向主种群迁移了 {added_count} 个个体")
    
    def _update_operator_rewards(self, new_individuals: List[Peptide], selected_population: List[Peptide]):
        """
        根据子代是否存活以及是否进入帕累托前沿来更新变异算子的奖励
        """
        if not new_individuals:
            return
            
        selected_seqs = {ind.sequence for ind in selected_population}
        # 获取当前种群的帕累托前沿
        fronts = self._non_dominated_sort(selected_population)
        pareto_front_seqs = {ind.sequence for ind in fronts[0]} if fronts else set()
        
        for ind in new_individuals:
            op = getattr(ind, 'last_operator', None)
            if op:
                reward = 0.0
                if ind.sequence in selected_seqs:
                    reward += 1.0  # 存活奖励
                if ind.sequence in pareto_front_seqs:
                    reward += 2.0  # 帕累托前沿奖励
                    
                # 更新算子分数
                self.operator_selector.update(op, reward)

    def _environmental_selection(self, combined_population: List[Peptide], population_size: int, role: str = 'general') -> List[Peptide]:
        """
        环境选择：从合并的种群中选择指定数量的个体
        
        Args:
            combined_population: 合并后的种群（父代+子代）
            population_size: 要选择的个体数量
            role: 膜的角色，用于膜2的选择压力加权
            
        Returns:
            选择出的个体列表
        """
        # 对合并种群进行非支配排序
        fronts = self._non_dominated_sort(combined_population)
        
        # 选择个体
        new_population = []
        front_index = 0
        
        # 按前沿逐个选择个体，直到超出种群大小
        while len(new_population) + len(fronts[front_index]) <= population_size:
            # 计算当前前沿中个体的拥挤距离
            fronts[front_index] = self._calculate_crowding_distance(fronts[front_index], role)
            new_population.extend(fronts[front_index])
            front_index += 1
            if front_index >= len(fronts):
                break
                
        # 如果还需要更多个体，则根据拥挤距离在当前前沿中选择
        if len(new_population) < population_size and front_index < len(fronts):
            # 计算当前前沿中个体的拥挤距离
            fronts[front_index] = self._calculate_crowding_distance(fronts[front_index], role)
            # 按拥挤距离降序排列
            fronts[front_index].sort(key=lambda x: x.crowding_distance, reverse=True)
            # 添加剩余需要的个体
            remaining = population_size - len(new_population)
            new_population.extend(fronts[front_index][:remaining])
            
        # 确保没有重复的序列
        unique_population = []
        seen_sequences = set()
        for individual in new_population:
            if individual.sequence not in seen_sequences:
                unique_population.append(individual)
                seen_sequences.add(individual.sequence)
                
        # 如果去重后数量不足，补充随机个体
        while len(unique_population) < population_size:
            random_individual = random.choice(combined_population)
            if random_individual.sequence not in seen_sequences:
                unique_population.append(random_individual)
                seen_sequences.add(random_individual.sequence)
                
        return unique_population[:population_size]
    
    def _non_dominated_sort(self, population: List[Peptide]) -> List[List[Peptide]]:
        """
        非支配排序
        
        Args:
            population: 要排序的种群
            
        Returns:
            分层的非支配前沿列表
        """
        fronts = [[]]  # 第一前沿
        
        # 初始化支配计数和被支配集合
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            
        # 计算支配关系
        for i, p in enumerate(population):
            for q in population[i+1:]:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                    q.domination_count += 1
                elif q.dominates(p):
                    q.dominated_solutions.append(p)
                    p.domination_count += 1
                    
        # 找到第一前沿（支配计数为0的个体）
        for individual in population:
            if individual.domination_count == 0:
                fronts[0].append(individual)
                
        # 构建后续前沿
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)
            
        # 移除最后一个空前沿
        if len(fronts[-1]) == 0:
            fronts.pop()
            
        return fronts
    
    def _calculate_crowding_distance(self, front: List[Peptide], role: str = 'general') -> List[Peptide]:
        """
        计算前沿中个体的拥挤距离
        
        Args:
            front: 一个前沿中的个体列表
            role: 膜的角色，用于膜2的选择压力加权
            
        Returns:
            更新了拥挤距离的个体列表
        """
        size = len(front)
        if size == 0:
            return front
            
        # 初始化拥挤距离
        for individual in front:
            individual.crowding_distance = 0.0
            
        # 如果只有一个个体，设置无穷大拥挤距离
        if size == 1:
            front[0].crowding_distance = float('inf')
            return front
            
        # 获取目标值列表
        # 确保所有目标值都不为None
        valid_front = [ind for ind in front if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None]
        if len(valid_front) == 0:
            return front
            
        antimicrobial_activities = [ind.antimicrobial_activity for ind in valid_front]
        hemolytic_activities = [ind.hemolytic_activity for ind in valid_front]
        
        # 按抗菌活性排序
        sorted_front1 = sorted(valid_front, key=lambda x: x.antimicrobial_activity)
        sorted_front1[0].crowding_distance = float('inf')
        sorted_front1[-1].crowding_distance = float('inf')
        
        if len(sorted_front1) > 2:
            min_obj1 = min(antimicrobial_activities)
            max_obj1 = max(antimicrobial_activities)
            scale1 = max_obj1 - min_obj1 if max_obj1 != min_obj1 else 1
            
            for i in range(1, len(sorted_front1) - 1):
                # 确保所有值都不是None
                if (sorted_front1[i].antimicrobial_activity is not None and 
                    sorted_front1[i-1].antimicrobial_activity is not None and 
                    sorted_front1[i+1].antimicrobial_activity is not None):
                    # 使用类型检查确保值不为None
                    activity_i_plus_1 = sorted_front1[i+1].antimicrobial_activity
                    activity_i_minus_1 = sorted_front1[i-1].antimicrobial_activity
                    if activity_i_plus_1 is not None and activity_i_minus_1 is not None:
                        distance = (activity_i_plus_1 - activity_i_minus_1) / scale1
                        sorted_front1[i].crowding_distance += distance
                        
        # 按溶血活性排序
        sorted_front2 = sorted(valid_front, key=lambda x: x.hemolytic_activity)
        sorted_front2[0].crowding_distance = float('inf')
        sorted_front2[-1].crowding_distance = float('inf')
        
        if len(sorted_front2) > 2:
            min_obj2 = min(hemolytic_activities)
            max_obj2 = max(hemolytic_activities)
            scale2 = max_obj2 - min_obj2 if max_obj2 != min_obj2 else 1
            
            # 膜3（溶血安全膜）对f2（溶血）目标进行额外加权
            weight = 1.5 if role == 'safety' else 1.0
            
            for i in range(1, len(sorted_front2) - 1):
                # 确保所有值都不是None
                if (sorted_front2[i].hemolytic_activity is not None and 
                    sorted_front2[i-1].hemolytic_activity is not None and 
                    sorted_front2[i+1].hemolytic_activity is not None):
                    # 使用类型检查确保值不为None
                    activity_i_plus_1 = sorted_front2[i+1].hemolytic_activity
                    activity_i_minus_1 = sorted_front2[i-1].hemolytic_activity
                    if activity_i_plus_1 is not None and activity_i_minus_1 is not None:
                        distance = (activity_i_plus_1 - activity_i_minus_1) / scale2
                        # 应用权重
                        sorted_front2[i].crowding_distance += distance * weight
                    
        # 按拥挤距离降序排列
        return sorted(front, key=lambda x: x.crowding_distance, reverse=True)
    
    def _save_pareto_front_to_txt(self, generation: int):
        """
        将帕累托前沿保存到文本文件
        
        Args:
            generation: 当前代数
        """
        # 获取所有膜的种群
        membrane1, membrane2, membrane3 = self.membrane_system.get_membrane_populations()
        
        # 评估种群目标值
        self._evaluate_population(membrane1, generation)
        self._evaluate_population(membrane2, generation)
        self._evaluate_population(membrane3, generation)
        
        # 获取三个膜的帕累托前沿
        m1_pareto_front = membrane1.get_pareto_front()
        m2_pareto_front = membrane2.get_pareto_front()
        m3_pareto_front = membrane3.get_pareto_front()
        
        # 保存到同一个文件中
        filename = f"pareto_front_generation_{generation}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"第 {generation} 代帕累托前沿\n")
            f.write("=" * 70 + "\n\n")
            
            # 写入膜1（物理化学探索膜）帕累托前沿
            f.write("膜1（物理化学探索膜）帕累托前沿:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'序号':<5} {'序列':<30} {'长度':<6} {'抗菌活性':<10} {'溶血活性':<10} {'helicity':<8} {'ε':<6} {'violation':<10} {'feasible':<8}\n")
            f.write("-" * 50 + "\n")
            for i, peptide in enumerate(m1_pareto_front, 1):
                hel = float(getattr(peptide, "helicity", 0.0))
                viol = float(getattr(peptide, "constraint_violation", 0.0))
                feas = bool(getattr(peptide, "is_feasible", True))
                eps = getattr(membrane1, "helicity_eps", None)
                eps_s = f"{eps:.2f}" if eps is not None else "-"
                f.write(f"{i:<5} {peptide.sequence:<30} {len(peptide.sequence):<6} "
                        f"{peptide.antimicrobial_activity:<10.4f} {peptide.hemolytic_activity:<10.4f} "
                        f"{hel:<8.3f} {eps_s:<6} {viol:<10.3f} {str(feas):<8}\n")
            
            f.write("\n")
            
            # 写入膜2（结构约束膜）帕累托前沿
            f.write("膜2（结构约束膜）帕累托前沿:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'序号':<5} {'序列':<30} {'长度':<6} {'抗菌活性':<10} {'溶血活性':<10} {'helicity':<8} {'ε':<6} {'violation':<10} {'feasible':<8}\n")
            f.write("-" * 50 + "\n")
            for i, peptide in enumerate(m2_pareto_front, 1):
                hel = float(getattr(peptide, "helicity", 0.0))
                viol = float(getattr(peptide, "constraint_violation", 0.0))
                feas = bool(getattr(peptide, "is_feasible", True))
                eps = getattr(membrane2, "helicity_eps", None)
                eps_s = f"{eps:.2f}" if eps is not None else "-"
                f.write(f"{i:<5} {peptide.sequence:<30} {len(peptide.sequence):<6} "
                        f"{peptide.antimicrobial_activity:<10.4f} {peptide.hemolytic_activity:<10.4f} "
                        f"{hel:<8.3f} {eps_s:<6} {viol:<10.3f} {str(feas):<8}\n")
            
            f.write("\n")
            
            # 写入膜3（溶血安全膜）帕累托前沿
            f.write("膜3（溶血安全膜）帕累托前沿:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'序号':<5} {'序列':<30} {'长度':<6} {'抗菌活性':<10} {'溶血活性':<10} {'helicity':<8} {'ε':<6} {'violation':<10} {'feasible':<8}\n")
            f.write("-" * 50 + "\n")
            for i, peptide in enumerate(m3_pareto_front, 1):
                hel = float(getattr(peptide, "helicity", 0.0))
                viol = float(getattr(peptide, "constraint_violation", 0.0))
                feas = bool(getattr(peptide, "is_feasible", True))
                eps = getattr(membrane3, "helicity_eps", None)
                eps_s = f"{eps:.2f}" if eps is not None else "-"
                f.write(f"{i:<5} {peptide.sequence:<30} {len(peptide.sequence):<6} "
                        f"{peptide.antimicrobial_activity:<10.4f} {peptide.hemolytic_activity:<10.4f} "
                        f"{hel:<8.3f} {eps_s:<6} {viol:<10.3f} {str(feas):<8}\n")
            
            # 添加膜1所有个体信息
            f.write("\n膜1（物理化学探索膜）所有个体信息:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'肽序列':<30} {'长度':<6} {'抗菌活性':<10} {'溶血活性':<10}\n")
            f.write("-" * 50 + "\n")
            for i, peptide in enumerate(membrane1.individuals):
                f.write(f"{peptide.sequence:<30} {len(peptide.sequence):<6} "
                        f"{peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0:<10.4f} "
                        f"{peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0:<10.4f}\n")
            
            # 添加膜2所有个体信息
            f.write("\n膜2（结构约束膜）所有个体信息:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'肽序列':<30} {'长度':<6} {'抗菌活性':<10} {'溶血活性':<10}\n")
            f.write("-" * 50 + "\n")
            for i, peptide in enumerate(membrane2.individuals):
                f.write(f"{peptide.sequence:<30} {len(peptide.sequence):<6} "
                        f"{peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0:<10.4f} "
                        f"{peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0:<10.4f}\n")
            
            # 添加膜3所有个体信息
            f.write("\n膜3（溶血安全膜）所有个体信息:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'肽序列':<30} {'长度':<6} {'抗菌活性':<10} {'溶血活性':<10}\n")
            f.write("-" * 50 + "\n")
            for i, peptide in enumerate(membrane3.individuals):
                f.write(f"{peptide.sequence:<30} {len(peptide.sequence):<6} "
                        f"{peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0:<10.4f} "
                        f"{peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0:<10.4f}\n")
                
        self.logger.info(f"第 {generation} 代帕累托前沿已保存到 {filename}")
    
    def _plot_pareto_front(self, generation: int):
        """
        绘制帕累托前沿图
        
        Args:
            generation: 当前代数
        """
        # 获取所有膜的种群
        membrane1, membrane2, membrane3 = self.membrane_system.get_membrane_populations()
        
        # 评估种群目标值
        self._evaluate_population(membrane1, generation)
        self._evaluate_population(membrane2, generation)
        self._evaluate_population(membrane3, generation)
        
        # 获取三个膜的帕累托前沿
        m1_pareto_front = membrane1.get_pareto_front()
        m2_pareto_front = membrane2.get_pareto_front()
        m3_pareto_front = membrane3.get_pareto_front()
        
        # 获取所有个体（用于背景散点图）
        m1_all = membrane1.individuals
        m2_all = membrane2.individuals
        m3_all = membrane3.individuals
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        def _split_points(individuals):
            feasible = []
            infeasible = []
            for ind in individuals:
                if ind.antimicrobial_activity is None or ind.hemolytic_activity is None:
                    continue
                if getattr(ind, "is_feasible", True):
                    feasible.append(ind)
                else:
                    infeasible.append(ind)
            return feasible, infeasible
        
        # 对三个膜的个体分别进行处理
        m1_feasible, m1_infeasible = _split_points(m1_all)
        m2_feasible, m2_infeasible = _split_points(m2_all)
        m3_feasible, m3_infeasible = _split_points(m3_all)
        
        # 提取可行个体的数据
        m1_antimicrobial_all = [ind.antimicrobial_activity for ind in m1_feasible]
        m1_hemolytic_all = [ind.hemolytic_activity for ind in m1_feasible]
        m2_antimicrobial_all = [ind.antimicrobial_activity for ind in m2_feasible]
        m2_hemolytic_all = [ind.hemolytic_activity for ind in m2_feasible]
        m3_antimicrobial_all = [ind.antimicrobial_activity for ind in m3_feasible]
        m3_hemolytic_all = [ind.hemolytic_activity for ind in m3_feasible]
        
        # 提取不可行个体的数据
        m1_antimicrobial_infeasible = [ind.antimicrobial_activity for ind in m1_infeasible]
        m1_hemolytic_infeasible = [ind.hemolytic_activity for ind in m1_infeasible]
        m2_antimicrobial_infeasible = [ind.antimicrobial_activity for ind in m2_infeasible]
        m2_hemolytic_infeasible = [ind.hemolytic_activity for ind in m2_infeasible]
        m3_antimicrobial_infeasible = [ind.antimicrobial_activity for ind in m3_infeasible]
        m3_hemolytic_infeasible = [ind.hemolytic_activity for ind in m3_infeasible]
        
        # 绘制可行个体
        plt.scatter(m1_antimicrobial_all, m1_hemolytic_all, c='lightblue', alpha=0.6, s=30, label='膜1可行个体')
        plt.scatter(m2_antimicrobial_all, m2_hemolytic_all, c='lightgreen', alpha=0.6, s=30, label='膜2可行个体')
        plt.scatter(m3_antimicrobial_all, m3_hemolytic_all, c='lightyellow', alpha=0.6, s=30, label='膜3可行个体')
        
        # 绘制不可行个体
        if m1_antimicrobial_infeasible:
            plt.scatter(m1_antimicrobial_infeasible, m1_hemolytic_infeasible, c='gray', alpha=0.35, s=25, marker='x', label='膜1不可行个体')
        if m2_antimicrobial_infeasible:
            plt.scatter(m2_antimicrobial_infeasible, m2_hemolytic_infeasible, c='dimgray', alpha=0.35, s=25, marker='x', label='膜2不可行个体')
        if m3_antimicrobial_infeasible:
            plt.scatter(m3_antimicrobial_infeasible, m3_hemolytic_infeasible, c='darkgray', alpha=0.35, s=25, marker='x', label='膜3不可行个体')
        
        # 绘制帕累托前沿（深色）
        # 使用真实的抗菌活性值（越小越好）
        m1_antimicrobial = [ind.antimicrobial_activity for ind in m1_pareto_front if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None]
        m1_hemolytic = [ind.hemolytic_activity for ind in m1_pareto_front if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None]
        m2_antimicrobial = [ind.antimicrobial_activity for ind in m2_pareto_front if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None]
        m2_hemolytic = [ind.hemolytic_activity for ind in m2_pareto_front if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None]
        m3_antimicrobial = [ind.antimicrobial_activity for ind in m3_pareto_front if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None]
        m3_hemolytic = [ind.hemolytic_activity for ind in m3_pareto_front if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None]
        
        plt.scatter(m1_antimicrobial, m1_hemolytic, c='darkblue', marker='s', s=60, label='膜1帕累托前沿', zorder=3)
        plt.scatter(m2_antimicrobial, m2_hemolytic, c='darkgreen', marker='^', s=60, label='膜2帕累托前沿', zorder=3)
        plt.scatter(m3_antimicrobial, m3_hemolytic, c='darkorange', marker='o', s=60, label='膜3帕累托前沿', zorder=3)
        
        eps_text = ""
        if self.structure_predictor:
            eps_val = self._get_helicity_epsilon(generation)
            eps_text = f" (ε={eps_val:.2f})"
        
        # 添加图例、标题和轴标签
        plt.xlabel('抗菌活性 (越小越好)')
        plt.ylabel('溶血活性 (越小越好)')
        plt.title(f'第 {generation} 代帕累托前沿{eps_text}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        filename = f"pareto_front_generation_{generation}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"第 {generation} 代帕累托前沿图已保存到 {filename}")
    
    def _save_population_to_csv(self):
        """
        保存最终种群的所有个体信息到CSV文件
        """
        import csv
        
        # 获取所有膜的种群
        membrane1, membrane2, membrane3 = self.membrane_system.get_membrane_populations()
        
        # 保存膜1（物理化学探索膜）所有个体信息到CSV文件
        m1_filename = "final_membrane1_population.csv"
        with open(m1_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['sequence', 'length', 'antimicrobial_activity', 'hemolytic_activity'])
            # 写入膜1所有个体数据
            for peptide in membrane1.individuals:
                writer.writerow([
                    peptide.sequence,
                    len(peptide.sequence),
                    peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0,
                    peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0
                ])
        
        # 保存膜2（结构约束膜）所有个体信息到CSV文件
        m2_filename = "final_membrane2_population.csv"
        with open(m2_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['sequence', 'length', 'antimicrobial_activity', 'hemolytic_activity'])
            # 写入膜2所有个体数据
            for peptide in membrane2.individuals:
                writer.writerow([
                    peptide.sequence,
                    len(peptide.sequence),
                    peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0,
                    peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0
                ])
        
        # 保存膜3（溶血安全膜）所有个体信息到CSV文件
        m3_filename = "final_membrane3_population.csv"
        with open(m3_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # 写入表头
            writer.writerow(['sequence', 'length', 'antimicrobial_activity', 'hemolytic_activity'])
            # 写入膜3所有个体数据
            for peptide in membrane3.individuals:
                writer.writerow([
                    peptide.sequence,
                    len(peptide.sequence),
                    peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0,
                    peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0
                ])
        
        self.logger.info(f"最终膜1（物理化学探索膜）所有个体信息已保存到 {m1_filename}")
        self.logger.info(f"最终膜2（结构约束膜）所有个体信息已保存到 {m2_filename}")
        self.logger.info(f"最终膜3（溶血安全膜）所有个体信息已保存到 {m3_filename}")
    
    def _save_elite_archive_to_fasta(self):
        """
        保存皮肤膜（精英档案）中的精英个体到fasta文件
        """
        # 获取皮肤膜精英个体
        elite_archive = self.membrane_system.get_elite_archive()
        
        # 解析肽序列和抗菌活性
        peptides = []
        for peptide in elite_archive:
            if peptide.sequence is not None:
                antimicrobial_activity = peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0
                peptides.append((antimicrobial_activity, peptide.sequence))
        
        # 按照抗菌活性从小到大排序
        peptides.sort(key=lambda x: x[0])
        
        # 生成fasta格式文件
        output_file = "elite_archive.fasta"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (activity, seq) in enumerate(peptides, 1):
                f.write(f'>elite_{i}_activity_{activity:.4f}\n')
                f.write(f'{seq}\n')
        
        self.logger.info(f"\n皮肤膜精英个体已保存到 {output_file}")
        self.logger.info(f"共保存 {len(peptides)} 个精英个体")
    
    def run(self):
        """
        运行进化算法
        """
        self.logger.info("开始进化算法...")
        self.logger.info(f"膜1（物理化学探索膜）大小: {self.membrane1_size}")
        self.logger.info(f"膜2（结构约束膜）大小: {self.membrane2_size}")
        self.logger.info(f"膜3（溶血安全膜）大小: {self.membrane3_size}")
        self.logger.info(f"最大进化代数: {self.max_generations}")
        if self.fixed_length is not None:
            self.logger.info(f"固定序列长度: {self.fixed_length}")
        else:
            self.logger.info(f"序列长度范围: {self.length_range}")
        
        # 评估初始种群
        self.logger.info("评估初始种群...")
        membrane1, membrane2, membrane3 = self.membrane_system.get_membrane_populations()
        self._evaluate_population(membrane1, 0)  # 第0代
        self._evaluate_population(membrane2, 0)  # 第0代
        self._evaluate_population(membrane3, 0)  # 第0代
        
        # 显示初始种群统计信息
        def calculate_stats(population):
            avg_anti = np.mean([ind.antimicrobial_activity for ind in population.individuals 
                               if ind.antimicrobial_activity is not None]) if any(ind.antimicrobial_activity is not None for ind in population.individuals) else 0
            avg_hemolytic = np.mean([ind.hemolytic_activity for ind in population.individuals 
                               if ind.hemolytic_activity is not None]) if any(ind.hemolytic_activity is not None for ind in population.individuals) else 0
            return avg_anti, avg_hemolytic
        
        avg_m1_anti, avg_m1_hemolytic = calculate_stats(membrane1)
        avg_m2_anti, avg_m2_hemolytic = calculate_stats(membrane2)
        avg_m3_anti, avg_m3_hemolytic = calculate_stats(membrane3)
        
        self.logger.info(f"初始膜1 - 平均抗菌活性: {avg_m1_anti:.4f}, 平均溶血活性: {avg_m1_hemolytic:.4f}")
        self.logger.info(f"初始膜2 - 平均抗菌活性: {avg_m2_anti:.4f}, 平均溶血活性: {avg_m2_hemolytic:.4f}")
        self.logger.info(f"初始膜3 - 平均抗菌活性: {avg_m3_anti:.4f}, 平均溶血活性: {avg_m3_hemolytic:.4f}")
        
        # 进化过程
        for generation in range(self.max_generations):
            self.logger.info(f"\n第 {generation + 1} 代:")
            
            # 执行膜计算系统的进化
            self.membrane_system.evolve(generation, self.max_generations, self)
            
            # 获取当前进化阶段
            current_phase = self.membrane_system.evolution_phase

            # 输出各膜算子概率分布（每10代）
            if (generation + 1) % 10 == 0:
                mem_stats = self.membrane_system.get_operator_stats()
                for role, stats in mem_stats.items():
                    # 阶段2时只打印皮肤膜算子，跳过已溶解的功能膜和外膜
                    if current_phase == 2 and role not in ('skin',):
                        continue
                    self.logger.info(f"  [膜({role}) 算子选择器] 变异概率分布:")
                    for op, prob in stats['probabilities'].items():
                        self.logger.info(f"    - {op}: {prob:.2%} (使用次数: {stats['uses'][op]})")
                # 输出通信事件日志（最近一条）
                comm_log = self.membrane_system.get_comm_log()
                if comm_log:
                    self.logger.info(f"  [最近通信事件] {comm_log[-1]}")

            # 计算并显示统计信息（每10代显示一次）
            if (generation + 1) % 10 == 0 or generation == 0:
                if current_phase == 2:
                    # ── 阶段2：打印皮肤膜统计 ──────────────────────────────
                    skin_mem = getattr(self.membrane_system, 'skin_membrane', None)
                    if skin_mem is not None and skin_mem.individuals:
                        avg_skin_anti, avg_skin_hemo = calculate_stats(skin_mem)
                        skin_pareto = skin_mem.get_pareto_front()
                        self.logger.info(
                            f"  [皮肤膜] 种群数: {len(skin_mem.individuals)}, "
                            f"Pareto前沿数: {len(skin_pareto)}, "
                            f"平均抗菌活性: {avg_skin_anti:.4f}, "
                            f"平均溶血活性: {avg_skin_hemo:.4f}"
                        )
                        # 打印皮肤膜 Pareto 前沿中最优个体（最小 MIC）
                        if skin_pareto:
                            best = min(skin_pareto, key=lambda x: (
                                x.antimicrobial_activity if x.antimicrobial_activity is not None else float('inf')
                            ))
                            self.logger.info(
                                f"  [皮肤膜最优] 序列: {best.sequence}, "
                                f"抗菌: {best.antimicrobial_activity:.4f}, "
                                f"溶血: {best.hemolytic_activity:.4f}"
                            )
                    else:
                        self.logger.info("  [皮肤膜] 尚未初始化或无有效个体")
                else:
                    # ── 阶段1：打印三个功能膜统计 ─────────────────────────
                    membrane1, membrane2, membrane3 = self.membrane_system.get_membrane_populations()
                    self._evaluate_population(membrane1, generation)
                    self._evaluate_population(membrane2, generation)
                    self._evaluate_population(membrane3, generation)

                    avg_m1_anti, avg_m1_hemolytic = calculate_stats(membrane1)
                    avg_m2_anti, avg_m2_hemolytic = calculate_stats(membrane2)
                    avg_m3_anti, avg_m3_hemolytic = calculate_stats(membrane3)

                    self.logger.info(f"  膜1 - 平均抗菌活性: {avg_m1_anti:.4f}, 平均溶血活性: {avg_m1_hemolytic:.4f}")
                    self.logger.info(f"  膜2 - 平均抗菌活性: {avg_m2_anti:.4f}, 平均溶血活性: {avg_m2_hemolytic:.4f}")
                    self.logger.info(f"  膜3 - 平均抗菌活性: {avg_m3_anti:.4f}, 平均溶血活性: {avg_m3_hemolytic:.4f}")

                # 保存帕累托前沿到txt文件
                self._save_pareto_front_to_txt(generation + 1)

                # 绘制帕累托前沿图
                self._plot_pareto_front(generation + 1)
                
        self.logger.info("\n进化算法结束!")
        
        # 保存最终结果
        self._save_pareto_front_to_txt(self.max_generations)
        self._plot_pareto_front(self.max_generations)
        
        # 显示最终帕累托前沿
        membrane1, membrane2, membrane3 = self.membrane_system.get_membrane_populations()
        self._evaluate_population(membrane1, self.max_generations)
        self._evaluate_population(membrane2, self.max_generations)
        self._evaluate_population(membrane3, self.max_generations)
        
        m1_pareto_front = membrane1.get_pareto_front()
        m2_pareto_front = membrane2.get_pareto_front()
        m3_pareto_front = membrane3.get_pareto_front()
        
        self.logger.info(f"\n最终膜1帕累托前沿 ({len(m1_pareto_front)} 个个体):")
        for i, peptide in enumerate(m1_pareto_front[:10]):  # 只显示前10个
            # 获取安全的属性值
            sequence = peptide.sequence if peptide.sequence is not None else "[序列为空]"
            length = len(sequence) if sequence != "[序列为空]" else 0
            antimicrobial_activity = peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0
            hemolytic_activity = peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0
            
            self.logger.info(f"  {i+1}. {sequence} (长度: {length}, "
                  f"抗菌活性: {antimicrobial_activity:.4f}, "
                  f"溶血活性: {hemolytic_activity:.4f})")
            
        self.logger.info(f"\n最终膜2帕累托前沿 ({len(m2_pareto_front)} 个个体):")
        for i, peptide in enumerate(m2_pareto_front[:10]):  # 只显示前10个
            # 获取安全的属性值
            sequence = peptide.sequence if peptide.sequence is not None else "[序列为空]"
            length = len(sequence) if sequence != "[序列为空]" else 0
            antimicrobial_activity = peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0
            hemolytic_activity = peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0
            
            self.logger.info(f"  {i+1}. {sequence} (长度: {length}, "
                  f"抗菌活性: {antimicrobial_activity:.4f}, "
                  f"溶血活性: {hemolytic_activity:.4f})")
        
        self.logger.info(f"\n最终膜3帕累托前沿 ({len(m3_pareto_front)} 个个体):")
        for i, peptide in enumerate(m3_pareto_front[:10]):  # 只显示前10个
            # 获取安全的属性值
            sequence = peptide.sequence if peptide.sequence is not None else "[序列为空]"
            length = len(sequence) if sequence != "[序列为空]" else 0
            antimicrobial_activity = peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0
            hemolytic_activity = peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0
            
            self.logger.info(f"  {i+1}. {sequence} (长度: {length}, "
                  f"抗菌活性: {antimicrobial_activity:.4f}, "
                  f"溶血活性: {hemolytic_activity:.4f})")
        
        # 显示全局精英档案
        elite_archive = self.membrane_system.get_elite_archive()
        self.logger.info(f"\n全局精英档案 ({len(elite_archive)} 个个体):")
        for i, peptide in enumerate(elite_archive[:15]):  # 只显示前15个
            # 获取安全的属性值
            sequence = peptide.sequence if peptide.sequence is not None else "[序列为空]"
            length = len(sequence) if sequence != "[序列为空]" else 0
            antimicrobial_activity = peptide.antimicrobial_activity if peptide.antimicrobial_activity is not None else 0.0
            hemolytic_activity = peptide.hemolytic_activity if peptide.hemolytic_activity is not None else 0.0
            
            self.logger.info(f"  {i+1}. {sequence} (长度: {length}, "
                  f"抗菌活性: {antimicrobial_activity:.4f}, "
                  f"溶血活性: {hemolytic_activity:.4f})")
        
        # 保存最终种群的所有个体信息到CSV文件
        self._save_population_to_csv()
        
        # 生成皮肤膜精英个体的fasta文件
        self._save_elite_archive_to_fasta()
