import random
from typing import List, Callable, Optional, Tuple
from peptide import Peptide
from physicochemical_constraints import PhysicochemicalConstraints

class Population:
    """种群类"""
    
    def __init__(self, size: int, is_constrained: bool = False, length_range: tuple = (5, 30), fixed_length: Optional[int] = None):
        """
        初始化种群
        
        Args:
            size: 种群大小
            is_constrained: 是否为受约束的种群（主种群）
            length_range: 肽序列长度范围
            fixed_length: 固定长度，如果指定则使用固定长度生成序列
        """
        self.size = size
        self.is_constrained = is_constrained
        self.length_range = length_range
        self.fixed_length = fixed_length
        self.individuals: List[Peptide] = []
        self.role = 'general'  # 膜角色：exploration, structure, safety, general
        self.helicity_eps = 0.3  # helicity 约束值
        self.eps_progress = 0.0  # ε 进度
        self.charge_constraint = True  # 电荷约束
        self._initialize_population()
    
    def _initialize_population(self):
        """初始化种群"""
        # 用于存储已生成的序列，避免重复
        generated_sequences = set()
        
        for _ in range(self.size):
            # 生成不重复的肽序列
            peptide = Peptide(length_range=self.length_range, fixed_length=self.fixed_length)
            attempts = 0
            max_attempts = 100
            
            # 如果是受约束的种群，确保生成的序列满足约束条件
            if self.is_constrained:
                while (attempts < max_attempts and 
                       (peptide.sequence in generated_sequences or 
                        not PhysicochemicalConstraints.check_constraints(peptide))):
                    peptide = Peptide(length_range=self.length_range, fixed_length=self.fixed_length)
                    attempts += 1
            else:
                # 不受约束的种群只需要避免重复序列
                while attempts < max_attempts and peptide.sequence in generated_sequences:
                    peptide = Peptide(length_range=self.length_range, fixed_length=self.fixed_length)
                    attempts += 1
                    
            # 如果成功生成了不重复的序列（且满足约束条件），则添加到种群中
            if peptide.sequence not in generated_sequences:
                self.individuals.append(peptide)
                generated_sequences.add(peptide.sequence)
            else:
                # 如果无法生成不重复的序列，就添加一个重复的（这是权宜之计）
                self.individuals.append(peptide)
                
    def add_individual(self, peptide: Peptide):
        """
        向种群中添加个体
        
        Args:
            peptide: 要添加的肽序列个体
        """
        self.individuals.append(peptide)
        # 如果种群大小超过设定值，移除最差的个体
        if len(self.individuals) > self.size:
            # 简单地移除最后一个（这种方法可以在环境选择中优化）
            self.individuals.pop()
    
    def get_pareto_front(self) -> List[Peptide]:
        """
        获取种群的帕累托前沿
        
        Returns:
            帕累托前沿个体列表
        """
        # 非支配排序
        fronts = self._non_dominated_sort(self.individuals)
        # 返回第一前沿（帕累托前沿）
        return fronts[0] if fronts else []
    
    def get_best_individuals(self, n: int = 5) -> List[Peptide]:
        """
        获取种群中最好的n个个体（基于帕累托前沿）
        
        Args:
            n: 要获取的个体数量
            
        Returns:
            最好的个体列表
        """
        pareto_front = self.get_pareto_front()
        # 如果帕累托前沿个体数少于n，则返回所有帕累托前沿个体
        if len(pareto_front) <= n:
            return pareto_front
            
        # 否则根据拥挤距离排序，返回前n个
        sorted_front = self._calculate_crowding_distance(pareto_front)
        return sorted_front[:n]
    
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
    
    def _calculate_crowding_distance(self, front: List[Peptide]) -> List[Peptide]:
        """
        计算前沿中个体的拥挤距离
        
        Args:
            front: 一个前沿中的个体列表
            
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
                    sorted_front1[i].crowding_distance += (activity_i_plus_1 - activity_i_minus_1) / scale1
        
        # 按溶血活性排序
        sorted_front2 = sorted(valid_front, key=lambda x: x.hemolytic_activity)
        sorted_front2[0].crowding_distance = float('inf')
        sorted_front2[-1].crowding_distance = float('inf')
        
        if len(sorted_front2) > 2:
            min_obj2 = min(hemolytic_activities)
            max_obj2 = max(hemolytic_activities)
            scale2 = max_obj2 - min_obj2 if max_obj2 != min_obj2 else 1
            
            for i in range(1, len(sorted_front2) - 1):
                # 确保所有值都不是None
                if (sorted_front2[i].hemolytic_activity is not None and 
                    sorted_front2[i-1].hemolytic_activity is not None and 
                    sorted_front2[i+1].hemolytic_activity is not None):
                    # 使用类型检查确保值不为None
                    activity_i_plus_1 = sorted_front2[i+1].hemolytic_activity
                    activity_i_minus_1 = sorted_front2[i-1].hemolytic_activity
                    sorted_front2[i].crowding_distance += (activity_i_plus_1 - activity_i_minus_1) / scale2
                    
        # 按拥挤距离降序排列
        return sorted(front, key=lambda x: x.crowding_distance, reverse=True)
    
    def select_parents(self, tournament_size: int = 3) -> Tuple[Peptide, Peptide]:
        """
        锦标赛选择父代个体
        
        Args:
            tournament_size: 锦标赛大小
            
        Returns:
            两个被选中的父代个体
        """
        def tournament_selection() -> Peptide:
            # 随机选择tournament_size个个体
            tournament_candidates = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
            
            # 根据小生境因子调整适应度（小生境计数少的个体有更高机会被选中）
            adjusted_candidates = []
            for candidate in tournament_candidates:
                # 考虑小生境因子的适应度调整
                adjusted_fitness = candidate.niche_factor if hasattr(candidate, 'niche_factor') else 1.0
                adjusted_candidates.append((candidate, adjusted_fitness))
            
            # 选择适应度最高的个体
            winner = max(adjusted_candidates, key=lambda x: x[1])[0]
            return winner
            
        # 选择两个父代
        parent1 = tournament_selection()
        parent2 = tournament_selection()
        
        # 确保两个父代不相同
        attempts = 0
        max_attempts = 10
        while parent1.sequence == parent2.sequence and attempts < max_attempts:
            parent2 = tournament_selection()
            attempts += 1
            
        return parent1, parent2