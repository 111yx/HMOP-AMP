import random
import math
import logging
from typing import List, Optional, Dict
from population import Population
from peptide import Peptide
from adaptive_operator import AdaptiveOperatorSelector

# 创建默认的logger作为备用
default_logger = logging.getLogger('MembraneSystem')
default_logger.setLevel(logging.INFO)
if not default_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    default_logger.addHandler(console_handler)

# 全局固定 HV 参考点（统一用于所有 HV 计算，保证跨运行可比性）
# f1 上界：非AMP惩罚3.9133 + 最大螺旋度惩罚 ≈ 5.0
# f2 上界：HC50极端溶血 -log10(0.001) ≈ 3.0，留安全余量
HV_REF_POINT = (5.0, 3.0)


# ---------------------------------------------------------------------------
# 辅助：超体积指标（二维，最小化方向，参考点为右上角）
# ---------------------------------------------------------------------------
def _hypervolume_2d(points: List[tuple], ref: tuple) -> float:
    """
    计算二维超体积指标（适用于最小化问题）。
    points: [(f1, f2), ...]  ref: (r1, r2)  r1 >= max(f1), r2 >= max(f2)
    """
    if not points:
        return 0.0
    # 过滤支配参考点的解（即 f1 >= r1 或 f2 >= r2 的点无贡献）
    pts = [(f1, f2) for f1, f2 in points if f1 < ref[0] and f2 < ref[1]]
    if not pts:
        return 0.0
    # 按 f1 升序排列
    pts.sort(key=lambda x: x[0])
    hv = 0.0
    prev_f1 = pts[0][0]
    min_f2 = pts[0][1]
    for i in range(1, len(pts)):
        # 宽度 × 高度
        hv += (pts[i][0] - prev_f1) * (ref[1] - min_f2)
        min_f2 = min(min_f2, pts[i][1])
        prev_f1 = pts[i][0]
    # 最后一段
    hv += (ref[0] - prev_f1) * (ref[1] - min_f2)
    return hv


def _get_pareto_points(individuals: List[Peptide]) -> List[tuple]:
    """提取非 None 目标值的 (f1, f2) 点列表。"""
    return [
        (ind.antimicrobial_activity, ind.hemolytic_activity)
        for ind in individuals
        if ind.antimicrobial_activity is not None and ind.hemolytic_activity is not None
    ]


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class MembraneSystem:
    """
    层次化 P 系统膜计算类。

    结构：
        皮膜（skin）= 全局精英档案（不进化，只收集非支配解）
        ├─ 膜1（exploration）：物理化学探索膜，约束放宽，高变异/交叉
        ├─ 膜2（structure）  ：结构约束膜，helicity ε 动态收紧（0.3→0.6）
        └─ 膜3（safety）     ：溶血安全膜，偏向降低溶血毒性

    改进点（与旧版的主要差异）：
        1. 每个膜拥有独立的 AdaptiveOperatorSelector（膜级 UCB 奖励）。
        2. 膜间通信改为事件驱动：基于超体积停滞度量触发迁入；
           迁出由"rank=1 且 crowding_distance ≥ 阈值"条件驱动。
        3. 皮膜精英档案裁剪使用真正的 Pareto 非支配排序 + 拥挤距离。
        4. 通信时使用"替换最差个体"而非直接追加，避免种群膨胀。
        5. 各膜独立维护 helicity epsilon（膜1固定0.0，膜2动态，膜3固定0.3）。
    """

    # 消融模式常量
    ABLATION_FULL            = 'full'           # 标准策略（默认）：去重迁出+自适应迁入+内容感知横向
    ABLATION_NO_COMM         = 'no_comm'        # 禁用所有膜间通信（迁入/迁出/横向）
    ABLATION_FIXED_COMM      = 'fixed_comm'     # 每 fixed_comm_interval 代强制迁移（替代事件驱动）
    ABLATION_NO_SKIN         = 'no_skin'        # 禁用皮肤膜精英档案（三膜结果直接合并）
    ABLATION_SINGLE_MEMBRANE = 'single_membrane' # 只用单膜（膜2），相当于普通 NSGA-II
    ABLATION_EVENT_DRIVEN    = 'event_driven'   # 旧版事件驱动通信（仅停滞检测敏感性分析内部使用）

    def __init__(self,
                 membrane1_size: int = 50,
                 membrane2_size: int = 50,
                 membrane3_size: int = 50,
                 length_range: tuple = (5, 30),
                 fixed_length: Optional[int] = None,
                 helicity_eps_start: float = 0.3,
                 helicity_eps_end: float = 0.6,
                 skin_helicity_eps: float = 0.4,
                 stagnation_threshold: float = 0.01,
                 stagnation_window: int = 10,
                 ablation_mode: str = 'full',
                 fixed_comm_interval: int = 5,
                 base_migration_size: int = 5,
                 base_p_out: float = 0.3,
                 interval_min: int = 3,
                 interval_max: int = 10,
                 logger: Optional[logging.Logger] = None):
        """
        初始化膜计算系统。

        Args:
            membrane1_size: 膜1（物理化学探索膜）种群大小
            membrane2_size: 膜2（结构约束膜）种群大小
            membrane3_size: 膜3（溶血安全膜）种群大小
            length_range:   肽序列长度范围
            fixed_length:   固定长度（若指定）
            helicity_eps_start: 膜2 螺旋度约束初始值
            helicity_eps_end:   膜2 螺旋度约束终止值
            skin_helicity_eps:  皮肤膜螺旋度约束（第三阶段使用）
            stagnation_threshold: 停滞检测 HV 变化阈值（小于此值认为停滞）
            stagnation_window:    停滞检测时间窗口（代数）
            ablation_mode:  消融模式，可选：
                'full'             - 标准策略（默认）：去重迁出+自适应迁入+内容感知横向
                'event_driven'     - 旧版事件驱动通信（仅停滞检测敏感性分析内部使用）
                'no_comm'          - 禁用所有膜间通信
                'fixed_comm'       - 每 fixed_comm_interval 代强制迁移
                'no_skin'          - 禁用皮肤膜精英档案
                'single_membrane'  - 只用单膜（膜2）
            fixed_comm_interval: fixed_comm 模式下强制迁移间隔（代数），默认 5
            base_migration_size: 基础迁移个体数，默认 5（动态范围 3~15）
            base_p_out: 基础迁出概率，默认 0.3
            interval_min: 迁入最小间隔（代数），默认 3
            interval_max: 迁入最大间隔（代数），默认 10
            logger: 日志记录器
        """
        self.helicity_eps_start = helicity_eps_start
        self.helicity_eps_end = helicity_eps_end
        self.skin_helicity_eps = skin_helicity_eps
        self.ablation_mode = ablation_mode
        self.fixed_comm_interval = fixed_comm_interval
        self.base_migration_size = base_migration_size
        self.base_p_out = base_p_out
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.logger = logger if logger is not None else default_logger

        # ── 初始化三个功能膜 ──────────────────────────────────────────────
        self.membrane1 = Population(
            size=membrane1_size,
            is_constrained=False,
            length_range=length_range,
            fixed_length=fixed_length
        )
        self.membrane1.role = 'exploration'
        self.membrane1.helicity_eps = 0.0          # 完全放开
        self.membrane1.charge_constraint = False
        self.membrane1.status = 'active'  # 膜状态：active, inactive, dissolved

        self.membrane2 = Population(
            size=membrane2_size,
            is_constrained=True,
            length_range=length_range,
            fixed_length=fixed_length
        )
        self.membrane2.role = 'structure'
        self.membrane2.helicity_eps = helicity_eps_start
        self.membrane2.eps_progress = 0.0
        self.membrane2.status = 'active'

        self.membrane3 = Population(
            size=membrane3_size,
            is_constrained=True,
            length_range=length_range,
            fixed_length=fixed_length
        )
        self.membrane3.role = 'safety'
        self.membrane3.helicity_eps = 0.3           # 适度固定
        self.membrane3.status = 'active'

        # ── 初始化三个外膜（精英档案膜）──────────────────────────────────
        self.outer_membrane1 = Population(
            size=membrane1_size * 1,  # 外膜容量为对应功能膜的1倍
            is_constrained=False,
            length_range=length_range,
            fixed_length=fixed_length
        )
        self.outer_membrane1.role = 'outer_exploration'
        self.outer_membrane1.max_capacity = int(membrane1_size * 1.5)  # 外膜最大容量
        self.outer_membrane1.status = 'active'

        self.outer_membrane2 = Population(
            size=membrane2_size * 1,
            is_constrained=False,
            length_range=length_range,
            fixed_length=fixed_length
        )
        self.outer_membrane2.role = 'outer_structure'
        self.outer_membrane2.max_capacity = int(membrane2_size * 1.5)
        self.outer_membrane2.status = 'active'

        self.outer_membrane3 = Population(
            size=membrane3_size * 1,
            is_constrained=False,
            length_range=length_range,
            fixed_length=fixed_length
        )
        self.outer_membrane3.role = 'outer_safety'
        self.outer_membrane3.max_capacity = int(membrane3_size * 1.5)
        self.outer_membrane3.status = 'active'

        # ── 膜级独立算子选择器 ────────────────────────────────────────────
        # 算子列表在外部（evolutionary_algorithm）确定后传入，这里先用占位；
        # evolutionary_algorithm 在 __init__ 完成后会调用 set_operator_lists()。
        self._pending_operators: Optional[List[str]] = None

        # 算子选择器字典，key 为膜角色
        self.operator_selectors: Dict[str, AdaptiveOperatorSelector] = {}

        # ── 膜间通信参数 ───────────────────────────────────────────────────
        self.stagnation_window = stagnation_window        # 停滞检测窗口（代数）
        self.stagnation_threshold = stagnation_threshold  # HV 变化量阈值
        self.crowding_out_thresh = 0.1   # 迁出拥挤距离阈值
        self.base_t_min = 2              # 基础个体最小年龄
        # 动态参数
        self.migration_size = self.base_migration_size
        self.p_out = self.base_p_out
        self.t_min = self.base_t_min

        # ── 通信事件统计（用于敏感性分析）────────────────────────────────
        self.comm_migrate_in_count = 0   # 迁入事件次数（停滞触发）
        self.comm_migrate_out_count = 0  # 迁出事件次数（精英迁出到皮膜）
        self.comm_lateral_count = 0      # 横向通信次数
        self.stagnation_events: Dict[str, int] = {
            'exploration': 0, 'structure': 0, 'safety': 0, 'skin': 0
        }  # 各膜停滞事件累计次数

        # ── 超体积历史（用于停滞检测）────────────────────────────────────
        self._hv_history: Dict[str, List[float]] = {
            'exploration': [],
            'structure': [],
            'safety': [],
            'skin': []
        }

        # ── 全局精英档案（皮膜）──────────────────────────────────────────
        self.elite_archive: List[Peptide] = []
        self.base_max_archive_size = 100  # 基础皮膜容量上限
        self.max_archive_size = self.base_max_archive_size  # 动态调整的容量上限

        # ── 通信日志（用于调试）──────────────────────────────────────────
        self._comm_log: List[str] = []

        # ── 两阶段进化相关属性 ──────────────────────────────────────────
        self.evolution_phase = 1  # 1: 功能膜与外膜协同进化, 2: 皮肤膜最终进化
        self._dissolution_triggered = False  # 膜溶解事件是否已触发
        self._dissolution_generation = 0  # 膜溶解事件触发的代数
        self.total_generations = 0  # 总进化代数
        self.phase_switch_threshold = 0.7  # 进化代数达到70%时触发膜溶解
        self.pareto_stagnation_count = 0  # 帕累托前沿停滞计数
        self.max_pareto_stagnation = 5  # 最大帕累托停滞代数
        # Phase 1 末期 archive 快照（Phase 切换时写入；无切换时为 None）
        self.phase1_final_archive_hv   = None
        self.phase1_final_archive_size = 0

    # ======================================================================
    # 公共接口：设置算子列表（由 EvolutionaryAlgorithm 调用）
    # ======================================================================
    def set_operator_lists(self, available_ops: List[str]):
        """
        为每个膜创建独立的算子选择器。
        膜1（exploration）会为 ESM 算子设置更低的初始分，
        膜3（safety）会额外压低高变异算子。
        """
        self._pending_operators = available_ops
        for role in ['exploration', 'structure', 'safety', 'outer_exploration', 'outer_structure', 'outer_safety']:
            self.operator_selectors[role] = AdaptiveOperatorSelector(
                operators=available_ops,
                alpha=0.1,
                min_prob=0.05
            )

    def _get_selector(self, membrane: Population) -> AdaptiveOperatorSelector:
        """根据膜角色返回对应的算子选择器，若尚未初始化则返回默认实例。"""
        role = getattr(membrane, 'role', 'structure')
        if role not in self.operator_selectors:
            # 兜底：未被外部初始化时创建临时选择器
            self.operator_selectors[role] = AdaptiveOperatorSelector(
                operators=['random_replace'], alpha=0.1, min_prob=0.05
            )
        return self.operator_selectors[role]

    # ======================================================================
    # 主进化接口
    # ======================================================================
    def evolve(self, generation: int, max_generations: int, algorithm):
        """
        执行一代进化。

        Args:
            generation:      当前进化代数（从 0 开始）
            max_generations: 最大进化代数
            algorithm:       EvolutionaryAlgorithm 实例
        """
        self.total_generations = max_generations
        
        # 检查是否需要触发膜溶解事件（Phase 1 → Phase 2）
        self._check_phase_switch(generation, max_generations)
        
        if self.evolution_phase == 1:
            # 阶段1：功能膜与外膜协同进化
            self._phase1_evolve(generation, max_generations, algorithm)
        else:  # evolution_phase == 2
            # 阶段2：皮肤膜最终进化
            self._phase2_evolve(generation, max_generations, algorithm)

    def _check_phase_switch(self, generation: int, max_generations: int):
        """
        检查是否需要触发膜溶解事件，从阶段1切换到阶段2。
        """
        if self.evolution_phase == 1 and not self._dissolution_triggered:
            # 检查是否满足阶段切换条件
            progress = generation / (max_generations - 1)
            
            # 记录当前进度
            self.logger.info("[Phase Check] 第 %d 代，进化进度: %.2f, 阈值: %.2f", generation, progress, self.phase_switch_threshold)
            
            # 条件1：进化代数达到总代数的70%
            if progress >= self.phase_switch_threshold:
                self.logger.info("[Phase Check] 满足条件1：进化进度达到阈值，触发膜溶解事件")
                self._trigger_dissolution(generation)
                return
            
            # 条件2：所有功能膜都停滞
            all_stagnating = (
                self._is_stagnating(self.membrane1) and
                self._is_stagnating(self.membrane2) and
                self._is_stagnating(self.membrane3)
            )
            if all_stagnating:
                self.logger.info("[Phase Check] 满足条件2：所有功能膜都停滞，触发膜溶解事件")
                self._trigger_dissolution(generation)
                return
            
            # 条件3：帕累托前沿连续5代无新解
            # 这里简化处理，实际实现需要比较前后代的帕累托前沿
            # 暂时只使用代数条件

    def _trigger_dissolution(self, generation: int):
        """
        触发膜溶解事件：溶解功能膜，外膜精英汇入皮肤膜，切换到阶段2。
        该事件在满足条件时由 _check_phase_switch 调用，
        溶解操作在下一代的 _phase2_evolve 首次进入时执行。
        """
        self._dissolution_triggered = True
        self._dissolution_generation = generation
        self.logger.info("[Dissolution] 膜溶解事件已标记，将在下一代进入阶段2时执行 (Generation: %d)", generation)

        # ── Phase 1 末期 archive HV 快照（用于消融实验公平对比） ─────────
        if self.elite_archive:
            pts = [(p.antimicrobial_activity, p.hemolytic_activity)
                   for p in self.elite_archive
                   if p.antimicrobial_activity is not None and p.hemolytic_activity is not None]
            self.phase1_final_archive_hv = _hypervolume_2d(pts, HV_REF_POINT)
        else:
            self.phase1_final_archive_hv = None
        self.phase1_final_archive_size = len(self.elite_archive)

    def _phase1_evolve(self, generation: int, max_generations: int, algorithm):
        """
        第一阶段：功能膜与外膜协同进化。
        single_membrane 消融模式下仅进化膜2，跳过所有膜间通信。
        """
        # 1. 更新膜2的 helicity epsilon（线性插值）
        if max_generations > 1:
            progress = generation / (max_generations - 1)
        else:
            progress = 1.0
        self.membrane2.helicity_eps = (
            self.helicity_eps_start
            + (self.helicity_eps_end - self.helicity_eps_start) * progress
        )

        # ── 消融：single_membrane → 只进化膜2，其余全部跳过 ─────────────
        if self.ablation_mode == self.ABLATION_SINGLE_MEMBRANE:
            self._adjust_communication_params(generation, max_generations)
            self.membrane2.eps_progress = progress
            self._evolve_membrane(self.membrane2, generation, max_generations, algorithm)
            self._record_hv(self.membrane2)
            return
        
        # 2. 动态调整通信参数
        self._adjust_communication_params(generation, max_generations)
        self.membrane2.eps_progress = progress

        # 3. 各功能膜独立进化
        if self.membrane1.status == 'active':
            self._evolve_membrane(self.membrane1, generation, max_generations, algorithm)
        if self.membrane2.status == 'active':
            self._evolve_membrane(self.membrane2, generation, max_generations, algorithm)
        if self.membrane3.status == 'active':
            self._evolve_membrane(self.membrane3, generation, max_generations, algorithm)

        # 4. 记录各膜超体积历史
        self._record_hv(self.membrane1)
        self._record_hv(self.membrane2)
        self._record_hv(self.membrane3)

        # 5. 功能膜向对应外膜迁移精英个体
        self._migrate_to_outer_membrane(self.membrane1, self.outer_membrane1)
        self._migrate_to_outer_membrane(self.membrane2, self.outer_membrane2)
        self._migrate_to_outer_membrane(self.membrane3, self.outer_membrane3)

        # 6. 事件驱动膜间通信
        self._communicate_event_driven(generation)

        # 7. 更新全局精英档案（皮膜）
        self._update_elite_archive()

    def _phase2_evolve(self, generation: int, max_generations: int, algorithm):
        """
        阶段2：皮肤膜最终进化。
        首次进入时执行膜溶解事件（功能膜溶解 + 外膜精英汇入皮肤膜），
        后续代数仅执行皮肤膜进化 + 停滞扰动。
        """
        # ── 一次性初始化：膜溶解事件 ─────────────────────────────
        if not self._dissolution_triggered:
            # 正常流程不应走到这里（_check_phase_switch 先触发），
            # 但作为安全兜底，确保功能膜已溶解
            pass

        # 首次进入阶段2时执行膜溶解 + 外膜汇入
        if not hasattr(self, '_skin_initialized') or not self._skin_initialized:
            self._execute_dissolution()
            self._skin_initialized = True

        # ── 皮肤膜进化 ──────────────────────────────────────────
        # 1. 动态调整通信参数
        self._adjust_communication_params(generation, max_generations)

        # 2. 皮肤膜进化
        self._evolve_membrane(self.skin_membrane, generation, max_generations, algorithm)

        # 3. 记录皮肤膜超体积历史（用于停滞检测）
        self._record_hv(self.skin_membrane)

        # 4. 皮肤膜停滞扰动机制
        #    与阶段1的迁入逻辑对称：停滞时注入精英/随机个体，打破局部最优
        if self._is_stagnating(self.skin_membrane):
            if len(self.elite_archive) > 0:
                # 优先从全局精英档案注入
                self._migrate_in(self.skin_membrane)
                self.comm_migrate_in_count += 1
                self.stagnation_events['skin'] = self.stagnation_events.get('skin', 0) + 1
                self._comm_log.append(
                    f"[gen={generation}] 皮肤膜停滞，从全局精英档案注入 "
                    f"{self.migration_size} 个个体"
                )
            else:
                # 档案为空时注入随机个体
                n_inject = max(3, self.migration_size)
                injected = 0
                # 替换最差的若干个体
                sorted_inds = sorted(
                    self.skin_membrane.individuals,
                    key=lambda x: getattr(x, 'domination_count', 0),
                    reverse=True
                )
                replace_seqs = {ind.sequence for ind in sorted_inds[:n_inject]}
                new_pop = [
                    ind for ind in self.skin_membrane.individuals
                    if ind.sequence not in replace_seqs
                ]
                for _ in range(n_inject):
                    rand_ind = self._generate_random_individual(self.skin_membrane)
                    rand_ind._age = 0
                    new_pop.append(rand_ind)
                    injected += 1
                self.skin_membrane.individuals = new_pop[:self.skin_membrane.size]
                self._comm_log.append(
                    f"[gen={generation}] 皮肤膜停滞（档案为空），注入 {injected} 个随机个体"
                )
            self.logger.info(
                "[Phase 2] 皮肤膜停滞，已执行扰动注入 (gen=%d)", generation
            )

        # 5. 更新全局精英档案
        self._update_elite_archive()

    def _execute_dissolution(self):
        """
        执行膜溶解事件（一次性操作）：
        1. 溶解三个功能膜
        2. 将外膜精英汇入皮肤膜
        3. 切换到阶段2
        """
        # 1. 溶解功能膜
        self.membrane1.status = 'dissolved'
        self.membrane2.status = 'dissolved'
        self.membrane3.status = 'dissolved'
        if self.logger:
            self.logger.info("[Dissolution] 功能膜已溶解")
        else:
            print("[Dissolution] 功能膜已溶解")
        
        # 2. 外膜精英汇入皮肤膜
        self._merge_outer_membranes_to_skin()
        if self.logger:
            self.logger.info("[Dissolution] 外膜精英已汇入皮肤膜")
        else:
            print("[Dissolution] 外膜精英已汇入皮肤膜")
        
        # 3. 切换到阶段2
        self.evolution_phase = 2
        if self.logger:
            self.logger.info("[Phase Switch] 进入阶段2：皮肤膜最终进化")
        else:
            print("[Phase Switch] 进入阶段2：皮肤膜最终进化")

    def _migrate_to_outer_membrane(self, source: Population, target: Population):
        """
        从功能膜向对应外膜迁移满足条件的精英个体。
        """
        if source.status != 'active' or target.status != 'active':
            return
        
        # 获取源膜的帕累托前沿
        pareto_front = source.get_pareto_front()
        
        # 筛选满足目标膜条件的个体
        eligible_individuals = []
        for ind in pareto_front:
            # 检查个体是否满足外膜的条件
            if self._is_eligible_for_outer_membrane(ind, target):
                eligible_individuals.append(ind)
        
        # 迁移符合条件的个体到外膜
        for ind in eligible_individuals:
            # 检查外膜是否已满
            if len(target.individuals) >= getattr(target, 'max_capacity', 100):
                break
            
            # 检查外膜中是否已有相同序列的个体
            if not self._is_similar_to_population(ind, target):
                cloned = ind.clone()
                cloned._age = 0
                target.individuals.append(cloned)
        
        # 对外膜进行裁剪，保持最大容量
        self._trim_outer_membrane(target)

    def _is_eligible_for_outer_membrane(self, individual: Peptide, outer_membrane: Population) -> bool:
        """
        检查个体是否满足外膜的条件。
        """
        # 基本条件：目标值非空
        if individual.antimicrobial_activity is None or individual.hemolytic_activity is None:
            return False
        
        # 根据外膜类型设置不同的条件
        role = getattr(outer_membrane, 'role', '')
        
        if role == 'outer_exploration':
            # 外膜1：探索性强的个体
            return True  # 所有帕累托前沿个体都可迁移
        elif role == 'outer_structure':
            # 外膜2：结构良好的个体
            # 可以添加螺旋度等结构相关的条件
            return True
        elif role == 'outer_safety':
            # 外膜3：低毒性的个体
            # 优先迁移溶血活性低的个体
            return individual.hemolytic_activity < 2.0  # 示例阈值
        
        return True

    def _trim_outer_membrane(self, outer_membrane: Population):
        """
        对外膜进行裁剪，保持最大容量。
        使用非支配排序 + 拥挤距离的 Pareto 保持策略，
        并根据外膜角色应用差异化筛选逻辑。
        """
        max_capacity = getattr(outer_membrane, 'max_capacity', 100)
        if len(outer_membrane.individuals) <= max_capacity:
            return

        role = getattr(outer_membrane, 'role', 'unknown')

        # 过滤掉目标值缺失的无效个体
        valid = [
            ind for ind in outer_membrane.individuals
            if ind.antimicrobial_activity is not None
            and ind.hemolytic_activity is not None
        ]
        invalid = [
            ind for ind in outer_membrane.individuals
            if ind.antimicrobial_activity is None
            or ind.hemolytic_activity is None
        ]

        if role == 'outer_exploration':
            # ── 外膜1（exploration）：非支配排序 + 拥挤距离 + 长度多样性 ──
            # 宽松策略：所有有效个体参与排序，辅以长度多样性保序
            trimmed = self._pareto_trim_with_diversity(valid, max_capacity)
        elif role == 'outer_structure':
            # ── 外膜2（structure）：feasible 优先 + 非支配排序 ──
            # feasible 个体用非支配排序 + 拥挤距离，
            # infeasible 个体按 constraint_violation 升序排，feasible 填完后补充
            trimmed = self._constraint_aware_trim(valid, max_capacity)
        elif role == 'outer_safety':
            # ── 外膜3（safety）：溶血阈值过滤 + 非支配排序 ──
            # 先过滤 hemo ≤ 动态阈值（Q25），再用非支配排序保前沿；
            # 若过滤后不足，逐步放宽阈值补充
            trimmed = self._safety_aware_trim(valid, max_capacity)
        else:
            # 默认：标准非支配排序 + 拥挤距离
            trimmed = self._pareto_trim(valid, max_capacity)

        # 无效个体不主动保留，仅在有剩余容量时补充
        remaining_slots = max_capacity - len(trimmed)
        if remaining_slots > 0 and invalid:
            trimmed.extend(invalid[:remaining_slots])

        outer_membrane.individuals = trimmed

    # ------------------------------------------------------------------
    #  外膜裁剪辅助方法
    # ------------------------------------------------------------------

    def _pareto_trim(self, individuals: List[Peptide], capacity: int) -> List[Peptide]:
        """标准非支配排序 + 拥挤距离裁剪（与皮膜档案裁剪逻辑一致）。"""
        if not individuals:
            return individuals
        fronts = self._non_dominated_sort_archive(individuals)
        result = []
        for front in fronts:
            if len(result) + len(front) <= capacity:
                result.extend(front)
            else:
                remaining = capacity - len(result)
                front_sorted = self._crowding_sort(front)
                result.extend(front_sorted[:remaining])
                break
        return result

    def _pareto_trim_with_diversity(self, individuals: List[Peptide], capacity: int) -> List[Peptide]:
        """
        非支配排序 + 拥挤距离裁剪，同一前沿内额外考虑序列长度多样性。
        用于外膜1（exploration），避免长度同质化。
        """
        if not individuals:
            return individuals

        fronts = self._non_dominated_sort_archive(individuals)
        result = []
        for front in fronts:
            if len(result) + len(front) <= capacity:
                # 前沿全部保留前，按长度多样性重排
                result.extend(self._length_diverse_sort(front))
            else:
                remaining = capacity - len(result)
                # 拥挤距离排序
                front_by_cd = self._crowding_sort(front)
                # 拥挤距离与长度多样性各占 50% 权重融合排序
                diverse_sorted = self._merge_cd_and_length(front_by_cd, remaining)
                result.extend(diverse_sorted[:remaining])
                break
        return result

    def _constraint_aware_trim(self, individuals: List[Peptide], capacity: int) -> List[Peptide]:
        """
        feasible 优先 + 非支配排序裁剪。用于外膜2（structure）。
        先填 feasible 个体（非支配排序+拥挤距离），
        有余量再用 infeasible 个体（按 constraint_violation 升序）补充。
        """
        feasible = [ind for ind in individuals if getattr(ind, 'is_feasible', True)]
        infeasible = [ind for ind in individuals if not getattr(ind, 'is_feasible', True)]

        # feasible 用非支配排序填充
        result = self._pareto_trim(feasible, capacity)

        # 有余量时用 infeasible 补充（违反度小的优先）
        remaining = capacity - len(result)
        if remaining > 0 and infeasible:
            infeasible.sort(key=lambda x: getattr(x, 'constraint_violation', float('inf')))
            result.extend(infeasible[:remaining])

        return result

    def _safety_aware_trim(self, individuals: List[Peptide], capacity: int) -> List[Peptide]:
        """
        溶血阈值过滤 + 非支配排序裁剪。用于外膜3（safety）。
        先保留 hemo ≤ Q25 的个体做非支配排序；
        若不足则逐步放宽到 Q50、Q75，最后全量补充。
        """
        if not individuals:
            return individuals

        hemo_vals = [ind.hemolytic_activity for ind in individuals]
        hemo_vals_sorted = sorted(hemo_vals)

        # 计算分位数
        def percentile(data, p):
            k = (len(data) - 1) * p / 100.0
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])

        q25 = percentile(hemo_vals_sorted, 25) if len(hemo_vals_sorted) >= 4 else hemo_vals_sorted[0]
        q50 = percentile(hemo_vals_sorted, 50) if len(hemo_vals_sorted) >= 2 else hemo_vals_sorted[0]

        # 第一轮：hemo ≤ Q25 的个体
        low_hemo = [ind for ind in individuals if ind.hemolytic_activity <= q25]
        result = self._pareto_trim(low_hemo, capacity)

        # 第二轮：Q25 < hemo ≤ Q50 补充
        if len(result) < capacity:
            mid_hemo = [ind for ind in individuals if q25 < ind.hemolytic_activity <= q50]
            mid_sorted = self._pareto_trim(mid_hemo, capacity - len(result))
            result.extend(mid_sorted)

        # 第三轮：全量补充
        if len(result) < capacity:
            remaining_inds = [ind for ind in individuals if ind not in result]
            remaining_sorted = self._pareto_trim(remaining_inds, capacity - len(result))
            result.extend(remaining_sorted)

        return result

    def _length_diverse_sort(self, front: List[Peptide]) -> List[Peptide]:
        """
        在同一 Pareto 前沿内，按序列长度多样性重排。
        贪心策略：每次选与已选个体长度差异最大的个体。
        """
        if len(front) <= 2:
            return front

        sorted_by_len = sorted(front, key=lambda x: len(x.sequence))
        result = [sorted_by_len[0]]  # 从最短序列开始
        remaining = list(sorted_by_len[1:])

        while remaining:
            last_len = len(result[-1].sequence)
            # 选与最后一个选入个体长度差最大的
            remaining.sort(key=lambda x: -abs(len(x.sequence) - last_len))
            result.append(remaining.pop(0))

        return result

    def _merge_cd_and_length(self, front_by_cd: List[Peptide], n_select: int) -> List[Peptide]:
        """
        融合拥挤距离排序与长度多样性排序，各占 50% 权重。
        返回融合排序后的前 n_select 个个体。
        """
        if len(front_by_cd) <= n_select:
            return front_by_cd

        # 拥挤距离排名
        cd_rank = {id(ind): i for i, ind in enumerate(front_by_cd)}

        # 长度多样性排名
        len_sorted = self._length_diverse_sort(front_by_cd)
        len_rank = {id(ind): i for i, ind in enumerate(len_sorted)}

        # 综合排名（越小越优）
        total = len(front_by_cd)
        combined = sorted(
            front_by_cd,
            key=lambda x: (cd_rank[id(x)] + len_rank[id(x)]) / 2.0
        )
        return combined[:n_select]

    def _merge_outer_membranes_to_skin(self):
        """
        将外膜的精英个体汇入皮肤膜。
        """
        # 收集所有外膜的精英个体
        all_elite = []
        
        for outer_membrane in [self.outer_membrane1, self.outer_membrane2, self.outer_membrane3]:
            if outer_membrane.status == 'active':
                all_elite.extend(outer_membrane.individuals)
        
        # 去重
        seen = set()
        unique_elite = []
        for ind in all_elite:
            if ind.sequence not in seen:
                seen.add(ind.sequence)
                unique_elite.append(ind)
        
        # 初始化皮肤膜
        self._initialize_skin_membrane()
        
        # 将精英个体添加到皮肤膜
        self.skin_membrane.individuals = unique_elite
        print(f"[Dissolution] 共汇入 {len(unique_elite)} 个精英个体到皮肤膜")

    def _initialize_skin_membrane(self):
        """
        初始化皮肤膜。
        """
        if not hasattr(self, 'skin_membrane'):
            # 计算皮肤膜的种群大小（各外膜大小之和的1.5倍）
            total_size = len(self.outer_membrane1.individuals) + \
                         len(self.outer_membrane2.individuals) + \
                         len(self.outer_membrane3.individuals)
            skin_size = int(total_size * 1.5)
            
            self.skin_membrane = Population(
                size=skin_size,
                is_constrained=True,
                length_range=self.membrane1.length_range,
                fixed_length=self.membrane1.fixed_length
            )
            self.skin_membrane.role = 'skin'
            self.skin_membrane.helicity_eps = self.skin_helicity_eps
            self.skin_membrane.status = 'active'
            
            # 为皮肤膜创建算子选择器
            if 'skin' not in self.operator_selectors:
                available_ops = list(self.operator_selectors.values())[0].operators
                self.operator_selectors['skin'] = AdaptiveOperatorSelector(
                    operators=available_ops,
                    alpha=0.1,
                    min_prob=0.05
                )
        else:
            # 确保皮肤膜状态为活跃
            self.skin_membrane.status = 'active'

    # ======================================================================
    # 膜内进化
    # ======================================================================
    def _evolve_membrane(self, membrane: Population, generation: int,
                         max_generations: int, algorithm):
        """执行单个膜的一代进化（评估→交叉变异→NSGA-II 环境选择）。"""
        # 临时将 algorithm 的 helicity_eps_fixed 设为本膜的 helicity_eps，
        # 使得 _evaluate_population 中的约束违反计算使用各膜独立的 ε 值。
        _prev_fixed = algorithm.helicity_eps_fixed
        algorithm.helicity_eps_fixed = membrane.helicity_eps

        # 评估
        algorithm._evaluate_population(membrane, generation)

        # 恢复
        algorithm.helicity_eps_fixed = _prev_fixed

        # 生成子代（使用本膜独立算子选择器）
        new_individuals = self._perform_crossover_and_mutation(
            membrane=membrane,
            algorithm=algorithm,
            crossover_rate=self._get_crossover_rate(membrane.role),
            mutation_rate=self._get_mutation_rate(membrane.role)
        )

        if not new_individuals:
            return

        # 评估子代
        for ind in new_individuals:
            ind.antimicrobial_activity = None
            ind.hemolytic_activity = None
            ind.base_antimicrobial_activity = None
        algorithm._evaluate_population_list(new_individuals, generation)

        # 更新算子奖励（膜级独立）
        self._update_operator_rewards(
            new_individuals=new_individuals,
            combined=membrane.individuals + new_individuals,
            membrane=membrane,
            algorithm=algorithm
        )

        # 年龄递增
        for ind in membrane.individuals:
            if not hasattr(ind, '_age'):
                ind._age = 0
            ind._age += 1

        # NSGA-II 环境选择（合并父代+子代，选优）
        combined = membrane.individuals + new_individuals
        selected = algorithm._environmental_selection(combined, membrane.size, membrane.role)
        membrane.individuals = selected

    # ======================================================================
    # 交叉与变异（使用膜级算子选择器）
    # ======================================================================
    def _perform_crossover_and_mutation(self, membrane: Population, algorithm,
                                        crossover_rate: float,
                                        mutation_rate: float) -> List[Peptide]:
        """对膜种群执行交叉和变异，返回新生成的个体列表。"""
        from physicochemical_constraints import PhysicochemicalConstraints

        selector = self._get_selector(membrane)
        new_individuals = []
        target_size = membrane.size

        while len(new_individuals) < target_size:
            parent1, parent2 = membrane.select_parents()

            if random.random() < crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.clone(), parent2.clone()

            # 使用本膜算子选择器
            op1 = selector.select_operator()
            op2 = selector.select_operator()
            child1 = child1.mutate(mutation_rate, operator_type=op1)
            child2 = child2.mutate(mutation_rate, operator_type=op2)

            # 记录使用的算子
            child1.last_operator = op1
            child2.last_operator = op2

            # 受约束膜：重试直到满足物理化学约束
            if membrane.is_constrained:
                max_attempts = 10
                for attempts in range(max_attempts):
                    if PhysicochemicalConstraints.check_constraints(child1):
                        break
                    child1 = child1.mutate(mutation_rate, operator_type=op1)
                    child1.last_operator = op1
                for attempts in range(max_attempts):
                    if PhysicochemicalConstraints.check_constraints(child2):
                        break
                    child2 = child2.mutate(mutation_rate, operator_type=op2)
                    child2.last_operator = op2

            new_individuals.append(child1)
            if len(new_individuals) < target_size:
                new_individuals.append(child2)

        # 受约束膜过滤
        if membrane.is_constrained:
            from physicochemical_constraints import PhysicochemicalConstraints
            new_individuals = [
                ind for ind in new_individuals
                if PhysicochemicalConstraints.check_constraints(ind)
            ]

        return new_individuals[:target_size]

    # ======================================================================
    # 膜级算子奖励更新
    # ======================================================================
    def _update_operator_rewards(self, new_individuals: List[Peptide],
                                  combined: List[Peptide],
                                  membrane: Population,
                                  algorithm):
        """根据子代是否进入 Pareto 前沿或存活来更新本膜算子选择器。"""
        if not new_individuals:
            return
        selector = self._get_selector(membrane)

        # 计算合并种群的 Pareto 前沿
        fronts = algorithm._non_dominated_sort(combined)
        pareto_seqs = {ind.sequence for ind in fronts[0]} if fronts else set()

        # 环境选择后存活的序列
        selected = algorithm._environmental_selection(combined, membrane.size)
        survived_seqs = {ind.sequence for ind in selected}

        for ind in new_individuals:
            op = getattr(ind, 'last_operator', None)
            if op is None:
                continue
            reward = 0.0
            if ind.sequence in survived_seqs:
                reward += 1.0
            if ind.sequence in pareto_seqs:
                reward += 2.0
            selector.update(op, reward)

    # ======================================================================
    # 超体积历史记录
    # ======================================================================
    def _record_hv(self, membrane: Population):
        """记录膜当前 Pareto 前沿的超体积值（用于停滞检测）。"""
        role = getattr(membrane, 'role', 'structure')
        pts = _get_pareto_points(membrane.individuals)
        if not pts:
            hv = 0.0
        else:
            f1_vals = [p[0] for p in pts]
            f2_vals = [p[1] for p in pts]
            hv = _hypervolume_2d(pts, HV_REF_POINT)
        self._hv_history[role].append(hv)

    def _is_stagnating(self, membrane: Population) -> bool:
        """
        判断膜是否停滞：最近 stagnation_window 代内 HV 变化量 < 阈值。
        """
        role = getattr(membrane, 'role', 'structure')
        history = self._hv_history.get(role, [])
        if len(history) < self.stagnation_window:
            return False
        recent = history[-self.stagnation_window:]
        delta_hv = abs(recent[-1] - recent[0])
        return delta_hv < self.stagnation_threshold

    # ======================================================================
    # 事件驱动膜间通信
    # ======================================================================
    def _adjust_communication_params(self, generation: int, max_generations: int):
        """
        动态调整通信参数，根据进化进度和膜状态优化通信策略。
        
        Args:
            generation:      当前进化代数
            max_generations: 最大进化代数
        """
        if max_generations <= 1:
            return
        
        # 计算进化进度
        progress = generation / (max_generations - 1)
        
        # 1. 动态调整迁出概率：进化后期增加迁出概率
        self.p_out = self.base_p_out + (0.5 - self.base_p_out) * progress
        
        # 2. 动态调整个体最小年龄：根据进化长度调整
        self.t_min = max(2, int(self.base_t_min + progress * 3))
        
        # 3. 动态调整停滞检测窗口：根据进化长度调整
        self.stagnation_window = max(5, min(20, int(10 + progress * 10)))
        
        # 4. 动态调整迁移个体数：以 base_migration_size 为基准，乘膜规模系数
        avg_population_size = (len(self.membrane1.individuals) + 
                              len(self.membrane2.individuals) + 
                              len(self.membrane3.individuals)) / 3
        pop_factor = avg_population_size / 50.0
        self.migration_size = max(3, min(15, int(self.base_migration_size * pop_factor)))
        
        # 5. 动态调整皮膜容量：根据总种群大小调整
        total_population = (len(self.membrane1.individuals) + 
                          len(self.membrane2.individuals) + 
                          len(self.membrane3.individuals))
        self.max_archive_size = max(50, min(200, int(total_population * 0.3)))

    def _communicate_event_driven(self, generation: int):
        """
        P 系统通信规则：
          迁出（Out）：rank=1 且 crowding_distance >= 阈值 且 age >= t_min
                       以概率 p_out 将副本迁移到皮膜精英档案。
          迁入（In） ：当某膜停滞时，从皮膜精英档案抽样注入（替换最差个体）。

        消融模式控制：
          no_comm          → 直接返回（完全不通信）
          fixed_comm       → 每 fixed_comm_interval 代强制迁移（不管停滞检测）
          no_skin          → 跳过迁入/迁出到皮膜，仅做横向通信
          single_membrane  → 直接返回（单膜模式无膜间通信）
          full             → 原始事件驱动逻辑
        """
        # ── 消融：no_comm / single_membrane → 完全不通信 ─────────────────
        if self.ablation_mode in (self.ABLATION_NO_COMM, self.ABLATION_SINGLE_MEMBRANE):
            return

        membranes = [self.membrane1, self.membrane2, self.membrane3]

        # ── 消融：fixed_comm → 按固定间隔强制迁移 ────────────────────────
        if self.ablation_mode == self.ABLATION_FIXED_COMM:
            if (generation % self.fixed_comm_interval) == 0:
                # 强制迁出
                for membrane in membranes:
                    self._migrate_out(membrane)
                    self.comm_migrate_out_count += 1
                # 强制迁入（不管是否停滞）
                if len(self.elite_archive) > 0:
                    for membrane in membranes:
                        self._migrate_in(membrane)
                        self.comm_migrate_in_count += 1
                        role = getattr(membrane, 'role', '?')
                        self._comm_log.append(
                            f"[gen={generation}][fixed_comm] 强制向膜({role})迁入 {self.migration_size} 个个体"
                        )
                # 横向通信（固定间隔同样执行）
                self._lateral_migrate(src=self.membrane1, dst=self.membrane2, n=self.migration_size)
                self._lateral_migrate(src=self.membrane2, dst=self.membrane3, n=self.migration_size)
                self.comm_lateral_count += 2
            return

        # ── 消融：no_skin → 只做横向通信，跳过皮膜迁入/迁出 ─────────────
        if self.ablation_mode == self.ABLATION_NO_SKIN:
            self._lateral_migrate(src=self.membrane1, dst=self.membrane2, n=self.migration_size)
            self._lateral_migrate(src=self.membrane2, dst=self.membrane3, n=self.migration_size)
            if self._should_feedback(self.membrane3, self.membrane1):
                self._lateral_migrate(src=self.membrane3, dst=self.membrane1, n=self.migration_size)
            if self._should_feedback(self.membrane3, self.membrane2):
                self._lateral_migrate(src=self.membrane3, dst=self.membrane2, n=self.migration_size)
            return

        # ── full：标准策略（去重迁出+自适应迁入+内容感知横向）───
        if self.ablation_mode == self.ABLATION_FULL:
            # ① 迁出：每代执行，但序列级去重 + 每次推入上限
            for membrane in membranes:
                self._migrate_out_v2(membrane)

            # ② 迁入：自适应间隔，追加不替换
            max_gen = getattr(self, 'total_generations', 100) or 100
            progress = generation / max(max_gen - 1, 1)
            interval = max(1, round(self.interval_min + progress * (self.interval_max - self.interval_min)))
            if generation % interval == 0 and len(self.elite_archive) > 0:
                for membrane in membranes:
                    self._migrate_in_v2(membrane)
                    role = getattr(membrane, 'role', '?')
                    self._comm_log.append(
                        f"[gen={generation}][v2] 向膜({role}) 注入互补解 (interval={interval})"
                    )
                    self.stagnation_events[role] = self.stagnation_events.get(role, 0) + 1

            # ③ 横向通信：自适应间隔 + 内容感知，追加不替换
            if generation % interval == 0:
                # 膜1→膜2：传与膜2相似度最低的解（多样性注入）
                self._lateral_explore_to_struct(n=self.migration_size)
                # 膜2→膜3：传螺旋度合规 + MIC 最优的解
                self._lateral_struct_to_safety(n=self.migration_size)
                # 膜3→膜1/2：溶血最低有条件反馈
                self._lateral_safety_feedback_v2(n=self.migration_size)
            return

        # ── event_driven：旧版事件驱动通信（仅停滞检测敏感性分析使用）─
        if self.ablation_mode == self.ABLATION_EVENT_DRIVEN:

            # ── 迁出：三个膜 → 皮膜 ──────────────────────────────────────────
            for membrane in membranes:
                self._migrate_out(membrane)
            # ── 迁入：停滞膜 ← 皮膜 ──────────────────────────────────────────
            for membrane in membranes:
                if self._is_stagnating(membrane) and len(self.elite_archive) > 0:
                    self._migrate_in(membrane)
                    role = getattr(membrane, 'role', '?')
                    self._comm_log.append(
                        f"[gen={generation}] 膜({role}) 停滞，从皮膜注入 {self.migration_size} 个个体"
                    )
                    self.comm_migrate_in_count += 1
                    self.stagnation_events[role] = self.stagnation_events.get(role, 0) + 1

            # ── 横向通信：基于膜状态的动态通信 ─────────────────────────
            # 1. 膜1（探索）→ 膜2（结构）：传递新颖序列
            self._lateral_migrate(src=self.membrane1, dst=self.membrane2,
                                   n=self.migration_size)
        
            # 2. 膜2（结构）→ 膜3（安全）：传递结构优化的序列
            self._lateral_migrate(src=self.membrane2, dst=self.membrane3,
                                   n=self.migration_size)
        
            # 3. 膜3（安全）→ 膜1/膜2：传递低毒序列（反馈机制）
            # 只有当膜3的平均溶血活性优于膜1/膜2时才进行反馈
            if self._should_feedback(self.membrane3, self.membrane1):
                self._lateral_migrate(src=self.membrane3, dst=self.membrane1,
                                       n=self.migration_size)
        
            if self._should_feedback(self.membrane3, self.membrane2):
                self._lateral_migrate(src=self.membrane3, dst=self.membrane2,
                                       n=self.migration_size)

    def _migrate_out(self, membrane: Population):
        """
        将膜中满足迁出条件的个体副本加入皮膜精英档案。
        条件：rank=1（Pareto 前沿）且 crowding_distance >= thresh 且 age >= t_min。
        """
        # 计算 Pareto 前沿
        pareto_front = membrane.get_pareto_front()
        pareto_seqs = {ind.sequence for ind in pareto_front}

        for ind in membrane.individuals:
            if ind.sequence not in pareto_seqs:
                continue
            cd = getattr(ind, 'crowding_distance', 0.0)
            age = getattr(ind, '_age', 0)
            if cd >= self.crowding_out_thresh and age >= self.t_min:
                if random.random() < self.p_out:
                    self.elite_archive.append(ind.clone())
                    self.comm_migrate_out_count += 1

    def _migrate_out_v2(self, membrane: Population):
        """
        标准策略迁出：序列级去重 + 每次推入上限。
        相比原版 _migrate_out，避免重复副本堆积导致 archive 膨胀。
        条件：rank=1（Pareto 前沿）且 crowding_distance >= thresh 且 age >= t_min。
        """
        pareto_front = membrane.get_pareto_front()
        pareto_seqs = {ind.sequence for ind in pareto_front}
        # 维护 archive 当前序列集合，O(1) 查询
        archive_seqs = {ind.sequence for ind in self.elite_archive}
        added = 0

        for ind in membrane.individuals:
            if added >= self.migration_size:
                break
            if ind.sequence not in pareto_seqs:
                continue
            if ind.sequence in archive_seqs:      # ← 核心：去重
                continue
            cd = getattr(ind, 'crowding_distance', 0.0)
            age = getattr(ind, '_age', 0)
            if cd >= self.crowding_out_thresh and age >= self.t_min:
                if random.random() < self.p_out:
                    self.elite_archive.append(ind.clone())
                    archive_seqs.add(ind.sequence) # 同步更新集合，防同代重入
                    self.comm_migrate_out_count += 1
                    added += 1

    def _is_similar_to_population(self, individual: Peptide, population: Population, threshold: float = 0.8) -> bool:
        """
        检查个体是否与种群中现有个体相似。
        
        Args:
            individual: 待检查的个体
            population: 目标种群
            threshold: 相似度阈值，超过此值则认为相似
            
        Returns:
            bool: 是否与种群中个体相似
        """
        for pop_ind in population.individuals:
            # 计算序列相似度（使用编辑距离的简单实现）
            seq1 = individual.sequence
            seq2 = pop_ind.sequence
            min_len = min(len(seq1), len(seq2))
            max_len = max(len(seq1), len(seq2))
            
            # 计算相同位置的氨基酸数量
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
            similarity = matches / max_len
            
            if similarity >= threshold:
                return True
        return False

    def _generate_random_individual(self, membrane: Population) -> Peptide:
        """
        生成新的随机个体。
        
        Args:
            membrane: 目标膜
            
        Returns:
            Peptide: 新的随机个体
        """
        import random
        from peptide import Peptide
        
        # 获取膜的长度范围
        length_range = getattr(membrane, 'length_range', (10, 30))
        fixed_length = getattr(membrane, 'fixed_length', None)
        
        # 生成随机长度
        if fixed_length is not None:
            length = fixed_length
        else:
            length = random.randint(length_range[0], length_range[1])
        
        # 生成随机氨基酸序列
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequence = ''.join(random.choice(amino_acids) for _ in range(length))
        
        return Peptide(sequence, length_range=length_range, fixed_length=fixed_length)

    def _migrate_in(self, membrane: Population):
        """
        从皮膜精英档案随机采样 migration_size 个个体注入膜，
        替换膜中 Pareto 等级最差（domination_count 最大）的个体。
        实现三层防护机制：
        1. 从精英档案采样
        2. 相似性过滤
        3. 备用策略（生成新的随机个体）
        """
        # 第一层：从精英档案采样
        if not self.elite_archive:
            return
        
        n = min(self.migration_size, len(self.elite_archive))
        candidates = random.sample(self.elite_archive, n)
        
        # 第二层：相似性过滤
        filtered_immigrants = []
        for immigrant in candidates:
            if not self._is_similar_to_population(immigrant, membrane):
                filtered_immigrants.append(immigrant)
        
        # 第三层：备用策略（生成新的随机个体）
        if len(filtered_immigrants) < n:
            remaining = n - len(filtered_immigrants)
            for _ in range(remaining):
                random_ind = self._generate_random_individual(membrane)
                filtered_immigrants.append(random_ind)
        
        # 找到膜中最差的 n 个个体（按 domination_count 降序）
        sorted_inds = sorted(
            membrane.individuals,
            key=lambda x: getattr(x, 'domination_count', 0),
            reverse=True
        )
        replace_count = min(len(filtered_immigrants), len(sorted_inds))
        replace_seqs = {ind.sequence for ind in sorted_inds[:replace_count]}
        
        new_population = [
            ind for ind in membrane.individuals
            if ind.sequence not in replace_seqs
        ]
        for immigrant in filtered_immigrants[:replace_count]:
            cloned = immigrant.clone() if hasattr(immigrant, 'clone') else immigrant
            cloned._age = 0   # 重置年龄
            new_population.append(cloned)
        
        membrane.individuals = new_population[:membrane.size]

    def _migrate_in_v2(self, membrane: Population):
        """
        标准策略迁入：
        - 内容互补：优先从 archive 中选目标膜 Pareto 前沿"未覆盖"的序列
        - 追加不替换：注入个体直接追加到种群，由下一代 NSGA-II 环境选择淘汰
        - 保留相似度过滤（threshold=0.8）防止同质注入
        """
        if not self.elite_archive:
            return

        # ── 内容互补：选 archive 中目标膜 Pareto 前沿尚未包含的序列 ──────
        membrane_front_seqs = {ind.sequence for ind in membrane.get_pareto_front()}
        complementary = [p for p in self.elite_archive
                         if p.sequence not in membrane_front_seqs]
        pool = complementary if len(complementary) >= self.migration_size else self.elite_archive

        n = min(self.migration_size, len(pool))
        candidates = random.sample(pool, n)

        # ── 相似度过滤 ───────────────────────────────────────────────────
        filtered = [c for c in candidates
                    if not self._is_similar_to_population(c, membrane)]
        if not filtered:
            filtered = candidates  # 过滤过严时降级

        # ── 追加不替换（让 NSGA-II 下一代选择淘汰）─────────────────────
        for ind in filtered:
            cloned = ind.clone() if hasattr(ind, 'clone') else ind
            cloned._age = 0
            membrane.individuals.append(cloned)
        self.comm_migrate_in_count += 1

    def _should_feedback(self, src: Population, dst: Population) -> bool:
        """
        判断源膜是否应该向目标膜反馈个体。
        当源膜的平均溶血活性优于目标膜时返回True。
        
        Args:
            src: 源膜
            dst: 目标膜
            
        Returns:
            bool: 是否应该反馈
        """
        # 计算源膜的平均溶血活性
        src_hemo_values = [ind.hemolytic_activity for ind in src.individuals 
                          if ind.hemolytic_activity is not None]
        if not src_hemo_values:
            return False
        src_avg_hemo = sum(src_hemo_values) / len(src_hemo_values)
        
        # 计算目标膜的平均溶血活性
        dst_hemo_values = [ind.hemolytic_activity for ind in dst.individuals 
                          if ind.hemolytic_activity is not None]
        if not dst_hemo_values:
            return True  # 如果目标膜没有评估值，默认进行反馈
        dst_avg_hemo = sum(dst_hemo_values) / len(dst_hemo_values)
        
        # 溶血活性值越小越好，当源膜平均溶血活性优于目标膜时反馈
        return src_avg_hemo < dst_avg_hemo * 0.95  # 有5%的优势才反馈

    def _lateral_migrate(self, src: Population, dst: Population, n: int):
        """
        横向通信：将 src 膜的 top-n Pareto 精英副本注入 dst 膜，
        替换 dst 膜中最差的 n 个个体（避免种群膨胀）。
        """
        elites = src.get_best_individuals(n=n)
        if not elites:
            return

        # 去除已存在于 dst 的序列
        dst_seqs = {ind.sequence for ind in dst.individuals}
        new_elites = [e for e in elites if e.sequence not in dst_seqs]
        if not new_elites:
            return

        replace_count = min(len(new_elites), len(dst.individuals))
        # 按 domination_count 降序找最差个体
        sorted_dst = sorted(
            dst.individuals,
            key=lambda x: getattr(x, 'domination_count', 0),
            reverse=True
        )
        replace_seqs = {ind.sequence for ind in sorted_dst[:replace_count]}

        new_population = [
            ind for ind in dst.individuals
            if ind.sequence not in replace_seqs
        ]
        for elite in new_elites[:replace_count]:
            cloned = elite.clone()
            cloned._age = 0
            new_population.append(cloned)

        dst.individuals = new_population[:dst.size]
        self.comm_lateral_count += 1

    def _lateral_explore_to_struct(self, n: int):
        """
        横向：膜1（探索）→ 膜2（结构）
        传递与膜2当前种群相似度最低的 n 个膜1 Pareto 解（最大化多样性注入）。
        注入方式：追加，让 NSGA-II 选择。
        """
        elites = self.membrane1.get_pareto_front()
        if not elites:
            return
        dst_seqs = {ind.sequence for ind in self.membrane2.individuals}
        new_elites = [e for e in elites if e.sequence not in dst_seqs]
        if not new_elites:
            return

        # 按"与膜2种群的最大相似度"升序排列，取前 n（相似度最低=最新颖）
        def max_sim_to_dst(ind):
            if not self.membrane2.individuals:
                return 0.0
            sims = []
            for pop_ind in self.membrane2.individuals:
                max_len = max(len(ind.sequence), len(pop_ind.sequence))
                if max_len == 0:
                    sims.append(0.0)
                    continue
                matches = sum(a == b for a, b in zip(ind.sequence, pop_ind.sequence))
                sims.append(matches / max_len)
            return max(sims) if sims else 0.0

        new_elites.sort(key=max_sim_to_dst)      # 相似度升序 → 最新颖在前
        send = new_elites[:n]

        for ind in send:
            cloned = ind.clone()
            cloned._age = 0
            self.membrane2.individuals.append(cloned)
        self.comm_lateral_count += 1

    def _lateral_struct_to_safety(self, n: int):
        """
        横向：膜2（结构）→ 膜3（安全）
        传递满足当前螺旋度约束 AND antimicrobial_activity 最优（最小）的 n 个解。
        注入方式：追加，让 NSGA-II 选择。
        """
        elites = self.membrane2.get_pareto_front()
        if not elites:
            return
        dst_seqs = {ind.sequence for ind in self.membrane3.individuals}
        new_elites = [e for e in elites if e.sequence not in dst_seqs]
        if not new_elites:
            return

        # 筛选满足螺旋度约束的解（螺旋度 >= 当前 eps）
        current_eps = getattr(self.membrane2, 'helicity_eps',
                              self.helicity_eps_start)
        qualified = [e for e in new_elites
                     if getattr(e, 'helicity', 0.0) is not None
                     and getattr(e, 'helicity', 0.0) >= current_eps]
        if not qualified:
            qualified = new_elites  # 无合格解时降级（传全部新颖解）

        # 按 antimicrobial_activity 升序（MIC 越小越好）
        qualified.sort(key=lambda x: x.antimicrobial_activity
                       if x.antimicrobial_activity is not None else float('inf'))
        send = qualified[:n]

        for ind in send:
            cloned = ind.clone()
            cloned._age = 0
            self.membrane3.individuals.append(cloned)
        self.comm_lateral_count += 1

    def _lateral_safety_feedback_v2(self, n: int):
        """
        横向：膜3（安全）→ 膜1/膜2 有条件反馈
        条件：膜3平均溶血优于目标膜 5%；传递溶血活性最低的 n 个解。
        注入方式：追加，让 NSGA-II 选择。
        """
        elites3 = self.membrane3.get_pareto_front()
        if not elites3:
            return

        # 按溶血活性升序（越小越好）
        elites3_sorted = sorted(
            elites3,
            key=lambda x: x.hemolytic_activity
            if x.hemolytic_activity is not None else float('inf')
        )

        for target in (self.membrane1, self.membrane2):
            if not self._should_feedback(self.membrane3, target):
                continue
            dst_seqs = {ind.sequence for ind in target.individuals}
            send = [e for e in elites3_sorted if e.sequence not in dst_seqs][:n]
            for ind in send:
                cloned = ind.clone()
                cloned._age = 0
                target.individuals.append(cloned)
            self.comm_lateral_count += 1

    def _update_elite_archive(self):
        """
        将各膜的 Pareto 前沿个体收集到皮膜档案，
        使用真正的非支配排序 + 拥挤距离裁剪到 max_archive_size。

        消融模式 no_skin / single_membrane 下直接跳过（无皮肤膜档案）。
        """
        # ── 消融：no_skin / single_membrane → 不维护精英档案 ────────────
        if self.ablation_mode in (self.ABLATION_NO_SKIN, self.ABLATION_SINGLE_MEMBRANE):
            return

        # 收集各膜 Pareto 前沿个体（已评估且目标值非 None 的）
        candidates = []
        for membrane in [self.membrane1, self.membrane2, self.membrane3]:
            for ind in membrane.get_pareto_front():
                if (ind.antimicrobial_activity is not None
                        and ind.hemolytic_activity is not None):
                    candidates.append(ind)

        if not candidates:
            return

        # 合并到现有档案
        all_candidates = self.elite_archive + candidates

        # 去重（按序列字符串）
        seen = set()
        unique = []
        for ind in all_candidates:
            if ind.sequence not in seen:
                seen.add(ind.sequence)
                unique.append(ind)

        # 非支配排序
        fronts = self._non_dominated_sort_archive(unique)

        # 按前沿逐层填充，直到达到 max_archive_size
        new_archive = []
        for front in fronts:
            if len(new_archive) + len(front) <= self.max_archive_size:
                new_archive.extend(front)
            else:
                # 用拥挤距离填充剩余位置
                remaining = self.max_archive_size - len(new_archive)
                front_sorted = self._crowding_sort(front)
                new_archive.extend(front_sorted[:remaining])
                break

        self.elite_archive = new_archive

    def _non_dominated_sort_archive(self, population: List[Peptide]) -> List[List[Peptide]]:
        """对皮膜档案使用非支配排序（独立于进化算法中的同名方法）。"""
        fronts: List[List[Peptide]] = [[]]
        for ind in population:
            ind.domination_count = 0
            ind.dominated_solutions = []

        for i, p in enumerate(population):
            for q in population[i + 1:]:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                    q.domination_count += 1
                elif q.dominates(p):
                    q.dominated_solutions.append(p)
                    p.domination_count += 1

        for ind in population:
            if ind.domination_count == 0:
                fronts[0].append(ind)

        i = 0
        while fronts[i]:
            next_front = []
            for ind in fronts[i]:
                for dominated in ind.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)

        if not fronts[-1]:
            fronts.pop()
        return fronts

    def _crowding_sort(self, front: List[Peptide]) -> List[Peptide]:
        """计算拥挤距离并按降序排列。"""
        size = len(front)
        if size == 0:
            return front
        for ind in front:
            ind.crowding_distance = 0.0
        if size == 1:
            front[0].crowding_distance = float('inf')
            return front

        valid = [ind for ind in front
                 if ind.antimicrobial_activity is not None
                 and ind.hemolytic_activity is not None]
        if not valid:
            return front

        for key in ['antimicrobial_activity', 'hemolytic_activity']:
            vals = [getattr(ind, key) for ind in valid]
            sorted_v = sorted(valid, key=lambda x: getattr(x, key))
            sorted_v[0].crowding_distance = float('inf')
            sorted_v[-1].crowding_distance = float('inf')
            v_min, v_max = min(vals), max(vals)
            scale = v_max - v_min if v_max != v_min else 1.0
            for j in range(1, len(sorted_v) - 1):
                v_plus = getattr(sorted_v[j + 1], key)
                v_minus = getattr(sorted_v[j - 1], key)
                sorted_v[j].crowding_distance += (v_plus - v_minus) / scale

        return sorted(front, key=lambda x: x.crowding_distance, reverse=True)

    # ======================================================================
    # 辅助：交叉变异率
    # ======================================================================
    def _get_crossover_rate(self, role: str) -> float:
        return {'exploration': 0.9, 'structure': 0.8, 'safety': 0.7}.get(role, 0.8)

    def _get_mutation_rate(self, role: str) -> float:
        return {'exploration': 0.15, 'structure': 0.1, 'safety': 0.08}.get(role, 0.1)

    # ======================================================================
    # 公共查询接口
    # ======================================================================
    def get_elite_archive(self) -> List[Peptide]:
        """返回全局精英档案（皮膜）。"""
        return self.elite_archive

    def get_membrane_populations(self):
        """返回三个膜的 Population 对象。"""
        return self.membrane1, self.membrane2, self.membrane3

    def get_operator_stats(self) -> Dict[str, dict]:
        """返回各膜算子选择器的统计信息。"""
        return {
            role: selector.get_stats()
            for role, selector in self.operator_selectors.items()
        }

    def get_comm_log(self) -> List[str]:
        """返回通信事件日志。"""
        return self._comm_log
