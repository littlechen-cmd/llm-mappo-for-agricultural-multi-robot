# MAPPO + LLM 多机协同调度研究现状分析

## 一、文献概览

对目录下 10 篇论文（去重后有效 9 篇）进行了系统性阅读，以下按与"MAPPO+LLM 协同调度"主题的相关度进行分类分析。

---

## 二、论文分类与核心内容

### 第一梯队：直接使用 MAPPO + LLM 的工作（3 篇）

#### 1. **LLM-MAPPO：LLM 增强的混合流水车间动态调度** ⭐ 最核心参考
> Gu et al., "Large language model-empowered dynamic scheduling for intelligent hybrid flow shop using multi-agent deep reinforcement learning", *Advanced Engineering Informatics*, 2026.

- **问题场景**：混合流水车间带不相关并行机（HFSP-UPM），以最小化 makespan 为目标
- **核心技术路线**：
  - 每个加工阶段建模为一个自主 Agent（多智能体制造系统 MAMS）
  - 采用 CTDE（集中训练-分散执行）范式
  - 通过结构化 Prompt Engineering，利用 LLM 增强状态空间和动作选择的语义理解能力
  - 设计了**事件驱动的 Agent 间通信机制**以促进阶段间协调
  - 使用 **LLM-MAPPO** 训练调度模型
- **实验结果**：在 330 个实例上相比调度规则、遗传编程规则、多种先进 DRL 方法以及基线 MAPPO 均有显著提升，多数实例性能提升超过 **8%**；且在新生产场景下展现出良好的泛化能力和自适应调整能力
- **LLM 角色**：增强语义理解、辅助状态表征和策略更新

#### 2. **LLM-Enhanced MAPPO for Traffic Control**
> Chen & Meng, "LLM-Enhanced MARL for Smarter Traffic Control", *IEEE ICCA*, 2025.

- **问题场景**：多路口协同交通信号控制
- **核心技术路线**：
  - 以 MAPPO 为基础 MARL 算法，论证了 MAPPO 在 CTDE 框架下相比 MADDPG 具有更好的稳定性和收敛性
  - 两个关键创新：(1) 在 MAPPO 网络中集成 **Transformer 模块**以增强模型表达能力和加速收敛；(2) 引入 **LLM 用于任务规划**，利用其推理能力改善多 Agent 协作
- **LLM 角色**：高层策略推理 + 任务规划，弥补 MARL 在维度灾难和局部可观测性方面的不足

#### 3. **MAPPO-BDP / LLM-BDP：通信高效的多智能体协作**
> Su et al., "Towards Communication Efficient Multi-Agent Cooperations: Reinforcement Learning and LLM", *IEEE TVT*, 2026.

- **问题场景**：带宽受限条件下的多机器人导航协作
- **核心技术路线**：
  - 提出**面向目标的人类可解释特征通信协议**（Goal-Oriented Human-Interpretable Feature Communication），只传输语义最重要的特征
  - **MAPPO-BDP**（Bandwidth-Adaptive Dual-Policy）：双策略架构——通信策略（受限 MAPPO）+ 导航策略（MAPPO）
  - **LLM-BDP**：利用 LLM 替代 MAPPO-BDP，**仅修改 prompt 即可适配不同带宽约束，无需重新训练**
  - MAPPO 训练的机器人与 LLM-BDP 机器人可**无缝协作**，无需联合训练
- **LLM 角色**：替代 RL 策略进行通信与导航决策，仅通过 prompt 工程即可自适应带宽约束
- **核心贡献**：证明了 LLM 在消除 RL 重新训练需求方面的独特优势

---

### 第二梯队：LLM + MARL（非 MAPPO）的工作（1 篇）

#### 4. **LAMARL：LLM 辅助的 MARL 协作策略生成**
> Zhu et al., "LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation", *IEEE RA-L*, 2025.

- **问题场景**：多机器人形状组装任务
- **核心技术路线**：
  - 使用 LLM **全自动生成先验策略函数和奖励函数**，无需人工设计
  - 先验策略集成到 Actor Loss 中（通过正则化项约束 RL 策略模仿 LLM 生成的先验动作）
  - LLM 生成的奖励函数集成到 MARL 环境中
  - 使用 **MADDPG**（非 MAPPO）作为基础 MARL 算法
- **实验结果**：先验策略平均提升样本效率 185.9%；基于 CoT 和 API 的结构化 prompt 提升 LLM 输出成功率 28.5%-67.5%
- **LLM 角色**：自动化奖励设计和先验策略生成，解决 MARL 样本效率低的瓶颈
- **自称为首个将 LLM 与 MARL 结合实现全自主多机器人策略生成的工作**

---

### 第三梯队：LLM + 单智能体 RL 的工作（2 篇）

#### 5. **LLM + PPO 的机器人操作与任务规划**
> Huynh & Nguyen, "Integrative AI framework for robotics: LLM-enabled reinforcement learning in object manipulation and task planning", *Robotics and Autonomous Systems*, 2026.

- 使用 GPT-4 + PPO（单智能体），设计了 Adaptable Weighted Decision Module 动态平衡 LLM 和 RL 的输出权重
- 静态环境成功率 90%（vs RL-only 72%, GPT-4-only 78%），动态环境成功率 85%

#### 6. **LLM + 风险敏感 RL 的智能工厂人机协作**
> Wang & Zhou, "LLM-Guided risk-sensitive reinforcement learning for smart factories", *Expert Systems with Applications*, 2026.

- 双 Agent 框架：LLM Agent（知识增强 + 任务指令生成）+ Robot Agent（贝叶斯神经网络 + 风险敏感 RL）
- 使用轻量级 LLM（LLaMa-8B / Qwen-7B）以确保工厂级可部署性
- 通过贝叶斯概率量化静态和动态风险

---

### 第四梯队：LLM + 多智能体系统但无 MARL 的工作（1 篇）

#### 7. **LLM + MAS 的机器人导航**
> Samarathunga et al., "LLM-Guided Multi-Agent System for Natural Language-Based Robot Navigation", *IEEE IAAI*, 2025.

- GPT-4o + LangChain/LangGraph 多智能体系统 + RRT 路径规划
- **不涉及 RL**，基于 LLM 和规则的多 Agent 架构

---

### 第五梯队：与主题弱相关（1 篇）

#### 8. **LeAD：LLM + 端到端自动驾驶**
> Zhang et al., "LeAD: The LLM Enhanced Planning System Converged with End-to-end Autonomous Driving", 2025.

- 双速率架构：高频 E2E（模仿学习）+ 低频 LLM（CoT 推理）
- 不涉及 MARL 或 MAPPO

---

## 三、MAPPO + LLM 协同调度的核心研究趋势

### 3.1 LLM 在 MAPPO 框架中的四种角色定位

| 角色定位 | 代表工作 | 机制 |
|---|---|---|
| **状态/动作空间增强** | Gu et al. (LLM-MAPPO) | 通过 Prompt Engineering 增强 Agent 的语义理解，辅助状态表征 |
| **高层任务规划** | Chen & Meng (Traffic Control) | LLM 提供策略推理，嵌入 MAPPO 的决策流程 |
| **策略生成与自动化** | Su et al. (LLM-BDP) | LLM 直接替代 RL 策略，仅通过修改 prompt 即可自适应约束变化 |
| **奖励函数自动设计** | Zhu et al. (LAMARL) | LLM 自动生成先验策略和奖励函数，消除人工设计需求 |

### 3.2 CTDE 框架的普遍采用

三篇直接相关的 MAPPO+LLM 工作均采用 **CTDE（Centralized Training with Decentralized Execution）** 框架：
- 训练阶段利用全局信息进行集中式 Critic 评估
- 执行阶段各 Agent 仅基于局部观测做出决策
- 这被认为是 MAPPO 在多智能体系统中优于 MADDPG 等算法的基础

### 3.3 LLM 引入的关键优势

1. **语义理解与零样本泛化**：LLM 的预训练知识使 Agent 能够理解复杂的任务语义，并在新场景下（如新的带宽约束、新的生产场景）无需重新训练即可适应
2. **自动化设计**：LAMARL 等证明了 LLM 可以全自动生成奖励函数和先验策略，大幅降低 MARL 应用中的人工设计成本
3. **样本效率提升**：LLM 生成的先验策略可提升样本效率达 185.9%，这对实际应用中 RL 训练成本高昂的问题有显著缓解
4. **人机可解释性**：如 Su et al. 提出的人类可解释特征通信协议，LLM 使 Agent 间通信从抽象嵌入变为可理解的语义信息

### 3.4 当前研究的关键挑战

1. **LLM 推理延迟 vs 实时性需求**：LLM 推理速度远低于 RL 策略推理，LeAD 等采用双速率架构（高频 RL + 低频 LLM）是折中方案
2. **幻觉问题**：LLM 输出的不可靠性在安全关键场景（如工厂调度）中是一个核心障碍，Wang & Zhou 通过嵌入领域专家知识库来缓解
3. **LLM 部署成本**：大型 LLM（如 GPT-4）难以在资源受限的边缘设备上部署，轻量级模型（LLaMa-8B, Qwen-7B）是趋势
4. **多机通信开销**：Su et al. 提出的带宽约束下的选择通信是一个新兴方向，LLM 在此处有望大幅简化通信策略设计

---

## 四、研究空白与潜在方向

基于以上文献分析，当前 MAPPO + LLM 协同调度领域存在以下研究空白：

| 空白方向 | 现有工作覆盖 | 潜在切入点 |
|---|---|---|
| **LLM 直接嵌入 MAPPO 网络架构** | 仅在 Traffic Control 论文中通过 Transformer 模块集成 | 将 LLM 作为 MAPPO 的 Actor/Critic 特征提取器 |
| **LLM 驱动的 Agent 间自适应通信** | Su et al. 仅处理固定特征选择 | 利用 LLM 实时判断何时与谁通信、传输什么内容 |
| **多机动态任务分配中的 LLM 角色** | 尚无直接工作 | LLM 作为中央协调器或协商仲裁器 |
| **LLM 多机调度中的在线重规划** | ⚠️ 部分填补（见补充调研） | LLM 根据实时事件动态调整 MAPPO 策略 |
| **轻量级 LLM 的边缘部署方案** | Wang & Zhou 提及但未深入 | 蒸馏 + 量化 + MAPPO 联合优化 |
| **LLM 辅助的课程学习与迁移学习** | 空白 | LLM 生成训练课程，加速 MAPPO 在不同规模任务上的迁移 |

> 📎 **补充调研**：「LLM 多机调度中的在线重规划」空白已通过新增文献检索进行了系统补充，详见 [`LLM在线动态重规划文献补充调研.md`](LLM在线动态重规划文献补充调研.md)。
> 核心结论：LLM 在线重调度已有 DScheLLM、ALAS、LLM-OWA、SeEvo 等工作推进，但 **LLM + MAPPO + 在线重规划** 的三者结合仍属空白。

---

## 五、对研究方向的建议

1. **最值得追踪的工作**：Gu et al. 的 **LLM-MAPPO** 是唯一直接将 MAPPO 与 LLM 结合解决调度问题的工作，其事件驱动通信机制和阶段级 Agent 建模值得深入参考。

2. **LAMARL 的自动化思路可迁移**：Zhu et al. 的 LLM 自动生成奖励函数和先验策略的方法论可以适配到 MAPPO 框架中（原文使用 MADDPG），这是提升 MAPPO 样本效率的低成本路径。

3. **LLM 替代 RL 重训练**：Su et al. 证明 LLM 可通过 prompt 工程消除重训练需求，这一思路在多机调度的动态约束适应（如新任务类型、新资源约束）中有重要的借鉴价值。

4. **风险感知与可信度**：Wang & Zhou 的贝叶斯方法 + 领域知识增强 + 轻量级 LLM 路线，为 MAPPO+LLM 在实际工业部署中的可信度问题提供了解决思路。

---

## 六、论文清单

| # | 论文 | 年份 | 发表 | 与主题相关度 |
|---|------|------|------|:---:|
| 1 | LLM-MAPPO for HFSP-UPM (Gu et al.) | 2026 | *Advanced Engineering Informatics* | ⭐⭐⭐⭐⭐ |
| 2 | LLM-Enhanced MARL for Traffic (Chen & Meng) | 2025 | *IEEE ICCA* | ⭐⭐⭐⭐ |
| 3 | MAPPO-BDP / LLM-BDP (Su et al.) | 2026 | *IEEE TVT* | ⭐⭐⭐⭐ |
| 4 | LAMARL (Zhu et al.) | 2025 | *IEEE RA-L* | ⭐⭐⭐⭐ |
| 5 | LLM + PPO Robotics (Huynh & Nguyen) | 2026 | *RAS* | ⭐⭐⭐ |
| 6 | LLM + Risk-Sensitive RL (Wang & Zhou) | 2026 | *ESWA* | ⭐⭐⭐ |
| 7 | LLM + MAS Navigation (Samarathunga et al.) | 2025 | *IEEE IAAI* | ⭐⭐ |
| 8 | LeAD Autonomous Driving (Zhang et al.) | 2025 | arXiv | ⭐ |

> 注：`main.pdf` 与论文 #1 为同一篇论文的副本；`LLM-Enhanced_MARL_for_Smarter_Traffic_Control（会议）.pdf` 与论文 #2 为同一篇论文的会议版本。

---

*分析日期：2026-07-13*
