# LLM 在线动态重规划领域文献补充调研

> 针对「LLM 多机调度中的在线重规划」这一研究空白，补充检索了 2024–2026 年最新文献。结论：该空白**已开始被填补**，但**多智能体 RL（特别是 MAPPO）+ LLM 在线重规划**的结合仍属前沿空白。

---

## 一、新发现的核心文献（9 篇）

### 1. DScheLLM：微调双系统 LLM 动态车间调度 ⭐⭐⭐⭐⭐
> Zhang et al., "DScheLLM: Enabling Dynamic Scheduling through a Fine-Tuned Dual-System Large Language Model", *arXiv:2601.09100*, 2026.

- **最接近在线重调度需求的工作**
- **双系统架构（Fast-Slow Reasoning）**：
  - Fast-thinking mode：快速生成高质量调度方案
  - Slow-thinking mode：生成与 OR 求解器兼容的结构化决策输入
- 基于华为庞戈嵌入式 7B 模型，使用 **LoRA 微调**
- 训练数据由 OR 求解器生成的精确调度结果构造
- 处理的动态扰动：加工时间变化、机器可用性变化、**意外任务插入**
- **关键贡献**：声称是**首个将 LLM 应用于动态车间调度的工作**
- 局限：仅是单智能体调度（非多机 Agent），不涉及 MARL/MAPPO

| 维度 | 评估 |
|------|------|
| 在线重规划 | ✅ Fast-slow 双模式，按扰动规模灵活响应 |
| 多智能体 | ❌ 单 LLM 决策 |
| MARL/MAPPO | ❌ 不涉及 |
| 调度场景 | ✅ 标准 Job Shop 基准 |

---

### 2. ALAS：有状态多 LLM Agent 的扰动感知规划 ⭐⭐⭐⭐⭐
> Chang & Geng, "ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning", *arXiv:2505.12501*, 2025.

- **核心贡献**：专门针对 LLM 在事务性规划中的四个根本缺陷：
  1. 缺乏自我验证（absence of self-verification）
  2. 上下文侵蚀（context erosion）
  3. 下一个 token 短视（next-token myopia）
  4. 缺乏持久状态（lack of persistent state）
- **重规划机制**：当扰动发生时，Agent 执行**历史感知的局部补偿（history-aware local compensation）**，避免代价高昂的全局重规划，抑制级联效应
- 角色特化的 Agent + 自动状态追踪 + 轻量协议协调
- 在真实大规模 Job-Shop 调度基准上取得了**静态和动态场景的双重新 SOTA**
- **关键启示**：局部补偿 vs 全局重规划的权衡——为 MAPPO+LLM 的在线重规划策略提供了直接参考

| 维度 | 评估 |
|------|------|
| 在线重规划 | ✅ 局部补偿，避免全局重规划 |
| 多智能体 | ✅ 角色特化 Agent |
| MARL/MAPPO | ❌ 不涉及 RL |
| 调度场景 | ✅ 大规模 Job-Shop |

---

### 3. LLM-OWA：编排-工作者 Agent 的闭环协作调度 ⭐⭐⭐⭐⭐
> Gao et al., "Multitask Fine-Tuning Agentic AI Based Collaborative Scheduling for Flexible Manufacturing Systems", *Journal of Manufacturing Systems*, 2026.

- **高度相关**：适用于柔性制造系统（FMS）的 LLM 编排-工作者 Agent（OWA）
- **闭环协作决策**：规划级 Agent + 执行级 Agent（部署在机器、AGV 的调度系统和工人 AR 设备上）
- **多任务微调（MFT）**：通过核心参数冻结 + DARE 减少跨任务参数干扰
- **异常事件处理**：
  - 状态故障（state failure）
  - 任务时间故障（task time failure）
  - **紧急订单插入（emergency order insertion）**
- 在异常事件下达到 **0.9146 决策一致性**和 **0.9496 可解释性得分**
- 相比 NSGA-II/III，解空间标准差降低 ≥18.06%，平均质量提升 ≥3.03%

| 维度 | 评估 |
|------|------|
| 在线重规划 | ✅ 闭环协作 + 异常事件处理 |
| 多智能体 | ✅ 编排-工作者 Agent |
| MARL/MAPPO | ❌ 不涉及 |
| 调度场景 | ✅ FMS 柔性制造 |

---

### 4. SeEvo：基于本地 LLM 的进化式重调度 ⭐⭐⭐⭐
> Huang et al., "Leveraging Large Language Models for Efficient Scheduling in Human-Robot Collaborative Flexible Manufacturing Systems", *npj Advanced Manufacturing (Nature)*, 2025.

- **核心机制**：离线构建精英 HDR（启发式调度规则）知识库 → 在线单次快速迭代生成重调度方案
- 种群自进化：个体协同进化 + 自进化 + 集体进化
- **在线响应**：急单/机器故障发生时，**1 分钟内**生成优化调度计划
- 微调的 Qwen2.5-SFT-7B，仅需一张 4090D GPU
- 54 个真实 HRC 场景验证，makespan 平均降低 **21.52%**
- **关键启示**：LLM + 进化算法结合的"离线训练+在线快速生成"范式，是可部署化的重调度方案

| 维度 | 评估 |
|------|------|
| 在线重规划 | ✅ 单次迭代快速重调度 |
| 多智能体 | ⚠️ 非显式 Agent 架构 |
| MARL/MAPPO | ❌ 进化算法替代 RL |
| 调度场景 | ✅ HRC 柔性制造 |

---

### 5. LLM + Digital Twin 的自适应多机器人任务分配 ⭐⭐⭐⭐
> Deng et al., "Integrating LLMs and Digital Twins for Adaptive Multi-Robot Task Allocation in Construction", *arXiv:2506.18178*, 2025.

- LLM + 数字孪生 + 整数规划（IP）的三合一框架
- **叙事驱动的调度自适应**：LLM 解释非结构化自然语言输入 → 自动更新优化约束 → 无需手动编码即可实现 human-in-the-loop 灵活性
- 数字孪生实现物理-数字实时同步，形成**闭环反馈框架**
- 顶级 LLM 模型在约束和参数提取上达到 **>97% 准确率**
- 处理的扰动：物料延迟、非预期现场条件、天气导致的干扰

| 维度 | 评估 |
|------|------|
| 在线重规划 | ✅ 叙事驱动 + 闭环反馈 |
| 多智能体 | ✅ 多机器人协作 |
| MARL/MAPPO | ❌ 整数规划替代 RL |
| 调度场景 | ✅ 建筑施工 |

---

### 6. BrainBody-LLM：闭环状态反馈的双 LLM 规划 ⭐⭐⭐
> Kakde et al., "Grounding LLMs for Robot Task Planning Using Closed-loop State Feedback", *arXiv:2402.08546*, 2024.

- 双 LLM 架构：高层规划 LLM + 低层控制 LLM
- **闭环状态反馈**：实时环境状态和错误消息在检测到偏差时触发规划修正
- 跨 LLM 后端均表现出鲁棒性，验证了改进语言模型 + 闭环重规划能力的收益
- 局限：仅针对单机器人操作任务，非多机调度

---

### 7. RePLan：VLM 驱动的机器人实时重规划 ⭐⭐⭐
> "RePLan: Robotic Replanning with Perception and Language Models", *arXiv:2401.04157*, 2024.

- 利用 VLM（视觉-语言模型）实现**在线重规划**
- 当物理执行与计划不一致时自动检测并触发重规划
- 专为长时域（long-horizon）机器人操作任务设计
- 局限：非调度场景，单机器人

---

### 8. LGC-MARL：LLM 规划器 + 图协作 MARL ⭐⭐⭐
> Jia et al., "Enhancing Multi-Agent Systems via Reinforcement Learning with LLM-based Planner and Graph-based Policy", *IEEE ICRA*, 2025.

- LLM 将复杂任务分解为可执行子任务 → 生成动作依赖图
- 基于图的协作元策略实现多 Agent 通信与协作，通过元学习适应新任务
- AI2-THOR 平台验证
- **与在线重规划相关**：LLM-Critic 模型评估子任务合理性，形成任务级反馈
- 局限：任务分解侧重离线规划，**非动态环境下的在线重规划**

---

### 9. IMR-LLM：工业多机器人任务规划 ⭐⭐⭐
> Su et al., "IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation using Large Language Models", *arXiv:2603.02669*, 2026.

- LLM + 析取图（Disjunctive Graph）+ 操作过程树
- 确定性求解器处理资源冲突，确保调度无死锁
- 7 机器人 × 24 操作成功完成
- **关键**：目前以**开环方式运行**，论文明确指出未来方向是"整合反馈回路，让 LLM 根据实时执行数据重新规划或调整程序"

---

## 二、综合对比分析

### 2.1 各方案的技术路线

| 论文 | 在线重规划机制 | 决策主体 | 规划粒度 | 部署难度 |
|------|:---:|---|---|---|
| DScheLLM | Fast-Slow 双模推理 | 单 LLM | 全局调度 | 低（LoRA，7B） |
| ALAS | 局部补偿（避免全局） | 多 LLM Agent | 全局+局部 | 中 |
| LLM-OWA | 闭环编排-执行 | 编排者+工作者 | 全局+分布式 | 中（多 Agent 部署） |
| SeEvo | 复用精英知识库 | 单 LLM | 全局时序 | 低（7B，单 GPU） |
| DT+LLM | 叙事触发+约束自适应 | LLM+IP 求解器 | 全局分配 | 中（需数字孪生） |
| BrainBody | 闭环状态反馈 | 双 LLM | 任务级 | 中 |
| RePLan | 感知触发 | VLM | 任务级 | 高（需 VLM） |
| LGC-MARL | LLM-Critic 评估 | LLM+MARL | 任务级 | 中 |
| IMR-LLM | **尚无（开环）** | LLM+确定性求解 | 操作级 | 中 |

### 2.2 与 MAPPO + LLM 在线重规划的距离

上述 9 篇文献可分为三类：

| 类别 | 文献 | 与目标距离 |
|------|------|:---:|
| **A类：LLM 在线重调度，但不涉及 MARL** | DScheLLM, ALAS, LLM-OWA, SeEvo, DT+LLM | 🔴 核心机制可用，缺多 Agent RL 层 |
| **B类：LLM + RL/元学习，但非在线重规划** | LGC-MARL (LLM+MARL) | 🟡 MARL 已有，但在线重规划能力弱 |
| **C类：LLM 重规划，但非调度场景** | BrainBody-LLM, RePLan | 🟠 重规划机制可参考，场景不匹配 |

**目前尚无文献同时满足以下三个条件**：
1. ✅ LLM 驱动的在线动态重规划
2. ✅ 多智能体强化学习（MAPPO）作为低层控制器
3. ✅ 面向制造/物流等协同调度场景

这正是「LLM 在线动态重规划 + MAPPO」的研究空白所在。

---

## 三、可借鉴的关键技术组件

从上述文献中提取可迁移到 MAPPO+LLM 在线重规划框架中的技术组件：

### 3.1 双系统架构（借鉴 DScheLLM）

```
┌─────────────────────────────────────────────┐
│                  调度系统                      │
│  ┌──────────────┐      ┌──────────────┐      │
│  │  LLM 慢系统   │      │  MAPPO 快系统  │      │
│  │  (低频触发)   │◄────►│  (高频执行)   │      │
│  │  · 态势理解   │      │  · 实时调度   │      │
│  │  · 策略调整   │      │  · 分散执行   │      │
│  │  · 通信协调   │      │  · 本地决策   │      │
│  └──────────────┘      └──────────────┘      │
└─────────────────────────────────────────────┘
```

- **MAPPO 快系统**：维持原有多机实时调度策略（毫秒级推理）
- **LLM 慢系统**：仅在检测到异常事件（机器故障、订单插入、加工延迟等）时触发，进行策略调整建议
- 借鉴 ALAS 的**局部补偿**思想：扰动仅影响局部 Agent 时，只重规划相关 Agent 的策略参数

### 3.2 闭环反馈机制（借鉴 DT+LLM 和 LLM-OWA）

- 在执行层建立状态监控
- 异常检测 → LLM 理解异常语义 → 判断是否需要全局/局部重规划 → 触发 MAPPO 策略调整

### 3.3 知识复用机制（借鉴 SeEvo）

- 维护"异常-响应"知识库
- 遇到已知类型异常时，从知识库快速检索历史应对方案
- 新类型异常由 LLM 推理生成响应并入库

---

## 四、研究空白精确定位

经过本次补充调研，原报告中「LLM 多机调度中的在线重规划」这一空白可以更精确地表述为：

> **LLM 驱动的在线异常检测 → LLM 语义理解与重规划决策 → MAPPO 多 Agent 策略自适应调整** 的完整闭环框架，目前尚未有工作实现。

具体未解决的问题包括：

1. **LLM 异常触发时机**：什么频率/什么条件下 LLM 应该介入？如何权衡推理延迟和响应时效？
2. **重规划粒度决策**：何时进行局部 Agent 策略调整（借鉴 ALAS 局部补偿），何时需要全局策略重规划？
3. **LLM → MAPPO 参数传递**：LLM 的语义决策如何转化为 MAPPO 策略网络的参数修改？
4. **多 Agent 间的通信协调**：LLM 重规划如何影响 Agent 间的通信策略（借鉴 Su et al.）？
5. **可解释性**：如何让 LLM 的重规划决策对人工监控者可追踪、可审计（借鉴 LLM-OWA 的 0.9496 可解释性得分）？

---

## 五、潜在切入方向

结合现有文献的启示，提出以下可落地切入方向：

### 方向 A：事件驱动的 MAPPO+LLM 双通道重规划

- MAPPO 正常执行分散调度
- 事件检测模块（基于阈值/异常检测器）触发 LLM 介入
- LLM 分析事件语义（"机器 3 故障，影响 Stage 2"），判断影响范围
- 若为局部影响，LLM 调整对应 Agent 的策略参数（如动作掩码、奖励权重）
- 若为全局影响，LLM 生成新的宏观任务分配方案，MAPPO 各 Agent 据此重新学习

### 方向 B：LLM 作为"重规划仲裁器"

- 多个 MAPPO Agent 在异常事件下可能产生冲突的策略调整
- LLM 作为仲裁器，根据全局目标（如最小化 makespan）裁决策略冲突
- 借鉴 LLM-OWA 的编排-工作者架构

### 方向 C：异常经验知识库 + 轻量级 LLM 推理

- 借鉴 SeEvo 的精英知识库思路
- 每次 MAPPO+LLM 成功处理异常事件后，将经验存入知识库
- 后续同类异常通过检索匹配加速响应（秒级），未知异常由 LLM 推理（分钟级）

---

## 六、文献清单汇总

| # | 论文 | 年份 | 来源 | 与"在线重规划"相关度 |
|---|------|------|------|:---:|
| 1 | DScheLLM (Zhang et al.) | 2026 | arXiv | ⭐⭐⭐⭐⭐ |
| 2 | ALAS (Chang & Geng) | 2025 | arXiv | ⭐⭐⭐⭐⭐ |
| 3 | LLM-OWA for FMS (Gao et al.) | 2026 | *J. Manuf. Syst.* | ⭐⭐⭐⭐⭐ |
| 4 | SeEvo HRC (Huang et al.) | 2025 | *npj Adv. Manuf.* (Nature) | ⭐⭐⭐⭐ |
| 5 | LLM+DT Task Alloc (Deng et al.) | 2025 | arXiv | ⭐⭐⭐⭐ |
| 6 | BrainBody-LLM (Kakde et al.) | 2024 | arXiv / *Adv. Robotics Research* | ⭐⭐⭐ |
| 7 | RePLan | 2024 | arXiv | ⭐⭐⭐ |
| 8 | LGC-MARL (Jia et al.) | 2025 | *IEEE ICRA* | ⭐⭐⭐ |
| 9 | IMR-LLM (Su et al.) | 2026 | arXiv | ⭐⭐ |

---

*调研日期：2026-07-13*
