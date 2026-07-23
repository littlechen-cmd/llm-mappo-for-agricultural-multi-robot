# MAPPO+LLM 仓储异构机器人调度的异步调用机制设计

> 本方案综合参考了目录下 4 篇核心论文的异步/事件驱动机制：LeAD 的双速率异步耦合、Gu et al. (LLM-MAPPO) 的事件驱动通信 + KL 散度语义对齐、ALAS 的局部补偿重规划、DScheLLM 的 Fast-Slow 双系统架构。

---

## 一、总体架构：三层异步金字塔

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 3 · LLM 编排层（事件驱动，≥5s 间隔）                      │
│  触发条件：异常事件 / 订单插入 / 人工指令 / 性能偏离阈值          │
│  职责：全局任务重分配、策略参数调整、异常影响评估                  │
├──────────────────────────────────────────────────────────────┤
│  Layer 2 · LLM 语义增强层（定时触发，~30s 间隔）                  │
│  触发条件：固定周期定时器                                       │
│  职责：状态空间语义化、拥堵模式识别、区域语义标注                  │
├──────────────────────────────────────────────────────────────┤
│  Layer 1 · MAPPO 执行层（连续，~100ms/step）                     │
│  无中断运行，LLM 介入时仅更新观测/参数，不停机                     │
│  职责：实时路径规划、避碰协调、设备控制、局部决策                  │
└──────────────────────────────────────────────────────────────┘
```

### 设计原则（逐条溯源）

| 原则 | 来源论文 | 具体出处 |
|------|------|------|
| 高频执行层不停机 | LeAD | "high-frequency fast system ... uninterrupted" (Section III-B) |
| LLM 仅在条件满足时激活 | LeAD | "LLM redundant system activates when exceeding waiting threshold" |
| 事件驱动 + 决策点触发 | Gu et al. | "event-driven communication mechanism ... at each decision point" (Section 4.2) |
| 语义蒸馏而非替代 | Gu et al. | "LLM preference distribution → KL divergence into MAPPO objective" (Eq.16) |
| 局部补偿优于全局重规划 | ALAS | "history-aware local compensation, avoiding costly global replanning" |
| 双模式推理（Fast/Slow） | DScheLLM | "fast-thinking mode ... slow-thinking mode ... solver-compatible inputs" |

---

## 二、Layer 1 → Layer 2：语义增强接口

### 2.1 触发机制：定时 + 事件补充

借鉴 Gu et al. 的决策点触发原则，语义增强采用「固定周期 + 性能触发」混合策略：

```
触发条件（满足任一即触发）：
  A. 定时周期到达（T_semantic = 30s）
  B. 任一 Agent 的瞬时 Reward 连续 N = 5 步低于历史均值的 80%
  C. 系统级拥堵检测：>2 个 Agent 同时触发避碰等待
```

### 2.2 数据流

```
MAPPO 观测缓冲区（最近 10 步的全局状态快照）
    │
    ▼
┌────────────────────────────────────────────┐
│  Gu et al. 式结构化 Prompt 构建             │
│  · 12 维结构化特征（各Agent位置/速度/负载）  │
│  · N 维非结构化语义（订单描述 / 设备事件）    │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│  LLM 推理（轻量级模型，目标 <2s）             │
│  输出：区域拥堵等级 / 紧急语义标注 /         │
│        异构能力约束更新                      │
└───────────────┬────────────────────────────┘
                ▼
┌────────────────────────────────────────────┐
│  Gu et al. 式 KL 语义蒸馏                   │
│  将 LLM 输出的语义分布 p_LLM 通过            │
│  L_total = L_MAPPO + α·D_KL(p_LLM ‖ π_θ)  │
│  注入 MAPPO Critic 网络                     │
└───────────────┬────────────────────────────┘
                ▼
        MAPPO Agent 观测增强
```

### 2.3 关键参数

| 参数 | 推荐值 | 来源 |
|------|:---:|------|
| α (KL 权重) | 0.05-0.15 | Gu et al. KL 项平衡系数 |
| T_semantic | 30s | 仓储场景经验值，远低于 LLM 推理时延 |
| LLM 推理超时 | 2s | 低于 LeAD 的"waiting threshold"概念 |

---

## 三、Layer 1 → Layer 3：事件驱动的编排重规划接口

### 3.1 触发条件分类

借鉴 LeAD 的阈值触发机制 + ALAS 的局部/全局区分策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    事件分类与响应策略                          │
├──────────────────┬──────────────┬───────────────────────────┤
│ 事件类型          │ 影响范围     │ LLM 响应                  │
├──────────────────┼──────────────┼───────────────────────────┤
│ 单台AGV故障       │ 局部         │ 仅重分配该AGV任务队列      │
│ 单台AGV低电量预警  │ 局部         │ 调整该AGV充电优先级参数    │
│ 紧急订单插入      │ 全局(可降级) │ 重评估全局优先级权重       │
│ 连续拥堵(>30s)    │ 区域         │ 调整该区域Agent路径约束    │
│ 拣选失败/库存异常  │ 局部         │ 重新分配该SKU的拣选任务   │
│ 人工干预指令      │ 可变         │ NLP解析→约束/目标更新     │
│ 全局makespan偏离  │ 全局         │ 全局策略参数调整           │
│ 充电站排队超阈值  │ 全局         │ 调整充电策略奖励权重       │
└──────────────────┴──────────────┴───────────────────────────┘
```

### 3.2 状态机设计

```
                  ┌─────────────┐
                  │   NORMAL    │  MAPPO 自由执行
                  │   (常态)     │\__________________________
                  └──────┬──────┘                             \
         事件触发        │                                      \
         检测到异常       │  LLM已在运行?                         \
                         ▼                                       \
              ┌─────────────────┐                    ┌─────────────────┐
              │ 事件分类与影响    │                    │ 加入待处理队列    │
              │ 评估(本地)       │                    │ (FIFO, 容量=5)   │
              └────────┬────────┘                    └────────┬────────┘
                       │                                      │
          ┌────────────┼────────────┐                         │
          ▼            ▼            ▼                         │
    ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
    │ 局部事件  │ │ 区域事件  │ │ 全局事件  │                    │
    │ (ALAS风格)│ │          │ │          │                    │
    └────┬─────┘ └────┬─────┘ └────┬─────┘                    │
         │            │            │                           │
         ▼            ▼            ▼                           │
    ┌─────────────────────────────────────────┐               │
    │        ALAS 局部补偿模块（借鉴）           │              │
    │  · 历史感知：查询异常知识库中           │               │
    │    同类事件的历史处理方案                │              │
    │  · 若命中（相似度>0.8）：                │              │
    │    直接应用，跳过 LLM 调用              │              │
    │  · 若未命中：转 LLM 推理               │              │
    └────────────────┬────────────────────────┘               │
                     │                                          │
              ┌──────┴──────┐                                  │
              ▼             ▼                                  │
     ┌────────────┐  ┌────────────┐                            │
     │ 知识库命中  │  │ 需要LLM推理 │◄──────────────────────────┘
     │ (秒级响应)  │  │ (异步调用)  │
     └─────┬──────┘  └─────┬──────┘
           │               │
           │               ▼
           │      ┌─────────────────────┐
           │      │  DScheLLM式 推理模式  │
           │      │  选择（借鉴）         │
           │      │                     │
           │      │ 局部/区域事件:       │
           │      │  → Fast Mode        │
           │      │  (约束式快速响应)     │
           │      │                     │
           │      │ 全局事件:            │
           │      │  → Slow Mode        │
           │      │  (链式推理+完整方案)  │
           │      └──────────┬──────────┘
           │                 │
           ▼                 ▼
    ┌─────────────────────────────────────────┐
    │            LLM 输出 → 策略参数修改        │
    │                                         │
    │  局部事件输出：                           │
    │    · 任务重分配列表（JSON）                │
    │    · 约束参数更新（动作掩码）              │
    │                                         │
    │  全局事件输出：                           │
    │    · 全局优先级权重向量                    │
    │    · 奖励函数权重调整                      │
    │    · 通信拓扑结构修改（可选）              │
    └──────────────────┬──────────────────────┘
                       ▼
              ┌─────────────────┐
              │  LeAD式安全校验  │
              │  · 硬约束不变    │
              │  · 碰撞安全保证  │
              │  · 死锁预检测    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 回写 MAPPO 环境  │
              │ · 更新任务队列   │
              │ · 更新约束参数   │
              │ · 记录知识库     │
              └────────┬────────┘
                       │
                       ▼
                  ┌─────────────┐
                  │   NORMAL    │  ← 返回常态
                  └─────────────┘
```

### 3.3 知识库匹配机制（ALAS 局部补偿的精化）

借鉴 ALAS 的"history-aware local compensation"，设计异常知识库：

```python
# 异常事件特征向量
EventFeature = {
    "event_type": str,      # "AGV_FAULT" | "ORDER_INSERT" | "CONGESTION" | ...
    "affected_agents": list, # 受影响 Agent ID 列表
    "location_zone": str,    # 仓库区域编码
    "severity_level": int,   # 1-5 严重等级
    "context_state": dict,   # 触发时的全局状态摘要
}

# 知识库条目
KnowledgeEntry = {
    "feature": EventFeature,
    "response": {
        "action_type": str,      # "LOCAL_REASSIGN" | "REGION_CONSTRAINT" | "GLOBAL_REPLAN"
        "parameter_patch": dict, # 策略参数修改
        "effectiveness": float,  # 历史成功率
    },
    "timestamp": datetime,
    "use_count": int,
}

# 匹配函数：余弦相似度 > 0.8 → 直接应用；否则 → LLM推理
```

**关键设计**：知识库仅存储参数修改（parameter_patch），不存储完整计划，确保修改是局部化的、可叠加的。

---

## 四、核心时序设计

### 4.1 LLM 推理期间的 MAPPO 运行策略

借鉴 LeAD 的核心思路——**"fast system uninterrupted"**：

```
                        LLM 推理周期(1-5s)
MAPPO执行 ───────────────────────────────────────────────────→
           │                                                  │
           ├─ LLM 触发点                                       │
           │  MAPPO 继续使用旧参数运行                          │
           │  等待LLM推理完成                                   │
           │                                                  │
           │    LLM推理完成 ─────────────────────→              │
           │    更新MAPPO参数                                  │
           │    (热更新, 不reset环境)                           │
           │                                                  │
           └──────────────────────────────────────────────────→
```

**两层 LLM 的并发控制**：

```
                Layer 2    Layer 3
时间轴          语义增强    编排重规划
────────────────────────────────────────────
t=0          │ 定时触发  │         │
t=2s         │ 推理完成  │         │
t=10s        │          │ 紧急订单 │ ← 打断 Layer 2? 否，可并行
t=12s        │          │ 推理完成 │
t=30s        │ 定时触发  │         │
────────────────────────────────────────────
并行策略：Layer 2 和 Layer 3 使用不同的 LLM 实例或请求槽位，互不阻塞
```

### 4.2 多层并发安全保证

| 场景 | 处理策略 | 来源 |
|------|------|------|
| Layer 3 推理中，Layer 2 定时到达 | 并行调用（两个独立 LLM 实例/槽位） | — |
| Layer 3 推理中，新的 Layer 3 事件到达 | 入队列，当前推理完成后批量处理 | LeAD 的单次等待策略 |
| Layer 3 推理返回时，状态已变更 | LeAD 式安全校验：验证约束有效性后写入，否则丢弃 | LeAD Safety Controller |
| 同一事件重复触发 | 知识库去重：token 匹配最近 10 条事件 | ALAS 历史感知 |
| LLM 调用超时（>5s） | 返回降级方案：仅应用 Fast Mode 的逻辑规则 | DScheLLM Fast Mode |

---

## 五、LLM 输出 → MAPPO 策略参数修改的对接通道

这是本方案的核心创新——打通"LLM 语义决策 → MAPPO 数值参数"的通道。参考 Gu et al. 的 KL 语义蒸馏和 LLM 动作偏好分布机制。

### 5.1 三层修改通道

```
LLM 输出（JSON）
    │
    ├─ 通道 A：任务队列修改（外部环境更新）
    │   {"reassign": {"agent_2": ["job_15", "job_18"]},
    │    "defer":    {"agent_4": ["job_22"]}}
    │   → 直接修改 MAPPO 环境的任务分配表
    │
    ├─ 通道 B：约束参数修改（动作空间调整）
    │   {"action_mask": {"agent_3": [0,1,1,0,0]},    # 禁用某些路径
    │    "priority_weight": {"order_urgent": 2.5}}     # 优先级倍率
    │   → 修改 MAPPO Agent 的动作掩码和观测权重
    │
    └─ 通道 C：奖励函数注入（目标函数调整）
        {"reward_weights": {"efficiency": 0.6,          # 原 0.5
                            "energy_saving": 0.2,       # 原 0.3
                            "priority_bonus": 0.3}}     # 新增
        → 修改 MAPPO Critic 的奖励权重，下一 training step 生效
```

### 5.2 通道 C 的数学形式（直接借鉴 Gu et al.）

```python
# 原始 MAPPO 目标函数
L_MAPPO(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

# 加入 LLM 语义对齐（Gu et al. Eq.16 形式）
L_total(θ) = L_MAPPO(θ) + α · D_KL(p_LLM ‖ π_θ)

# 通道 C：LLM 修改奖励权重
R_new(s, a) = Σ_i w_i(t) · r_i(s, a)
# 其中 w_i(t) 由 LLM Layer 3 在事件触发时动态写入
```

---

## 六、与传统方案的对比

| 维度 | 传统 MAPPO | Gu et al. LLM-MAPPO | 本方案 |
|------|:---:|:---:|:---:|
| LLM 介入频率 | 无 | 每个决策点 | 事件驱动（异步） |
| LLM 推理时机 | — | 同步（阻塞MAPPO） | 异步（不阻塞） |
| 重规划粒度 | — | — | 三级（局部/区域/全局） |
| 知识积累 | — | — | 异常知识库 |
| 多Agent覆盖 | ✓ | ✓ (同构阶段) | ✓ (异构机器人) |
| 任务分配 | 隐式 | 隐式 | LLM 显式分配 |
| 奖励自适应 | ✗ | ✗ | LLM 动态调整 |

---

## 七、关键实现参数汇总

| 参数 | 推荐值 | 说明 |
|------|:---:|------|
| LLM Layer 2 周期 | 30s | 语义增强，不宜过频 |
| LLM Layer 3 最小间隔 | 5s | 防止连续异常导致频繁重规划 |
| LLM 推理超时 | 5s | 超过则降级到规则 |
| LLM 模型规格 | Qwen-7B LoRA | 单卡可部署，推理 <2s |
| 知识库匹配阈值 | 0.8 | 余弦相似度阈值 |
| 知识库容量 | 100 条 | LRU 淘汰 |
| KL 蒸馏系数 α | 0.1 | 平衡 MAPPO 策略与 LLM 建议 |
| 事件队列容量 | 5 | 超过则丢弃最旧事件 |

---

## 八、伪代码：主循环

```python
def main_loop():
    # 初始化
    env = WarehouseEnv(robots=[AGV, AGV, Picker, Sorter, ...])
    agents = [MAPPO_Agent(id) for id in env.robot_ids]
    llm_layer2 = LLM_Layer2(model="qwen-7b-lora", interval=30)  # 语义增强
    llm_layer3 = LLM_Layer3(model="qwen-7b-lora")                # 编排重规划
    kb = KnowledgeBase(max_size=100, match_threshold=0.8)
    event_queue = deque(maxlen=5)

    llm_layer2.next_trigger = time() + 30

    while not env.done:
        # ── Layer 1: MAPPO 高频执行（每步都运行）──
        obs = env.get_observations()
        actions = {aid: agent.act(obs[aid]) for aid, agent in agents.items()}
        rewards, next_obs, events = env.step(actions)

        # ── Layer 2: 定时语义增强 ──
        if time() >= llm_layer2.next_trigger and not llm_layer2.busy:
            llm_layer2.call_async(obs)  # 异步调用，不阻塞
            llm_layer2.next_trigger += 30

        # ── Layer 3: 事件驱动的重规划 ──
        if events and not llm_layer3.busy:
            # 分类事件
            local_events, region_events, global_events = classify(events)

            # ALAS风格：先尝试知识库
            resolved = kb.match_and_apply(local_events, agents, env)
            unresolved = [e for e in events if e not in resolved]

            if unresolved:
                # DScheLLM风格：选择推理模式
                mode = 'fast' if all(is_local(e) for e in unresolved) else 'slow'
                llm_layer3.call_async(unresolved, mode=mode)

        # ── 处理 LLM 完成回调 ──
        for llm_result in check_llm_completions():
            # LeAD风格：安全校验
            if safety_check(llm_result, env):
                apply_to_env(llm_result, env, agents)
                kb.add(llm_result.event, llm_result.response)
```

---

*设计日期：2026-07-16*
