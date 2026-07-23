# MAPPO 执行器设计

## 职责边界

`RulePlanner` 是当前高层规划基线，未来 `LLMPlanner` 必须实现相同的
`HighLevelPlanner.plan(snapshot) -> PlannerDecision` 接口。规划器只能输出
任务类型、目标位置、货架 ID、优先级和计划有效期，不能输出低层动作。

MAPPO Actor 是所有同构 AGV 共用的单一参数网络。每个 AGV 独立接收：

- 环境原始向量观测；
- 高层任务条件向量；
- 以自身为中心的 10 通道局部网格；
- 环境安全层计算的动作掩码。

Actor 输出 `NOOP / FORWARD / LEFT / RIGHT / TOGGLE_LOAD / CHARGE` 的离散分布。

集中式 Critic 使用 10 通道全局特征图，包括货架、请求货架、AGV、载货 AGV、
死亡 AGV、拣选机器人、停靠位、充电站和高层任务目标/优先级。Critic 只在训练时
使用该全局状态，执行时只运行参数共享 Actor，满足 CTDE。

## 训练入口

```powershell
D:\Anaconda3\envs\py310\python.exe train\train_mappo.py --config configs\mappo_tiny_2ag.yaml --device cpu
```

服务器上使用 CUDA PyTorch 后，将 `--device cuda`。训练 checkpoint 默认写入
`artifacts/`，该目录已被 Git 忽略。

训练入口会优先从统一仓库中的 `robotic-warehouse/` 加载 RWARE 源码，避免误用
Python 环境里可能已安装的旧版 `rware`。服务器首次部署时仍建议执行
`pip install -e robotic-warehouse`，使交互式环境命令也指向本项目版本。

## 当前阶段

这是一条可运行的纯 MAPPO + RulePlanner 基线。真实 LLM 尚未接入训练循环；接入时
只替换规划器实现，并保留 `PlannerDecision`、状态编码、安全掩码和 MAPPO 网络不变。
