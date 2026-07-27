from collections import Counter

from train.train_mappo import log_evaluation_scalars, log_training_scalars


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = (value, step)


def test_training_tensorboard_logs_required_scalars_and_events():
    writer = RecordingWriter()
    log_training_scalars(
        writer=writer,
        episode=12,
        native_return=1.5,
        completed_tasks=3,
        target_completed_tasks=4,
        event_counts=Counter({"PICKING_COMPLETED": 3, "AGV_DEAD": 1}),
        battery_mean=8.25,
        prior_mix=0.5,
        metrics={
            "entropy": 0.2,
            "critic_loss": 0.3,
            "actor_loss": -0.1,
            "prior_loss": 0.4,
            "shaped_return": 2.0,
            "task_shaping_return": 0.5,
        },
    )

    required_tags = {
        "native_return",
        "success_rate",
        "deaths",
        "prior_mix",
        "entropy",
        "critic_loss",
        "actor_loss",
        "battery_mean",
        "events_count/PICKING_COMPLETED",
        "events_count/AGV_DEAD",
    }
    assert required_tags <= writer.scalars.keys()
    assert writer.scalars["success_rate"] == (0, 12)
    assert writer.scalars["events_count/CHARGED"] == (0.0, 12)


def test_evaluation_tensorboard_logs_actor_only_metrics_and_events():
    writer = RecordingWriter()
    log_evaluation_scalars(
        writer,
        episode=100,
        evaluation={
            "episodes": 20,
            "native_return": 1.8,
            "full_success_rate": 0.9,
            "deaths": 2,
            "entropy": 0.15,
            "battery_mean": 8.5,
            "mean_completed_tasks": 3.8,
            "task_completion_ratio": 0.95,
            "mean_steps_to_full_completion": 75.0,
            "event_counts": Counter({"PICKING_COMPLETED": 76, "AGV_DEAD": 2}),
        },
    )

    required_tags = {
        "native_return",
        "success_rate",
        "deaths",
        "prior_mix",
        "entropy",
        "battery_mean",
        "events_count/PICKING_COMPLETED",
        "events_total/PICKING_COMPLETED",
    }
    assert required_tags <= writer.scalars.keys()
    assert writer.scalars["prior_mix"] == (0.0, 100)
    assert writer.scalars["events_count/PICKING_COMPLETED"] == (3.8, 100)
