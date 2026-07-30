from train.validate_phase1_medium import load_config, run_validation


def test_phase1_dynamic_validation_uses_distinct_seeded_task_sequences():
    result = run_validation(
        load_config("configs/phase1_medium_dynamic.yaml"), episodes=3, seed_start=2000
    )

    assert result["summary"]["crash_count"] == 0
    assert result["summary"]["episode_count"] == 3
    assert result["summary"]["distinct_release_sequences"] == 3
    assert all(episode["released_batch_count"] > 1 for episode in result["episodes"])
