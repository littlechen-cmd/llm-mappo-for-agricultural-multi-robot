"""Deterministic visual acceptance demo for HeterogeneousWarehouse.

This script deliberately uses fixed actions rather than an RL policy.  It
shows transport, two-step picking, charging, and a separate battery-depletion
check for the second AGV.
"""

from argparse import ArgumentParser
from time import sleep

from rware.heterogeneous import HeterogeneousAction, HeterogeneousWarehouse


FRAME_DELAY_SECONDS = 0.45


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto-close",
        action="store_true",
        help="Close after the deterministic acceptance sequence finishes.",
    )
    return parser.parse_args()


def _step(env, first_action, second_action=HeterogeneousAction.NOOP):
    observations, rewards, terminated, truncated, info = env.step(
        [first_action.value, second_action.value]
    )
    env.render()
    print(
        "step={step} actions=({first}, {second}) rewards={rewards} "
        "battery={battery} events={events}".format(
            step=info["steps"],
            first=first_action.name,
            second=second_action.name,
            rewards=[round(reward, 3) for reward in rewards],
            battery=[round(value, 3) for value in info["battery"]],
            events=[event["type"] for event in info["events"]],
        )
    )
    if terminated or truncated:
        raise RuntimeError("demo ended before all acceptance actions completed")
    sleep(FRAME_DELAY_SECONDS)
    return observations


def main(auto_close=False):
    env = HeterogeneousWarehouse(
        layout=HeterogeneousWarehouse.DEMO_LAYOUT,
        render_mode="human",
        max_steps=100,
    )
    env.reset(seed=7)
    env.render()
    sleep(FRAME_DELAY_SECONDS)

    # AGV 1 drives left to the requested shelf, loads it, then docks at P1.
    for _ in range(4):
        _step(env, HeterogeneousAction.FORWARD)
    _step(env, HeterogeneousAction.TOGGLE_LOAD)
    _step(env, HeterogeneousAction.FORWARD)
    _step(env, HeterogeneousAction.LEFT)
    for _ in range(5):
        _step(env, HeterogeneousAction.FORWARD)

    # Two locked steps are required before the picker confirms completion.
    _step(env, HeterogeneousAction.NOOP)
    _step(env, HeterogeneousAction.NOOP)

    # Dock is next to the charger. Turn right from DOWN, move, and charge.
    _step(env, HeterogeneousAction.LEFT)
    _step(env, HeterogeneousAction.FORWARD)
    _step(env, HeterogeneousAction.CHARGE)

    # A separate deterministic depletion check visualises a dead blocking AGV.
    env.agents[1].battery = env.standby_drain
    _step(env, HeterogeneousAction.NOOP, HeterogeneousAction.NOOP)
    print("Acceptance demo finished. Close the render window to exit.")
    if auto_close:
        env.close()
        return
    while env.render():
        sleep(FRAME_DELAY_SECONDS)


if __name__ == "__main__":
    main(auto_close=parse_args().auto_close)
