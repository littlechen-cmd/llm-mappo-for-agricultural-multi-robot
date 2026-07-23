"""Interactively inspect a named heterogeneous RWARE environment.

Controls: arrows move/turn the selected AGV, P/L loads or unloads, C charges,
Space waits, Tab changes the selected AGV, R resets, D toggles status output,
and Escape exits.
"""

from argparse import ArgumentParser
import warnings

import gymnasium as gym
import numpy as np
import rware  # Registers named heterogeneous environments.

from rware.heterogeneous import HeterogeneousAction


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default="rware-heterogeneous-tiny-2ag-v0",
        help="Named heterogeneous environment, for example rware-heterogeneous-medium-4ag-v0.",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--display-info", action="store_true")
    return parser.parse_args()


class InteractiveHeterogeneousEnv:
    """Keyboard controller for the fixed-picker heterogeneous environments."""

    def __init__(self, env_name, max_steps, display_info):
        self.env = gym.make(env_name, render_mode="human", max_steps=max_steps)
        self.n_agents = self.env.unwrapped.n_agents
        self.current_agent_index = 0
        self.current_action = None
        self.running = True
        self.reset_requested = False
        self.display_info = display_info
        self.step_count = 0
        self.returns = np.zeros(self.n_agents)

        observations, _ = self.env.reset()
        self.env.render()
        self.env.unwrapped.renderer.window.on_key_press = self._key_press
        if self.display_info:
            self._display_info(observations, [0.0] * self.n_agents, False, {})
        self._cycle()

    def _selected_agv(self):
        return self.env.unwrapped.agents[self.current_agent_index]

    def _display_info(self, observations, rewards, finished, info):
        agv = self._selected_agv()
        print(
            f"step={self.step_count} selected=AGV {agv.id} "
            f"position=({agv.x}, {agv.y}) battery={agv.battery:.3f} "
            f"loaded={agv.carrying_shelf is not None} dead={agv.dead} "
            f"locked={agv.locked} reward={rewards[self.current_agent_index]:.3f} "
            f"finished={finished} events={info.get('events', [])}"
        )
        print(f"observation={observations[self.current_agent_index]}")

    def _help(self):
        print("Arrows: move or turn the selected AGV")
        print("P/L: load or unload a shelf; C: charge at a charger; Space: wait")
        print("Tab: select another AGV; R: reset; D: toggle status; Escape: exit")

    def _key_press(self, key_code, modifiers):
        from pyglet.window import key

        action_keys = {
            key.LEFT: HeterogeneousAction.LEFT,
            key.RIGHT: HeterogeneousAction.RIGHT,
            key.UP: HeterogeneousAction.FORWARD,
            key.P: HeterogeneousAction.TOGGLE_LOAD,
            key.L: HeterogeneousAction.TOGGLE_LOAD,
            key.C: HeterogeneousAction.CHARGE,
            key.SPACE: HeterogeneousAction.NOOP,
        }
        if key_code in action_keys:
            self.current_action = action_keys[key_code]
        elif key_code == key.TAB:
            self.current_agent_index = (self.current_agent_index + 1) % self.n_agents
            self.current_action = None
        elif key_code == key.R:
            self.reset_requested = True
            self.current_action = None
        elif key_code == key.H:
            self._help()
            self.current_action = None
        elif key_code == key.D:
            self.display_info = not self.display_info
            self.current_action = None
        elif key_code == key.ESCAPE:
            self.running = False
        else:
            self.current_action = None
            warnings.warn(f"Key {key_code} not recognized")

    def _cycle(self):
        while self.running:
            if self.reset_requested:
                observations, _ = self.env.reset()
                self.step_count = 0
                self.returns = np.zeros(self.n_agents)
                self.reset_requested = False
                if self.display_info:
                    self._display_info(observations, [0.0] * self.n_agents, False, {})

            if self.current_action is not None:
                actions = [HeterogeneousAction.NOOP.value] * self.n_agents
                actions[self.current_agent_index] = self.current_action.value
                observations, rewards, terminated, truncated, info = self.env.step(actions)
                self.returns += np.asarray(rewards)
                self.step_count += 1
                if self.display_info:
                    self._display_info(observations, rewards, terminated or truncated, info)
                if terminated or truncated:
                    self.reset_requested = True
                self.current_action = None
            self.env.render()
        self.env.close()


if __name__ == "__main__":
    args = parse_args()
    InteractiveHeterogeneousEnv(args.env, args.max_steps, args.display_info)
