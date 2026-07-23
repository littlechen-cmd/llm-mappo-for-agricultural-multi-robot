"""Heterogeneous warehouse environment with AGVs and fixed pickers.

The environment intentionally keeps the fixed picking robots outside the
multi-agent action interface.  Only AGVs are controlled by a policy.  A
loaded AGV starts a timed picking operation after reaching a dock directly to
the right of a picker.  This makes the picker an occupied map cell while still
allowing a physically unambiguous AGV-to-picker handoff.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from rware.warehouse import Direction


WAREHOUSE_SIZES = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}


def make_rware_style_layout(
    size: str, n_agvs: int, column_height: int = 8
) -> Tuple[str, ...]:
    """Build a RWARE-style rack layout with heterogeneous service stations.

    RWARE's standard layout uses three-cell shelf columns separated by vertical
    highways and horizontal highways between rack rows.  This generator keeps
    that placement rule, replaces the lower-left delivery area with a picker,
    dock, and charger (``P.C``), and reserves the remaining lower delivery row
    for AGV spawns.
    """

    if size not in WAREHOUSE_SIZES:
        supported = ", ".join(WAREHOUSE_SIZES)
        raise ValueError(f"unknown warehouse size '{size}'; use one of {supported}")
    if n_agvs < 1:
        raise ValueError("n_agvs must be at least one")
    if column_height < 1:
        raise ValueError("column_height must be at least one")

    shelf_rows, shelf_columns = WAREHOUSE_SIZES[size]
    rows = (column_height + 1) * shelf_rows + 2
    cols = 3 * shelf_columns + 1
    layout = [["." for _ in range(cols)] for _ in range(rows)]
    queue_columns = {cols // 2 - 1, cols // 2}

    for y in range(rows):
        for x in range(cols):
            vertical_highway = x % 3 == 0
            horizontal_highway = y % (column_height + 1) == 0
            delivery_row = y == rows - 1
            queue_lane = y > rows - (column_height + 3) and x in queue_columns
            if not (vertical_highway or horizontal_highway or delivery_row or queue_lane):
                layout[y][x] = "S"

    # The picker remains an occupied cell; an AGV hands off from its right-side dock.
    layout[rows - 1][0] = "P"
    layout[rows - 1][1] = "."
    layout[rows - 1][2] = "C"

    spawn_candidates = [
        (x, rows - 1)
        for x in range(cols - 1, -1, -1)
        if x not in {0, 1, 2}
    ]
    if n_agvs > len(spawn_candidates):
        raise ValueError(
            f"{size} supports at most {len(spawn_candidates)} AGV spawns on its service row"
        )
    for x, y in spawn_candidates[:n_agvs]:
        layout[y][x] = "A"

    return tuple("".join(row) for row in layout)


class HeterogeneousAction(Enum):
    """Actions available to an AGV in :class:`HeterogeneousWarehouse`."""

    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4
    CHARGE = 5


@dataclass
class HeterogeneousShelf:
    """A shelf holds one requested item and leaves the map after picking."""

    id: int
    x: int
    y: int
    active: bool = True


@dataclass
class AGV:
    """The only policy-controlled robot type in the environment."""

    id: int
    x: int
    y: int
    dir: Direction
    battery: float = 1.0
    carrying_shelf: Optional[HeterogeneousShelf] = None
    dead: bool = False
    death_penalty_applied: bool = False
    picking_remaining: int = 0
    picking_station_id: Optional[int] = None
    req_action: HeterogeneousAction = HeterogeneousAction.NOOP

    @property
    def locked(self) -> bool:
        return self.picking_remaining > 0


@dataclass(frozen=True)
class PickingRobot:
    """A fixed, non-RL robot that occupies one picking-station cell."""

    id: int
    x: int
    y: int


class HeterogeneousWarehouse(gym.Env):
    """A grid warehouse for AGV transport and fixed-station picking.

    Layout symbols are ``.`` for a corridor, ``S`` for a shelf, ``P`` for a
    fixed picker/picking station, ``C`` for a charging station, and ``A`` for
    an AGV spawn point.  The AGV docking location for each ``P`` is the cell
    immediately to its right.  The default map therefore places the picking
    station in the lower-left corner, as required by the project model.

    Every AGV observation is a float vector with the following fields:
    ``x, y, direction, battery, loaded, dead, locked, pick-progress,
    on-charger, distance-to-charger, distance-to-pick-dock,
    active-request-count``.  Coordinates and distances are normalised to the
    map dimensions; all state local to the AGV is explicitly exposed.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    DEMO_LAYOUT = (
        "..S...A",
        ".......",
        "..S...A",
        ".......",
        ".......",
        "P.C....",
    )

    def __init__(
        self,
        layout: Optional[Sequence[str]] = None,
        size: Optional[str] = None,
        n_agvs: Optional[int] = None,
        column_height: int = 8,
        request_queue_size: int = 1,
        picking_duration: int = 2,
        max_steps: Optional[int] = 200,
        terminate_on_death: bool = False,
        move_drain: float = 0.01,
        loaded_move_drain: float = 0.02,
        standby_drain: float = 0.002,
        charge_rate: float = 0.2,
        pick_reward: float = 1.0,
        death_penalty: float = -10.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if picking_duration < 1:
            raise ValueError("picking_duration must be at least one step")
        if request_queue_size < 1:
            raise ValueError("request_queue_size must be at least one")
        if min(move_drain, loaded_move_drain, standby_drain, charge_rate) < 0:
            raise ValueError("energy rates must be non-negative")

        if layout is None:
            self.size = size or "tiny"
            self.layout = make_rware_style_layout(
                self.size, n_agvs if n_agvs is not None else 2, column_height
            )
        else:
            self.size = "custom"
            self.layout = tuple(layout)
        self._parse_layout(self.layout)
        if request_queue_size > len(self.shelf_spawns):
            raise ValueError("request_queue_size cannot exceed the shelf count")

        self.request_queue_size = request_queue_size
        self.picking_duration = picking_duration
        self.max_steps = max_steps
        self.terminate_on_death = terminate_on_death
        self.move_drain = move_drain
        self.loaded_move_drain = loaded_move_drain
        self.standby_drain = standby_drain
        self.charge_rate = charge_rate
        self.pick_reward = pick_reward
        self.death_penalty = death_penalty
        self.render_mode = render_mode

        self.n_agents = len(self.agv_spawns)
        self.action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(len(HeterogeneousAction)) for _ in range(self.n_agents))
        )
        self._observation_length = 12
        single_observation = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._observation_length,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Tuple(
            tuple(single_observation for _ in range(self.n_agents))
        )

        self.renderer = None
        # Viewer expects this attribute for standard RWARE goal rendering.
        # Heterogeneous tasks finish at fixed picking stations instead.
        self.goals: List[Tuple[int, int]] = []
        self.agents: List[AGV] = []
        self.picking_robots: List[PickingRobot] = []
        self.shelfs: List[HeterogeneousShelf] = []
        self.request_queue: List[HeterogeneousShelf] = []
        self._steps = 0

    def _parse_layout(self, layout: Sequence[str]) -> None:
        if not layout:
            raise ValueError("layout must contain at least one row")
        width = len(layout[0])
        if width == 0 or any(len(row) != width for row in layout):
            raise ValueError("layout rows must be non-empty and have the same width")

        valid_cells = {".", "S", "P", "C", "A"}
        invalid = {cell for row in layout for cell in row if cell not in valid_cells}
        if invalid:
            raise ValueError(f"unsupported layout cells: {sorted(invalid)}")

        self.grid_size = (len(layout), width)
        self.shelf_spawns: List[Tuple[int, int]] = []
        self.picker_spawns: List[Tuple[int, int]] = []
        self.charging_stations: List[Tuple[int, int]] = []
        self.agv_spawns: List[Tuple[int, int]] = []
        for y, row in enumerate(layout):
            for x, cell in enumerate(row):
                if cell == "S":
                    self.shelf_spawns.append((x, y))
                elif cell == "P":
                    self.picker_spawns.append((x, y))
                elif cell == "C":
                    self.charging_stations.append((x, y))
                elif cell == "A":
                    self.agv_spawns.append((x, y))

        if not self.shelf_spawns:
            raise ValueError("layout must contain at least one shelf (S)")
        if not self.picker_spawns:
            raise ValueError("layout must contain at least one picker (P)")
        if not self.charging_stations:
            raise ValueError("layout must contain at least one charger (C)")
        if not self.agv_spawns:
            raise ValueError("layout must contain at least one AGV spawn (A)")

        self.picking_docks: List[Tuple[int, int]] = []
        static_cells = set(self.picker_spawns) | set(self.charging_stations) | set(
            self.shelf_spawns
        ) | set(self.agv_spawns)
        for picker_x, picker_y in self.picker_spawns:
            dock = (picker_x + 1, picker_y)
            if dock[0] >= width or dock in static_cells:
                raise ValueError(
                    "each picker requires an empty dock cell immediately to its right"
                )
            self.picking_docks.append(dock)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self.shelfs = [
            HeterogeneousShelf(index + 1, x, y)
            for index, (x, y) in enumerate(self.shelf_spawns)
        ]
        self.request_queue = list(self.shelfs[: self.request_queue_size])
        self.picking_robots = [
            PickingRobot(index + 1, x, y)
            for index, (x, y) in enumerate(self.picker_spawns)
        ]
        self.agents = [
            AGV(index + 1, x, y, Direction.LEFT)
            for index, (x, y) in enumerate(self.agv_spawns)
        ]
        return self._observations(), self._get_info(events=[])

    def step(self, actions: Sequence[int]):
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, received {len(actions)}")
        if not self.agents:
            raise RuntimeError("reset() must be called before step()")

        rewards = np.zeros(self.n_agents, dtype=np.float32)
        events: List[Dict[str, object]] = []
        parsed_actions = [self._parse_action(action) for action in actions]

        for agv, action in zip(self.agents, parsed_actions):
            agv.req_action = action

        successful_actions = [False] * self.n_agents
        desired_moves: Dict[int, Tuple[int, int]] = {}
        occupied = {(agv.x, agv.y) for agv in self.agents}
        static_blockers = {(picker.x, picker.y) for picker in self.picking_robots}

        for index, (agv, action) in enumerate(zip(self.agents, parsed_actions)):
            if agv.dead or agv.locked or action != HeterogeneousAction.FORWARD:
                continue
            target = self._forward_location(agv)
            if not self._can_enter(agv, target, occupied, static_blockers):
                continue
            desired_moves[index] = target

        duplicate_targets = {
            target
            for target in desired_moves.values()
            if sum(candidate == target for candidate in desired_moves.values()) > 1
        }
        for index, target in desired_moves.items():
            if target not in duplicate_targets:
                agv = self.agents[index]
                agv.x, agv.y = target
                if agv.carrying_shelf:
                    agv.carrying_shelf.x, agv.carrying_shelf.y = target
                successful_actions[index] = True

        for index, (agv, action) in enumerate(zip(self.agents, parsed_actions)):
            if agv.dead or agv.locked or action == HeterogeneousAction.FORWARD:
                continue
            if action in (HeterogeneousAction.LEFT, HeterogeneousAction.RIGHT):
                agv.dir = self._turned_direction(agv.dir, action)
                successful_actions[index] = True
            elif action == HeterogeneousAction.TOGGLE_LOAD:
                successful_actions[index] = self._toggle_load(agv)
            elif action == HeterogeneousAction.CHARGE:
                if (agv.x, agv.y) in self.charging_stations:
                    agv.battery = min(1.0, agv.battery + self.charge_rate)
                    successful_actions[index] = True
                    events.append({"type": "CHARGED", "agv_id": agv.id})

        self._advance_picking(rewards, events)
        self._start_available_picking(events)

        for index, (agv, action) in enumerate(zip(self.agents, parsed_actions)):
            if agv.dead or action == HeterogeneousAction.CHARGE and successful_actions[index]:
                continue
            drain = self._energy_drain(agv, action, successful_actions[index])
            agv.battery = max(0.0, agv.battery - drain)
            if agv.battery <= 0.0 and not agv.dead:
                agv.dead = True
                agv.picking_remaining = 0
                agv.picking_station_id = None
                rewards[index] += self.death_penalty
                events.append({"type": "AGV_DEAD", "agv_id": agv.id})

        self._steps += 1
        tasks_complete = not any(shelf.active for shelf in self.shelfs)
        any_death = any(agv.dead for agv in self.agents)
        terminated = tasks_complete or (self.terminate_on_death and any_death)
        truncated = self.max_steps is not None and self._steps >= self.max_steps
        return (
            self._observations(),
            list(rewards),
            terminated,
            truncated,
            self._get_info(events),
        )

    def _parse_action(self, action: int) -> HeterogeneousAction:
        value = action.value if isinstance(action, HeterogeneousAction) else action
        try:
            return HeterogeneousAction(int(value))
        except (TypeError, ValueError):
            return HeterogeneousAction.NOOP

    def _forward_location(self, agv: AGV) -> Tuple[int, int]:
        if agv.dir == Direction.UP:
            return agv.x, agv.y - 1
        if agv.dir == Direction.DOWN:
            return agv.x, agv.y + 1
        if agv.dir == Direction.LEFT:
            return agv.x - 1, agv.y
        return agv.x + 1, agv.y

    @staticmethod
    def _turned_direction(direction: Direction, action: HeterogeneousAction) -> Direction:
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        offset = 1 if action == HeterogeneousAction.RIGHT else -1
        return directions[(directions.index(direction) + offset) % len(directions)]

    def _can_enter(
        self,
        agv: AGV,
        target: Tuple[int, int],
        occupied: set,
        static_blockers: set,
    ) -> bool:
        x, y = target
        rows, cols = self.grid_size
        if not (0 <= x < cols and 0 <= y < rows):
            return False
        if target in occupied or target in static_blockers:
            return False
        shelf = self._shelf_at(target)
        return not (agv.carrying_shelf and shelf is not None)

    def _shelf_at(self, location: Tuple[int, int]) -> Optional[HeterogeneousShelf]:
        for shelf in self.shelfs:
            if shelf.active and (shelf.x, shelf.y) == location:
                return shelf
        return None

    def _toggle_load(self, agv: AGV) -> bool:
        if agv.carrying_shelf is None:
            shelf = self._shelf_at((agv.x, agv.y))
            if shelf is None:
                return False
            agv.carrying_shelf = shelf
            return True

        if (agv.x, agv.y) in self.picking_docks:
            return False
        if (agv.x, agv.y) in self.charging_stations:
            return False
        if self._shelf_at((agv.x, agv.y)) is not None:
            return False
        agv.carrying_shelf.x, agv.carrying_shelf.y = agv.x, agv.y
        agv.carrying_shelf = None
        return True

    def _start_available_picking(self, events: List[Dict[str, object]]) -> None:
        occupied_stations = {
            agv.picking_station_id
            for agv in self.agents
            if agv.picking_station_id is not None
        }
        for agv in self.agents:
            if agv.dead or agv.locked or agv.carrying_shelf not in self.request_queue:
                continue
            try:
                station_id = self.picking_docks.index((agv.x, agv.y))
            except ValueError:
                continue
            if station_id in occupied_stations:
                continue
            agv.picking_station_id = station_id
            agv.picking_remaining = self.picking_duration
            occupied_stations.add(station_id)
            events.append(
                {
                    "type": "PICKING_STARTED",
                    "agv_id": agv.id,
                    "station_id": station_id + 1,
                    "duration": self.picking_duration,
                }
            )

    def _advance_picking(
        self, rewards: np.ndarray, events: List[Dict[str, object]]
    ) -> None:
        for agv in self.agents:
            if not agv.locked:
                continue
            agv.picking_remaining -= 1
            if agv.picking_remaining > 0:
                continue
            shelf = agv.carrying_shelf
            if shelf is not None:
                shelf.active = False
                shelf.x, shelf.y = -1, -1
                if shelf in self.request_queue:
                    self.request_queue.remove(shelf)
                self._enqueue_next_request()
                agv.carrying_shelf = None
                rewards[agv.id - 1] += self.pick_reward
                events.append({"type": "PICKING_COMPLETED", "agv_id": agv.id})
            agv.picking_station_id = None

    def _enqueue_next_request(self) -> None:
        for shelf in self.shelfs:
            if shelf.active and shelf not in self.request_queue:
                self.request_queue.append(shelf)
                return

    def _energy_drain(
        self, agv: AGV, action: HeterogeneousAction, successful: bool
    ) -> float:
        if agv.locked or not successful or action == HeterogeneousAction.NOOP:
            return self.standby_drain
        if action in (
            HeterogeneousAction.FORWARD,
            HeterogeneousAction.LEFT,
            HeterogeneousAction.RIGHT,
            HeterogeneousAction.TOGGLE_LOAD,
        ):
            return self.loaded_move_drain if agv.carrying_shelf else self.move_drain
        return self.standby_drain

    def _distance_to_nearest(self, position: Tuple[int, int], targets: Sequence[Tuple[int, int]]) -> float:
        return min(abs(position[0] - x) + abs(position[1] - y) for x, y in targets)

    def _observation(self, agv: AGV) -> np.ndarray:
        rows, cols = self.grid_size
        scale = max(rows + cols - 2, 1)
        direction = agv.dir.value / (len(Direction) - 1)
        picking_progress = (
            agv.picking_remaining / self.picking_duration if agv.locked else 0.0
        )
        return np.array(
            [
                agv.x / max(cols - 1, 1),
                agv.y / max(rows - 1, 1),
                direction,
                agv.battery,
                float(agv.carrying_shelf is not None),
                float(agv.dead),
                float(agv.locked),
                picking_progress,
                float((agv.x, agv.y) in self.charging_stations),
                self._distance_to_nearest((agv.x, agv.y), self.charging_stations) / scale,
                self._distance_to_nearest((agv.x, agv.y), self.picking_docks) / scale,
                len(self.request_queue) / self.request_queue_size,
            ],
            dtype=np.float32,
        )

    def _observations(self) -> Tuple[np.ndarray, ...]:
        return tuple(self._observation(agv) for agv in self.agents)

    def _get_info(self, events: List[Dict[str, object]]) -> Dict[str, object]:
        return {
            "events": events,
            "steps": self._steps,
            "active_requests": [shelf.id for shelf in self.request_queue],
            "battery": [agv.battery for agv in self.agents],
            "dead_agvs": [agv.id for agv in self.agents if agv.dead],
        }

    def render(self):
        if self.renderer is None:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
