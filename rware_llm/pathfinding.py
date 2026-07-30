"""Deterministic four-direction A* routes over heterogeneous RWARE cells."""

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Optional, Tuple

from rware_llm.interfaces import Coordinate, RoutePlan


@dataclass(frozen=True)
class AStarRouteResult:
    route: Optional[RoutePlan]
    unreachable_reason: Optional[str] = None

    @property
    def reachable(self) -> bool:
        return self.route is not None


class AStarRoutePlanner:
    """Return legal, reproducible routes without assigning tasks or actions."""

    def route(
        self,
        env,
        agv_id: int,
        target: Coordinate,
        include_live_agvs: bool = False,
        reason: str = "dispatch",
    ) -> AStarRouteResult:
        agv = next((candidate for candidate in env.agents if candidate.id == agv_id), None)
        if agv is None:
            return AStarRouteResult(None, "unknown_agv")
        if agv.dead:
            return AStarRouteResult(None, "agv_dead")
        rows, cols = env.grid_size
        if not (0 <= target[0] < cols and 0 <= target[1] < rows):
            return AStarRouteResult(None, "target_out_of_bounds")
        if env._path_blocked(agv, target, include_live_agvs):
            return AStarRouteResult(None, "target_blocked")

        start = (agv.x, agv.y)
        if start == target:
            return AStarRouteResult(
                RoutePlan(agv_id, target, (), env._steps, reason=reason)
            )

        frontier = [(self._manhattan(start, target), 0, start)]
        best_cost = {start: 0}
        parents = {start: None}
        while frontier:
            _, cost, position = heappop(frontier)
            if cost != best_cost[position]:
                continue
            if position == target:
                waypoints = self._reconstruct(parents, target)
                return AStarRouteResult(
                    RoutePlan(
                        agv_id,
                        target,
                        waypoints,
                        env._steps + len(waypoints),
                        reason=reason,
                    )
                )
            for candidate in self._neighbors(position):
                if env._path_blocked(agv, candidate, include_live_agvs):
                    continue
                candidate_cost = cost + 1
                if candidate_cost >= best_cost.get(candidate, float("inf")):
                    continue
                best_cost[candidate] = candidate_cost
                parents[candidate] = position
                heappush(
                    frontier,
                    (
                        candidate_cost + self._manhattan(candidate, target),
                        candidate_cost,
                        candidate,
                    ),
                )
        return AStarRouteResult(None, "no_legal_route")

    @staticmethod
    def _neighbors(position: Coordinate) -> Tuple[Coordinate, ...]:
        x, y = position
        return ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))

    @staticmethod
    def _manhattan(first: Coordinate, second: Coordinate) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def _reconstruct(parents, target: Coordinate) -> Tuple[Coordinate, ...]:
        path = [target]
        while parents[path[-1]] is not None:
            path.append(parents[path[-1]])
        path.reverse()
        return tuple(path[1:])
