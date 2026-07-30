from random import Random

from rware.heterogeneous import HeterogeneousWarehouse
from rware_llm.pathfinding import AStarRoutePlanner


def test_astar_returns_legal_four_direction_routes_for_twenty_medium_targets():
    env = HeterogeneousWarehouse(
        size="medium", n_agvs=3, n_pickers=2, n_chargers=3
    )
    env.reset(seed=3)
    planner = AStarRoutePlanner()
    targets = Random(9).sample([(shelf.x, shelf.y) for shelf in env.shelfs], 20)

    for target in targets:
        result = planner.route(env, agv_id=1, target=target)
        assert result.reachable
        assert result.route.target == target
        assert result.route.eta_step == len(result.route.waypoints)
        previous = (env.agents[0].x, env.agents[0].y)
        for waypoint in result.route.waypoints:
            assert abs(waypoint[0] - previous[0]) + abs(waypoint[1] - previous[1]) == 1
            assert not env._path_blocked(env.agents[0], waypoint, False)
            previous = waypoint


def test_astar_reports_start_target_and_blocked_target_boundaries():
    env = HeterogeneousWarehouse(
        size="medium", n_agvs=3, n_pickers=2, n_chargers=3
    )
    env.reset(seed=4)
    planner = AStarRoutePlanner()
    position = env.agents[0].x, env.agents[0].y

    same = planner.route(env, 1, position)
    assert same.reachable
    assert same.route.waypoints == ()
    assert same.route.eta_step == 0
    assert planner.route(env, 1, (-1, 0)).unreachable_reason == "target_out_of_bounds"
    occupied = env.agents[1].x, env.agents[1].y
    assert (
        planner.route(env, 1, occupied, include_live_agvs=True).unreachable_reason
        == "target_blocked"
    )
