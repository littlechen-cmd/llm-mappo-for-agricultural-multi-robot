# Heterogeneous Warehouse Environment

`rware.heterogeneous.HeterogeneousWarehouse` extends the repository with a
multi-scale warehouse task for AGV transport and fixed picking stations.  It
leaves the original `rware.warehouse.Warehouse` implementation unchanged.

## Sizes And Naming

The generated layouts use RWARE's shelf-column and highway rule: racks are
separated by vertical highways, rack rows by horizontal highways, and the
bottom service row is free.  The heterogeneous environment reserves the left
end of that row as `P-D-C` (picker, AGV dock, charger), then spawns AGVs on
the right.  Sizes use the same rack dimensions as RWARE:

| Size | Rack rows | Shelf columns | Grid size |
| --- | ---: | ---: | ---: |
| `tiny` | 1 | 3 | 11 x 10 |
| `small` | 2 | 3 | 20 x 10 |
| `medium` | 2 | 5 | 20 x 16 |
| `large` | 3 | 5 | 29 x 16 |

After `import rware`, create a named environment with either alias:

```python
env = gym.make("rware-heterogeneous-tiny-2ag-v0")
env = gym.make("rware-tiny-2ag-heterogeneous-v0")
```

Names are registered for `tiny`, `small`, `medium`, and `large`, with one to
six AGVs.  A custom generated size can also be built directly:

```python
env = HeterogeneousWarehouse(size="medium", n_agvs=4)
```

## Model

- Only AGVs are RL-controlled.  Pickers are fixed map entities and occupy a
  cell; they never appear in the action tuple.
- A picker is represented by `P` in a layout.  Its AGV dock is the empty cell
  directly to the right, shown as `D` during rendering.
- An AGV carrying a requested one-item shelf starts automatic picking on the
  dock.  It is chassis-locked for `picking_duration` steps (default `2`), then
  the shelf task completes, the shelf leaves the map, and the AGV is released.
- `C` marks a charging station.  The `CHARGE` action restores battery only on
  that cell.  Movement and turns drain `0.01`, loaded active movement drains
  `0.02`, and idle or invalid actions drain `0.002` by default.
- A battery at or below zero causes one `-10` penalty, permanently immobilises
  the AGV, and leaves it as a collision blocker.  `terminate_on_death=False`
  is the default; set it to `True` to terminate the episode on any death.

The compact fixed layout used by the visual demo puts the picker in the
lower-left corner:

```text
..S...A
.......
..S...A
.......
.......
P.C....
```

## Use

Run the fixed visual acceptance sequence without training an RL policy:

```powershell
cd robotic-warehouse
python heterogeneous_demo.py
```

The demo visibly exercises shelf transport, the two-step pick lock, charging,
and a dead AGV.  It requires the package's normal `pyglet` rendering
dependency and an available desktop display.

For an automated visual smoke test that closes its window after the sequence,
pass `--auto-close`.

For manual tests on any generated scale, use the keyboard controller:

```powershell
python heterogeneous_play.py --env rware-heterogeneous-medium-4ag-v0 --display-info
```

Arrow keys move or turn the selected AGV, `P`/`L` loads or unloads, `C` charges
at a charging station, `Space` waits, and `Tab` switches the selected AGV.

The environment is also registered after `import rware` as
`rware-heterogeneous-v0`.
