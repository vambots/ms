from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})  # set True for headless

import math
import numpy as np
from pxr import UsdGeom, Gf
import omni.usd
from omni.isaac.core import World

# ======================= Config =======================
MAP_W, MAP_H = 20.0, 20.0      # world size (meters)
RES = 0.10                     # grid resolution (meters/cell)
GRID_ORIGIN = (-MAP_W/2.0, -MAP_H/2.0)  # world (x,y) of grid cell (0,0)'s LOWER-LEFT corner

START_XY = (-3.0, -5.0)
GOAL_XY  = (6.0,  6.0)

AGENT_SPEED = 0.4              # m/s
AGENT_SIZE  = 0.25             # cube size (m); used for Z and for inflation radius
INFLATE_MARGIN = AGENT_SIZE * 0.5  # conservative radius (m)

OBST_AABBS = [                 # static obstacles in world XY (xmin,ymin,xmax,ymax)
    (-1.2, -3.0,  2.2, -2.0),
    (-2.0,  1.0, -1.0,  4.0),
    ( 0.5,  0.5,  3.0,  1.2),
]

DO_LOS_THINNING = False        # turn OFF first to verify clearance visually
GOAL_TOL = 0.10                # meters

# ===================== Grid utilities =====================
def world_to_grid_idx(xy):
    """World (x,y) -> grid (i,j) using FLOOR so each point maps to the cell it lies in."""
    gx = int(np.floor((xy[0] - GRID_ORIGIN[0]) / RES))
    gy = int(np.floor((xy[1] - GRID_ORIGIN[1]) / RES))
    return gy, gx  # (row=i, col=j)

def world_to_grid_min(xy):
    """Floor mapping for AABB mins (conservative fill)."""
    return world_to_grid_idx(xy)

def world_to_grid_max(xy):
    """Ceil-1 mapping for AABB maxs (conservative fill)."""
    gx = int(np.ceil((xy[0] - GRID_ORIGIN[0]) / RES)) - 1
    gy = int(np.ceil((xy[1] - GRID_ORIGIN[1]) / RES)) - 1
    return gy, gx

def grid_to_world_center(ij):
    """Grid (i,j) -> world (x,y) at the CENTER of the cell."""
    i, j = ij
    wx = GRID_ORIGIN[0] + (j + 0.5) * RES
    wy = GRID_ORIGIN[1] + (i + 0.5) * RES
    return np.array([wx, wy], dtype=float)

def in_bounds(occ, i, j):
    return 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1]

# =================== Occupancy construction ===================
H = int(MAP_H / RES)
W = int(MAP_W / RES)
occ = np.zeros((H, W), dtype=bool)

def mark_box(aabb, occ):
    xmin, ymin, xmax, ymax = aabb
    gy0, gx0 = world_to_grid_min((xmin, ymin))
    gy1, gx1 = world_to_grid_max((xmax, ymax))
    gy0, gy1 = max(0, gy0), min(H-1, gy1)
    gx0, gx1 = max(0, gx0), min(W-1, gx1)
    if gy0 <= gy1 and gx0 <= gx1:
        occ[gy0:gy1+1, gx0:gx1+1] = True

for box in OBST_AABBS:
    mark_box(box, occ)

# Inflate obstacles by robot radius (square footprint approximation)
inflate_cells = max(1, int(math.ceil(INFLATE_MARGIN / RES)))
if inflate_cells > 0:
    occ_infl = occ.copy()
    rows, cols = np.where(occ)
    for r, c in zip(rows, cols):
        r0 = max(0, r - inflate_cells)
        r1 = min(H - 1, r + inflate_cells)
        c0 = max(0, c - inflate_cells)
        c1 = min(W - 1, c + inflate_cells)
        occ_infl[r0:r1+1, c0:c1+1] = True
else:
    occ_infl = occ

def valid_cell(occ_map, ij):
    i, j = ij
    return in_bounds(occ_map, i, j) and (not occ_map[i, j])

# ===================== A* (8-connected, no corner cutting) =====================
def astar(occ_map, start_ij, goal_ij):
    import heapq
    H, W = occ_map.shape
    if not valid_cell(occ_map, start_ij) or not valid_cell(occ_map, goal_ij):
        return None

    # 8-neighborhood
    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]
    costs = [1,1,1,1, math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)]

    def ok_step(ci, cj, ni, nj):
        if not in_bounds(occ_map, ni, nj) or occ_map[ni, nj]:
            return False
        di, dj = ni - ci, nj - cj
        # prevent diagonal corner cutting
        if abs(di)==1 and abs(dj)==1:
            if occ_map[ci, nj] or occ_map[ni, cj]:
                return False
        return True

    g = {start_ij: 0.0}
    parent = {start_ij: None}
    h0 = math.hypot(goal_ij[0]-start_ij[0], goal_ij[1]-start_ij[1])
    pq = [(h0, start_ij)]
    closed = set()

    while pq:
        f_cur, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal_ij:
            path = []
            n = cur
            while n is not None:
                path.append(n)
                n = parent[n]
            return list(reversed(path))
        closed.add(cur)
        ci, cj = cur
        for (dij, cost) in zip(moves, costs):
            di, dj = dij
            ni, nj = ci + di, cj + dj
            if not ok_step(ci, cj, ni, nj):
                continue
            ng = g[cur] + cost
            if (ni, nj) not in g or ng < g[(ni, nj)]:
                g[(ni, nj)] = ng
                h = math.hypot(goal_ij[0]-ni, goal_ij[1]-nj)
                f = ng + h
                parent[(ni, nj)] = cur
                heapq.heappush(pq, (f, (ni, nj)))
    return None

# =================== Path thinning (grid LOS) ===================
def bresenham_cells(i0, j0, i1, j1):
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = (di - dj)
    i, j = i0, j0
    while True:
        yield (i, j)
        if i == i1 and j == j1:
            break
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj

def los_free(occ_map, a, b):
    for (i, j) in bresenham_cells(a[0], a[1], b[0], b[1]):
        if not in_bounds(occ_map, i, j) or occ_map[i, j]:
            return False
    return True

def thin_path_grid(occ_map, path_ij):
    if not DO_LOS_THINNING or len(path_ij) <= 2:
        return path_ij
    thinned = [path_ij[0]]
    anchor_idx = 0
    k = 2
    while k < len(path_ij):
        if los_free(occ_map, path_ij[anchor_idx], path_ij[k]):
            k += 1
        else:
            thinned.append(path_ij[k-1])
            anchor_idx = k-1
            k += 1
    thinned.append(path_ij[-1])
    return thinned

# ===================== Build world visuals =====================
def colorize(prim, rgb):
    try:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    except Exception:
        pass

world = World()
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")

# Ground plane visual (centered at origin, spans [-W/2,W/2]x[-H/2,H/2])
gp = UsdGeom.Cube.Define(stage, "/World/GroundVisual")
gp.CreateSizeAttr(1.0)
UsdGeom.Xformable(gp.GetPrim()).AddScaleOp().Set(Gf.Vec3f(MAP_W, MAP_H, 0.05))
UsdGeom.Xformable(gp.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.03))
colorize(gp.GetPrim(), (0.25,0.25,0.25))

# Obstacles (visuals)
for k, (xmin,ymin,xmax,ymax) in enumerate(OBST_AABBS):
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    sx, sy = abs(xmax-xmin), abs(ymax-ymin)
    p = UsdGeom.Cube.Define(stage, f"/World/Obs_{k}")
    p.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(p.GetPrim())
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, 0.5))
    xf.AddTranslateOp().Set(Gf.Vec3f(cx, cy, 0.25))
    colorize(p.GetPrim(), (0.8,0.2,0.2))

# Goal (visual)
goal = UsdGeom.Sphere.Define(stage, "/World/Goal")
goal.CreateRadiusAttr(0.12)
UsdGeom.Xformable(goal.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(GOAL_XY[0], GOAL_XY[1], 0.12))
colorize(goal.GetPrim(), (0.2,0.9,0.2))

# Agent (visual)
agent = UsdGeom.Cube.Define(stage, "/World/Agent")
agent.CreateSizeAttr(AGENT_SIZE)
axf = UsdGeom.Xformable(agent.GetPrim())
axf.AddTranslateOp().Set(Gf.Vec3f(START_XY[0], START_XY[1], AGENT_SIZE/2))
colorize(agent.GetPrim(), (0.2,0.5,0.9))

# ======================== Plan once (static) ========================
start_ij = world_to_grid_idx(START_XY)
goal_ij  = world_to_grid_idx(GOAL_XY)

if not valid_cell(occ_infl, start_ij):
    raise ValueError(f"Start {START_XY} maps to blocked/out-of-bounds cell {start_ij}.")
if not valid_cell(occ_infl, goal_ij):
    raise ValueError(f"Goal  {GOAL_XY} maps to blocked/out-of-bounds cell {goal_ij}.")

path_ij = astar(occ_infl, start_ij, goal_ij)
if path_ij is None:
    print("No path found — adjust obstacles/start/goal/resolution.")
    simulation_app.close(); raise SystemExit

path_ij = thin_path_grid(occ_infl, path_ij)

waypoints = np.array([grid_to_world_center(p) for p in path_ij], dtype=float)
print(f"[A*] Planned {len(waypoints)} waypoints.")

# Visualize waypoints as tiny spheres (to confirm clearance)
for n, (wx, wy) in enumerate(waypoints):
    s = UsdGeom.Sphere.Define(stage, f"/World/WP_{n}")
    s.CreateRadiusAttr(0.03)
    UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(wx, wy, 0.03))
    colorize(s.GetPrim(), (0.1, 0.8, 0.95))

# ======================== Follow the path ========================
def step_polyline(pos, wpts, step):
    if len(wpts) == 0:
        return pos, wpts
    tgt = wpts[0]
    dv = tgt - pos
    d = float(np.linalg.norm(dv))
    if d <= step:
        pos = tgt
        wpts = wpts[1:]
    else:
        pos = pos + (dv / (d + 1e-9)) * step
    return pos, wpts

fixed_dt = world.get_physics_dt() or 1.0/60.0
xf_agent = UsdGeom.Xformable(agent.GetPrim())
xf_agent.ClearXformOpOrder()
xop_agent = xf_agent.AddTransformOp()

pos_xy = np.array(START_XY, dtype=float)
remaining = waypoints.copy()

for _ in range(20000):
    dt = fixed_dt
    pos_xy, remaining = step_polyline(pos_xy, remaining, AGENT_SPEED * dt)

    # orient toward motion direction
    yaw = 0.0
    if len(remaining) > 0:
        d = remaining[0] - pos_xy
        if np.linalg.norm(d) > 1e-6:
            yaw = math.atan2(d[1], d[0])

    m = Gf.Matrix4d().SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,0,1), math.degrees(yaw)))
    m.SetTranslateOnly(Gf.Vec3d(pos_xy[0], pos_xy[1], AGENT_SIZE/2))
    xop_agent.Set(m)

    world.step(render=True)

    if np.linalg.norm(pos_xy - np.array(GOAL_XY)) < GOAL_TOL or len(remaining) == 0:
        print("[DONE] Reached goal (within tolerance).")
        try:
            while True:
                world.step(render=True)
        except KeyboardInterrupt:
            print("Exiting on user request.")
        break

simulation_app.close()
