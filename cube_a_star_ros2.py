#!/usr/bin/env python3
# cube_a_star_ros2_usd.py
# Use Isaac Sim's Python to run:  ~/isaacsim/python.sh cube_a_star_ros2_usd.py
import math
import heapq
from typing import List, Tuple, Optional

# ---------- Isaac / Omniverse ----------
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.isaac.core import World
from pxr import Usd, UsdGeom, Gf

# ---------- ROS 2 ----------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

# ========== Config ==========
MAP_W, MAP_H = 20.0, 20.0         # meters (ground visual only)
RES = 0.10                        # grid resolution (m)
GRID_ORIGIN = (-MAP_W/2, -MAP_H/2)  # world XY of grid cell (0,0)
AGENT_SIZE = 0.5                  # cube edge (m)
START_XY = (-3.0, -3.0)
GOAL_XY  = ( 3.0,  3.0)
AGENT_SPEED = 0.4                 # m/s
ALLOW_DIAG = True
OCC_THRESH = 50                   # cells >= this are considered occupied

# Obstacles as AABBs in world XY (xmin, ymin, xmax, ymax)
OBST_AABBS = [
    (-1.5, -2.0,  1.5, -1.0),
    (-2.0,  1.0, -1.0,  4.0),
    ( 1.0,  1.1,  3.0,  1.9),
]

# ========== Simple color helper (USD visuals) ==========
def colorize(prim, rgb):
    try:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    except Exception:
        pass

# ========== Grid + A* ==========
GridCoord = Tuple[int, int]

class AStarGrid:
    def __init__(self, width_m: float, height_m: float, res: float, origin_xy: Tuple[float, float]):
        self.res = float(res)
        self.w_cells = int(round(width_m / res))
        self.h_cells = int(round(height_m / res))
        self.origin = (float(origin_xy[0]), float(origin_xy[1]))
        # -1 free, >=100 occupied, [0..99] cost
        self.data = [-1] * (self.w_cells * self.h_cells)

    def idx(self, g: GridCoord) -> int:
        return g[1] * self.w_cells + g[0]

    def in_bounds(self, g: GridCoord) -> bool:
        x, y = g
        return 0 <= x < self.w_cells and 0 <= y < self.h_cells

    def is_free(self, g: GridCoord, occ_thresh=OCC_THRESH) -> bool:
        return self.data[self.idx(g)] < occ_thresh and self.data[self.idx(g)] >= -1

    def world_to_grid(self, x: float, y: float) -> GridCoord:
        gx = int((x - self.origin[0]) / self.res)
        gy = int((y - self.origin[1]) / self.res)
        return (gx, gy)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = self.origin[0] + (gx + 0.5) * self.res
        y = self.origin[1] + (gy + 0.5) * self.res
        return (x, y)

    def mark_box_obstacle(self, xmin: float, ymin: float, xmax: float, ymax: float, val: int = 100):
        gx0, gy0 = self.world_to_grid(xmin, ymin)
        gx1, gy1 = self.world_to_grid(xmax, ymax)
        gx0, gx1 = min(gx0, gx1), max(gx0, gx1)
        gy0, gy1 = min(gy0, gy1), max(gy0, gy1)
        for gy in range(max(0, gy0), min(self.h_cells, gy1 + 1)):
            for gx in range(max(0, gx0), min(self.w_cells, gx1 + 1)):
                self.data[self.idx((gx, gy))] = val

    @staticmethod
    def heuristic(a: GridCoord, b: GridCoord) -> float:
        dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
        D, D2 = 1.0, math.sqrt(2.0)
        return D*(dx+dy) + (D2-2*D)*min(dx,dy)

    def neighbors(self, g: GridCoord, allow_diag=ALLOW_DIAG):
        x, y = g
        steps4 = [(1,0),(-1,0),(0,1),(0,-1)]
        stepsd = [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx, dy in (steps4 + stepsd if allow_diag else steps4):
            n = (x+dx, y+dy)
            if not self.in_bounds(n) or not self.is_free(n):
                continue
            yield n, (math.sqrt(2.0) if dx and dy else 1.0)

    def plan(self, start_xy: Tuple[float,float], goal_xy: Tuple[float,float]) -> List[Tuple[float,float]]:
        s = self.world_to_grid(*start_xy)
        g = self.world_to_grid(*goal_xy)
        if not (self.in_bounds(s) and self.in_bounds(g) and self.is_free(s) and self.is_free(g)):
            return []
        openq = []
        heapq.heappush(openq, (0.0, s))
        came = {}
        gsc = {s: 0.0}
        fsc = {s: self.heuristic(s, g)}
        closed = set()
        while openq:
            _, cur = heapq.heappop(openq)
            if cur in closed: 
                continue
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return [self.grid_to_world(px, py) for (px, py) in path]
            closed.add(cur)
            for nbr, step in self.neighbors(cur):
                tentative = gsc[cur] + step
                if nbr in gsc and tentative >= gsc[nbr]:
                    continue
                came[nbr] = cur
                gsc[nbr] = tentative
                fsc[nbr] = tentative + self.heuristic(nbr, g)
                heapq.heappush(openq, (fsc[nbr], nbr))
        return []

# ========== ROS bridge node ==========
class ROSBridge(Node):
    def __init__(self, global_frame='map', base_frame='base_link',
                 plan_topic='/plan', goal_topic='/goal_pose'):
        super().__init__('cube_a_star_bridge')
        self.global_frame = global_frame
        self.base_frame = base_frame
        self.plan_pub = self.create_publisher(Path, plan_topic, 10)
        self.goal_sub = self.create_subscription(PoseStamped, goal_topic, self._on_goal, 10)
        self.tf_br = TransformBroadcaster(self)
        self._latest_goal: Optional[Tuple[float,float]] = None

    def _on_goal(self, msg: PoseStamped):
        gx, gy = float(msg.pose.position.x), float(msg.pose.position.y)
        self._latest_goal = (gx, gy)
        self.get_logger().info(f"Goal received: ({gx:.2f}, {gy:.2f})")

    def pop_goal(self):
        g, self._latest_goal = self._latest_goal, None
        return g

    def publish_path(self, waypoints_xy: List[Tuple[float,float]]):
        if not waypoints_xy:
            return
        path = Path()
        path.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.global_frame)
        for i, (x, y) in enumerate(waypoints_xy):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = Point(x=float(x), y=float(y), z=0.0)
            nx, ny = waypoints_xy[i+1] if i+1 < len(waypoints_xy) else (x, y)
            yaw = math.atan2(ny - y, nx - x) if (nx != x or ny != y) else 0.0
            q = Gf.Quatf(math.cos(yaw/2.0), 0.0, 0.0, math.sin(yaw/2.0))
            ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            path.poses.append(ps)
        self.plan_pub.publish(path)

    def publish_tf(self, x: float, y: float, yaw: float = 0.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.global_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        q = Gf.Quatf(math.cos(yaw/2.0), 0.0, 0.0, math.sin(yaw/2.0))
        t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.tf_br.sendTransform(t)

# ========== Build stage with USD visuals (your style) ==========
world = World()
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")

# Ground plane visual
gp = UsdGeom.Cube.Define(stage, "/World/GroundVisual")
gp.CreateSizeAttr(1.0)
UsdGeom.Xformable(gp.GetPrim()).AddScaleOp().Set(Gf.Vec3f(MAP_W, MAP_H, 0.05))
UsdGeom.Xformable(gp.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.03))
colorize(gp.GetPrim(), (0.25, 0.25, 0.25))

# Obstacles (visuals)
for k, (xmin, ymin, xmax, ymax) in enumerate(OBST_AABBS):
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    sx, sy = abs(xmax-xmin), abs(ymax-ymin)
    p = UsdGeom.Cube.Define(stage, f"/World/Obs_{k}")
    p.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(p.GetPrim())
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, 0.5))
    xf.AddTranslateOp().Set(Gf.Vec3f(cx, cy, 0.25))
    colorize(p.GetPrim(), (0.8, 0.2, 0.2))

# Goal (visual)
goal_prim = UsdGeom.Sphere.Define(stage, "/World/Goal")
goal_prim.CreateRadiusAttr(0.12)
UsdGeom.Xformable(goal_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(GOAL_XY[0], GOAL_XY[1], 0.12))
colorize(goal_prim.GetPrim(), (0.2, 0.9, 0.2))

# Agent (visual)
agent_prim = UsdGeom.Cube.Define(stage, "/World/Agent")
agent_prim.CreateSizeAttr(AGENT_SIZE)
axf = UsdGeom.Xformable(agent_prim.GetPrim())
axf.AddTranslateOp().Set(Gf.Vec3f(START_XY[0], START_XY[1], AGENT_SIZE/2))
colorize(agent_prim.GetPrim(), (0.2, 0.5, 0.9))

world.reset() # reset to initial state
world.play()
# ========== Helpers to get/set the USD agent pose ==========
def get_agent_xy() -> Tuple[float, float]:
    xf = UsdGeom.Xformable(agent_prim.GetPrim())
    res = xf.GetLocalTransformation()
    m = res[0] if isinstance(res, tuple) else res
    p = m.ExtractTranslation()
    return float(p[0]), float(p[1])

def set_agent_xy(x: float, y: float):
    api = UsdGeom.XformCommonAPI(agent_prim.GetPrim())
    api.SetTranslate(Gf.Vec3d(float(x), float(y), float(AGENT_SIZE/2)),
                     Usd.TimeCode.Default())

def set_goal_xy(x: float, y: float):
    api = UsdGeom.XformCommonAPI(goal_prim.GetPrim())
    api.SetTranslate(Gf.Vec3d(float(x), float(y), 0.12),
                     Usd.TimeCode.Default())
# ========== Build planning grid from AABBs ==========
grid = AStarGrid(MAP_W, MAP_H, RES, GRID_ORIGIN)
for (xmin, ymin, xmax, ymax) in OBST_AABBS:
    grid.mark_box_obstacle(xmin, ymin, xmax, ymax, val=100)

# ========== ROS init ==========
if not rclpy.ok():
    rclpy.init(args=None)
ros = ROSBridge(global_frame="map", base_frame="base_link",
                plan_topic="/plan", goal_topic="/goal_pose")

# ========== Initial plan ==========
agent_xy = START_XY
goal_xy  = GOAL_XY
path_xy: List[Tuple[float,float]] = grid.plan(agent_xy, goal_xy)
ros.publish_path(path_xy)
path_i = 0
print("[A*] Ready. Publish /goal_pose to replan. Viewing /plan in RViz.")
print()
# ========== Main loop ==========
try:
    while simulation_app.is_running():
        # Sim step time
        dt = world.get_physics_dt()

        # Pump ROS callbacks
        rclpy.spin_once(ros, timeout_sec=0.0)

        # New goal?
        g = ros.pop_goal()
        if g is not None:
            goal_xy = (g[0], g[1])
            set_goal_xy(goal_xy[0], goal_xy[1])
            agent_xy = get_agent_xy()
            path_xy = grid.plan(agent_xy, goal_xy)
            ros.publish_path(path_xy)
            path_i = 0

        # Follow path (simple P-controller to next waypoint)
        if path_xy and path_i < len(path_xy):
            ax, ay = get_agent_xy()
            tx, ty = path_xy[path_i]
            dx, dy = tx - ax, ty - ay
            dist = math.hypot(dx, dy)
            step = AGENT_SPEED * dt
            if dist < max(0.02, step):
                path_i += 1
            else:
                nx = ax + (dx / max(dist, 1e-6)) * step
                ny = ay + (dy / max(dist, 1e-6)) * step
                set_agent_xy(nx, ny)
                agent_xy = (nx, ny)
        else:
            agent_xy = get_agent_xy()

        # Publish TF (map -> base_link) with a crude yaw along path
        yaw = 0.0
        if path_xy and path_i < len(path_xy):
            ax, ay = agent_xy
            tx, ty = path_xy[min(path_i, len(path_xy)-1)]
            if (tx != ax) or (ty != ay):
                yaw = math.atan2(ty - ay, tx - ax)
        ros.publish_tf(agent_xy[0], agent_xy[1], yaw)

        world.step(render=True)

except Exception as e:
    print("Exception:", e)
    pass
finally:
    try:
        ros.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass
    simulation_app.close()
# ========== END ==========