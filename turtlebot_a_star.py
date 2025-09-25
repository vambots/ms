#!/usr/bin/env python3
# turtlebot_a_star_keys_wrapper.py
import math, heapq
from typing import List, Tuple

# ---------- Isaac / Omniverse ----------
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.isaac.core import World
from pxr import Usd, UsdGeom, Gf

# Keyboard input (Omniverse Kit)
import omni.appwindow
import carb.input as carb_input

# ---------- ROS 2 ----------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

# ========== Config ==========
MAP_W, MAP_H = 20.0, 20.0
RES = 0.10
GRID_ORIGIN = (-MAP_W/2, -MAP_H/2)
START_XY = (-3.0, -3.0)
GOAL_XY  = ( 3.0,  3.0)
AGENT_SPEED = 0.4
MANUAL_SPEED = 1.2
ALLOW_DIAG = True
OCC_THRESH = 50

# Obstacles as AABBs in world XY (xmin, ymin, xmax, ymax)
OBST_AABBS = [
    (-1.5, -2.0,  1.5, -1.0),
    (-2.0,  1.0, -1.0,  4.0),
    ( 1.0,  1.1,  3.0,  1.9),
]

# Your local USD path (from your earlier log)
LOCAL_TURTLEBOT_USD = "/home/GTL/sgambhir/IsaacAssets/Turtlebot3/turtlebot3_burger.usd"
LOCAL_TURTLEBOT_USD = "/home/GTL/sgambhir/tmp_turtlebot3_src/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger/turtlebot3_burger.usd"

# We will move this wrapper (pure Xform). The robot USD sits as its child.
TURTLE_MOVE_PRIM  = "/World/TurtleRoot"        # we move/read this
TURTLE_CHILD_PRIM = "/World/TurtleRoot/Model"  # holds the USD reference
z_height = 0.20

# ===== Helper for coloring simple USD prims =====
def colorize(prim, rgb):
    try:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    except Exception:
        pass

# ===== Grid + A* =====
GridCoord = Tuple[int, int]

class AStarGrid:
    def __init__(self, width_m, height_m, res, origin_xy):
        self.res = float(res)
        self.w_cells = int(round(width_m / res))
        self.h_cells = int(round(height_m / res))
        self.origin = (float(origin_xy[0]), float(origin_xy[1]))
        self.data = [-1] * (self.w_cells * self.h_cells)  # -1 free, >=100 occupied

    def idx(self, g): return g[1] * self.w_cells + g[0]
    def in_bounds(self, g): return 0 <= g[0] < self.w_cells and 0 <= g[1] < self.h_cells
    def is_free(self, g, occ_thresh=OCC_THRESH):
        d = self.data[self.idx(g)]
        return (-1 <= d) and (d < occ_thresh)

    def world_to_grid(self, x, y):
        gx = int((x - self.origin[0]) / self.res)
        gy = int((y - self.origin[1]) / self.res)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        x = self.origin[0] + (gx + 0.5) * self.res
        y = self.origin[1] + (gy + 0.5) * self.res
        return (x, y)

    def mark_box_obstacle(self, xmin, ymin, xmax, ymax, val=100):
        gx0, gy0 = self.world_to_grid(xmin, ymin)
        gx1, gy1 = self.world_to_grid(xmax, ymax)
        gx0, gx1 = min(gx0, gx1), max(gx0, gx1)
        gy0, gy1 = min(gy0, gy1), max(gy0, gy1)
        for gy in range(max(0, gy0), min(self.h_cells, gy1 + 1)):
            for gx in range(max(0, gx0), min(self.w_cells, gx1 + 1)):
                self.data[self.idx((gx, gy))] = val

    @staticmethod
    def heuristic(a, b):
        dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
        D, D2 = 1.0, math.sqrt(2.0)
        return D*(dx+dy) + (D2-2*D)*min(dx,dy)

    def neighbors(self, g, allow_diag=ALLOW_DIAG):
        x, y = g
        steps4 = [(1,0),(-1,0),(0,1),(0,-1)]
        stepsd = [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx, dy in (steps4 + stepsd if allow_diag else steps4):
            n = (x+dx, y+dy)
            if not self.in_bounds(n) or not self.is_free(n):
                continue
            yield n, (math.sqrt(2.0) if dx and dy else 1.0)

    def plan(self, start_xy, goal_xy):
        s = self.world_to_grid(*start_xy)
        g = self.world_to_grid(*goal_xy)
        if not (self.in_bounds(s) and self.in_bounds(g) and self.is_free(s) and self.is_free(g)):
            return []
        openq = []
        heapq.heappush(openq, (0.0, s))
        came, gsc, fsc, closed = {}, {s: 0.0}, {s: self.heuristic(s, g)}, set()
        while openq:
            _, cur = heapq.heappop(openq)
            if cur in closed: continue
            if cur == g:
                path = [cur]
                while cur in came:
                    cur = came[cur]; path.append(cur)
                path.reverse()
                return [self.grid_to_world(px, py) for (px, py) in path]
            closed.add(cur)
            for nbr, step in self.neighbors(cur):
                tentative = gsc[cur] + step
                if nbr in gsc and tentative >= gsc[nbr]: continue
                came[nbr] = cur; gsc[nbr] = tentative
                fsc[nbr] = tentative + self.heuristic(nbr, g)
                heapq.heappush(openq, (fsc[nbr], nbr))
        return []

# ===== ROS node (publish /plan and tf) =====
class ROSBridge(Node):
    def __init__(self, global_frame='map', base_frame='base_link', plan_topic='/plan'):
        super().__init__('turtlebot_a_star_bridge')
        self.global_frame = global_frame
        self.base_frame = base_frame
        self.plan_pub = self.create_publisher(Path, plan_topic, 10)
        self.tf_br = TransformBroadcaster(self)

    def publish_path(self, waypoints_xy: List[Tuple[float,float]]):
        if not waypoints_xy: return
        path = Path()
        path.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.global_frame)
        for (x, y) in waypoints_xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = Point(x=float(x), y=float(y), z=0.0)
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
        t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.tf_br.sendTransform(t)

# ===== Build stage visuals =====
world = World()
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")

gp = UsdGeom.Cube.Define(stage, "/World/GroundVisual")
gp.CreateSizeAttr(1.0)
UsdGeom.Xformable(gp.GetPrim()).AddScaleOp().Set(Gf.Vec3f(MAP_W, MAP_H, 0.05))
UsdGeom.Xformable(gp.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.03))
colorize(gp.GetPrim(), (0.25, 0.25, 0.25))

for k, (xmin, ymin, xmax, ymax) in enumerate(OBST_AABBS):
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    sx, sy = abs(xmax-xmin), abs(ymax-ymin)
    p = UsdGeom.Cube.Define(stage, f"/World/Obs_{k}")
    p.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(p.GetPrim())
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, 0.5))
    xf.AddTranslateOp().Set(Gf.Vec3f(cx, cy, 0.25))
    colorize(p.GetPrim(), (0.8, 0.2, 0.2))

goal_prim = UsdGeom.Sphere.Define(stage, "/World/Goal")
goal_prim.CreateRadiusAttr(0.12)
UsdGeom.Xformable(goal_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(GOAL_XY[0], GOAL_XY[1], 0.12))
colorize(goal_prim.GetPrim(), (0.2, 0.9, 0.2))

# --- Ensure lighting (so you see the robot) ---
# light = UsdGeom.DistantLight.Define(stage, "/World/DebugLight")
# light.CreateIntensityAttr(3000.0)

# --- Put a camera looking at the bot spawn ---
# cam = UsdGeom.Camera.Define(stage, "/World/DebugCamera")
# cam_x, cam_y, cam_z = START_XY[0]-4.0, START_XY[1]-4.0, 2.0
# UsdGeom.XformCommonAPI(cam).SetTranslate(Gf.Vec3d(cam_x, cam_y, cam_z))
# look_at = Gf.Vec3d(float(START_XY[0]), float(START_XY[1]), z_height)
# pos = Gf.Vec3d(cam_x, cam_y, cam_z)
# forward = (look_at - pos).GetNormalized()
# right = Gf.Vec3d.Cross(Gf.Vec3d(0,0,1), forward).GetNormalized()
# up = Gf.Vec3d.Cross(forward, right).GetNormalized()
# m = Gf.Matrix4d(
#     right[0], right[1], right[2], 0.0,
#     up[0],    up[1],    up[2],    0.0,
#     forward[0], forward[1], forward[2], 0.0,
#     pos[0], pos[1], pos[2], 1.0
# )
# UsdGeom.Xformable(cam.GetPrim()).SetXformMatrix(m)
# print("[INFO] Switch viewport to /World/DebugCamera if needed.")

# ---------- Spawn TurtleBot using a wrapper Xform ----------
# ---------- Spawn TurtleBot (wrapper Xform + explicit primPath + payload load) ----------
from pxr import Usd, UsdGeom, Sdf

# 1) Your confirmed USD and the *root prim inside it*
LOCAL_TURTLEBOT_USD = "/home/GTL/sgambhir/tmp_turtlebot3_src/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger/turtlebot3_burger.usd"
TARGET_PRIM         = "/turtlebot3_burger"   # root prim inside that USD (from your screenshot)

# 2) We move this wrapper; the robot USD is referenced under it
TURTLE_MOVE_PRIM  = "/World/TurtleRoot"
TURTLE_CHILD_PRIM = "/World/TurtleRoot/Model"
z_height = 0.26   # raise a bit so it never clips into the ground

# (optional but recommended) make sure stage units are meters
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

def _ensure_xform(path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Imageable(prim).MakeVisible()
    return prim

try:
    # Create wrapper and child Xforms
    root_xf  = _ensure_xform(TURTLE_MOVE_PRIM)
    child_xf = _ensure_xform(TURTLE_CHILD_PRIM)

    # Clear any previous reference and attach the TurtleBot at the explicit prim
    refs = child_xf.GetReferences()
    refs.ClearReferences()
    refs.AddReference(Sdf.Reference(assetPath=LOCAL_TURTLEBOT_USD, primPath=TARGET_PRIM))

    # Many robot USDs use payloads -> ensure they load
    child_xf.Load(Usd.LoadWithDescendants)

    # Place the wrapper at START_XY (so all children move together)
    UsdGeom.XformCommonAPI(root_xf).SetTranslate(
        Gf.Vec3d(float(START_XY[0]), float(START_XY[1]), z_height)
    )

    # Debug prints so you can verify population
    kids = list(child_xf.GetChildren())
    print(f"[INFO] Referenced {LOCAL_TURTLEBOT_USD} @ {TARGET_PRIM} -> {TURTLE_CHILD_PRIM}")
    print(f"[INFO] Child count under Model: {len(kids)}")
    for c in kids[:12]:
        print("   [DEBUG]", c.GetPath())

except Exception as e:
    # Fallback: visible placeholder under the same wrapper path
    print("[WARN] Could not reference TurtleBot; using a cylinder placeholder. Error:", e)
    cyl = UsdGeom.Cylinder.Define(stage, f"{TURTLE_MOVE_PRIM}/Cylinder")
    cyl.CreateHeightAttr(0.2)
    cyl.CreateRadiusAttr(0.18)
    UsdGeom.Imageable(cyl.GetPrim()).MakeVisible()
    UsdGeom.XformCommonAPI(root_xf).SetTranslate(
        Gf.Vec3d(float(START_XY[0]), float(START_XY[1]), z_height)
    )


# ===== Xform utilities (version-proof) =====
def _get_local_mat(prim) -> Gf.Matrix4d:
    xf = UsdGeom.Xformable(prim)
    res = xf.GetLocalTransformation()
    return res[0] if isinstance(res, tuple) else res

# ===== Helpers to get/set agent pose (use the wrapper!) =====
def get_agent_xy() -> Tuple[float, float]:
    prim = stage.GetPrimAtPath(TURTLE_MOVE_PRIM)
    mat = _get_local_mat(prim)
    return float(mat[3][0]), float(mat[3][1])

def set_agent_xy(x: float, y: float):
    prim = stage.GetPrimAtPath(TURTLE_MOVE_PRIM)
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(float(x), float(y), z_height))

# ===== Build planning grid =====
grid = AStarGrid(MAP_W, MAP_H, RES, GRID_ORIGIN)
for (xmin, ymin, xmax, ymax) in OBST_AABBS:
    grid.mark_box_obstacle(xmin, ymin, xmax, ymax, val=100)

# ===== ROS init =====
if not rclpy.ok():
    rclpy.init(args=None)
ros = ROSBridge(global_frame="map", base_frame="base_link", plan_topic="/plan")

# ===== Keyboard handling =====
key_state = { "up": False, "down": False, "left": False, "right": False, "enter": False, "space": False }

app_window = omni.appwindow.get_default_app_window()
keyboard   = app_window.get_keyboard()
input_iface = carb_input.acquire_input_interface()

def cancel_follow():
    global following, path_i
    following = False; path_i = 0
    print("[keys] Follow cancelled. Waiting for a new goal to replan.")

def trigger_plan_and_follow():
    global path_xy, path_i, following
    start_xy = get_agent_xy()
    path_xy = grid.plan(start_xy, GOAL_XY)
    ros.publish_path(path_xy)
    path_i = 0; following = bool(path_xy)
    print(f"[A*] Planned {len(path_xy)} waypoints." if path_xy else "[A*] No path.")

def _on_key_event(event: carb_input.KeyboardEvent) -> bool:
    is_press = event.type == carb_input.KeyboardEventType.KEY_PRESS
    if event.input == carb_input.KeyboardInput.UP:
        key_state["up"] = is_press
        if is_press: cancel_follow()
    elif event.input == carb_input.KeyboardInput.DOWN:
        key_state["down"] = is_press
        if is_press: cancel_follow()
    elif event.input == carb_input.KeyboardInput.LEFT:
        key_state["left"] = is_press
        if is_press: cancel_follow()
    elif event.input == carb_input.KeyboardInput.RIGHT:
        key_state["right"] = is_press
        if is_press: cancel_follow()
    elif event.input == carb_input.KeyboardInput.ENTER:
        if is_press: key_state["enter"] = True
    elif event.input == carb_input.KeyboardInput.SPACE:
        if is_press: key_state["space"] = True
    return True

kb_subscription = input_iface.subscribe_to_keyboard_events(keyboard, _on_key_event)

# ===== Planner/driver state =====
path_xy: List[Tuple[float,float]] = []
path_i = 0
following = False

ros.publish_path(path_xy)
print("[A*] Ready. Arrow keys = manual; Enter = plan+follow to green sphere; Space = stop.")

# ===== Main loop =====
try:
    while simulation_app.is_running():
        dt = world.get_physics_dt()
        rclpy.spin_once(ros, timeout_sec=0.0)

        if key_state["enter"]:
            trigger_plan_and_follow(); key_state["enter"] = False
        if key_state["space"]:
            cancel_follow(); key_state["space"] = False

        # Manual driving (moves wrapper)
        ax, ay = get_agent_xy()
        move_x = float(key_state["right"]) - float(key_state["left"])
        move_y = float(key_state["up"]) - float(key_state["down"])
        if move_x or move_y:
            norm = math.hypot(move_x, move_y)
            step = MANUAL_SPEED * dt
            nx = ax + (move_x / norm) * step
            ny = ay + (move_y / norm) * step
            set_agent_xy(nx, ny)
            ax, ay = nx, ny

        # Follow planned path
        if following and path_xy and path_i < len(path_xy):
            tx, ty = path_xy[path_i]
            dx, dy = tx - ax, ty - ay
            dist = math.hypot(dx, dy)
            step = AGENT_SPEED * dt
            if dist < max(0.02, step):
                path_i += 1
                if path_i >= len(path_xy): following = False
            else:
                nx = ax + (dx / max(dist, 1e-6)) * step
                ny = ay + (dy / max(dist, 1e-6)) * step
                set_agent_xy(nx, ny)
                ax, ay = nx, ny

        # TF at current wrapper pose
        ros.publish_tf(ax, ay, 0.0)

        world.step(render=True)

except Exception as e:
    print("Exception:", e)
finally:
    try:
        if kb_subscription:
            input_iface.unsubscribe_to_keyboard_events(keyboard, kb_subscription)
    except Exception:
        pass
    try:
        ros.destroy_node(); rclpy.shutdown()
    except Exception:
        pass
    simulation_app.close()
# ===== END =====
