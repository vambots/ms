#!/usr/bin/env python3
# turtlebot_a_star_keys_ddrive_rclpy.py
import math, heapq, os
from typing import List, Tuple, Optional

# ---------- Isaac / Omniverse ----------
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from omni.isaac.core import World
from pxr import Usd, UsdGeom, Gf, Sdf

# Keyboard input (Omniverse Kit)
import omni.appwindow
import carb.input as carb_input

# ---------- ROS 2 (pure rclpy) ----------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped, Twist
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header, String

# ================== Config ==================
MAP_W, MAP_H = 20.0, 20.0
RES = 0.10
GRID_ORIGIN = (-MAP_W/2, -MAP_H/2)
START_XY = (-3.0, -3.0)
GOAL_XY  = ( 3.0,  3.0)
ALLOW_DIAG = False      # changed: never diagonal for x-theta
OCC_THRESH = 50

# ---------- X–THETA CONSTRAINTS ----------
LOCK_Y = True
LOCK_Y_VAL = START_XY[1]   # y stays fixed to this value

# Differential-drive tuning
MAX_V_MANUAL = 1.2
MAX_W_MANUAL = 2.0
AGENT_SPEED  = 0.6
MAX_W_AUTO   = 1.5
ANG_KP       = 2.5
ARRIVE_DIST  = 0.10
TWIST_TIMEOUT = 0.5

# Obstacles (world XY AABBs)
OBST_AABBS = [
    (-1.5, -2.0,  1.5, -1.0),
    (-2.0,  1.0, -1.0,  4.0),
    ( 1.0,  1.1,  3.0,  1.9),
]

# Your TurtleBot USD + prim
LOCAL_TURTLEBOT_USD = "/home/GTL/sgambhir/tmp_turtlebot3_src/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger/turtlebot3_burger.usd"
TARGET_PRIM_IN_FILE = "/turtlebot3_burger"
TURTLE_MOVE_PRIM  = "/World/TurtleRoot"
TURTLE_CHILD_PRIM = "/World/TurtleRoot/Model"
z_height = 0.05
BOT_SCALE = 5

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
        self.data = [-1] * (self.w_cells * self.h_cells)

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
        # Only x-distance matters if we lock y
        dx = abs(a[0]-b[0])
        return dx

    def neighbors(self, g, allow_diag=False):
        x, y = g
        # ---------- Only +/- x moves (no y change) ----------
        steps = [(1,0), (-1,0)]
        for dx, dy in steps:
            n = (x+dx, y+dy)
            if not self.in_bounds(n) or not self.is_free(n):
                continue
            yield n, 1.0

    def plan(self, start_xy, goal_xy):
        # Clamp both start/goal to the locked y-row if enabled
        sx, sy = start_xy
        gx, gy = goal_xy
        if LOCK_Y:
            sy = LOCK_Y_VAL
            gy = LOCK_Y_VAL
        s = self.world_to_grid(sx, sy)
        g = self.world_to_grid(gx, gy)
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
                wpts = [self.grid_to_world(px, py) for (px, py) in path]
                # Re-clamp y in world, for numerical safety
                if LOCK_Y:
                    wpts = [(wx, LOCK_Y_VAL) for (wx, _) in wpts]
                return wpts
            closed.add(cur)
            for nbr, step in self.neighbors(cur, allow_diag=False):
                tentative = gsc[cur] + step
                if nbr in gsc and tentative >= gsc[nbr]: continue
                came[nbr] = cur; gsc[nbr] = tentative
                fsc[nbr] = tentative + self.heuristic(nbr, g)
                heapq.heappush(openq, (fsc[nbr], nbr))
        return []

# ===== ROS bridge =====
class ROSBridge(Node):
    def __init__(self, global_frame='map', base_frame='base_link', plan_topic='/plan',
                 on_set_goal=None, on_plan=None, on_cancel=None,
                 twist_timeout=TWIST_TIMEOUT):
        super().__init__('turtlebot_ddrive_bridge')
        self.global_frame = global_frame
        self.base_frame   = base_frame
        self.plan_pub     = self.create_publisher(Path, plan_topic, 10)
        self.tf_br        = TransformBroadcaster(self)
        self.on_set_goal  = on_set_goal
        self.on_plan      = on_plan
        self.on_cancel    = on_cancel

        self.create_subscription(Twist, '/cmd_vel', self._twist_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self._goal_cb, 10)
        self.create_subscription(String, '/turtlebot_ctrl', self._ctrl_cb, 10)

        self._twist_v = 0.0
        self._twist_w = 0.0
        self._last_twist = None
        self._twist_timeout = float(twist_timeout)

    def publish_path(self, waypoints_xy: List[Tuple[float,float]]):
        path = Path()
        path.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.global_frame)
        for (x, y) in waypoints_xy:
            ps = PoseStamped()
            ps.header = path.header
            if LOCK_Y: y = LOCK_Y_VAL
            ps.pose.position = Point(x=float(x), y=float(y), z=0.0)
            ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            path.poses.append(ps)
        self.plan_pub.publish(path)

    def publish_tf(self, x: float, y: float, yaw: float = 0.0):
        if LOCK_Y: y = LOCK_Y_VAL
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.global_frame
        t.child_frame_id  = self.base_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        half = 0.5 * float(yaw)
        t.transform.rotation = Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
        self.tf_br.sendTransform(t)

    def _twist_cb(self, msg: Twist):
        self._twist_v = float(msg.linear.x)
        self._twist_w = float(msg.angular.z)
        self._last_twist = self.get_clock().now()

    def _goal_cb(self, msg: PoseStamped):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        if LOCK_Y: y = LOCK_Y_VAL    # clamp goal y
        if self.on_set_goal:
            self.on_set_goal(x, y)
        if self.on_plan:
            self.on_plan()

    def _ctrl_cb(self, msg: String):
        try:
            tokens = msg.data.strip().split()
            if not tokens: return
            cmd = tokens[0].lower()
            if cmd == "plan" and self.on_plan:
                self.on_plan()
            elif cmd in ("cancel","stop") and self.on_cancel:
                self.on_cancel()
            elif cmd == "set_goal" and len(tokens) >= 3 and self.on_set_goal:
                x, y = float(tokens[1]), float(tokens[2])
                if LOCK_Y: y = LOCK_Y_VAL
                self.on_set_goal(x, y)
        except Exception as e:
            self.get_logger().warn(f"/turtlebot_ctrl parse failed: {e}")

    def get_active_twist(self) -> Tuple[float,float,bool]:
        if self._last_twist is None:
            return 0.0, 0.0, False
        age = (self.get_clock().now() - self._last_twist).nanoseconds * 1e-9
        active = age <= self._twist_timeout
        return (self._twist_v if active else 0.0,
                self._twist_w if active else 0.0,
                active)

# ================== Stage build ==================
world = World()
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")
stage.SetEditTarget(stage.GetRootLayer())
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
assert os.path.isfile(LOCAL_TURTLEBOT_USD), f"USD not found: {LOCAL_TURTLEBOT_USD}"

# Ground
gp = UsdGeom.Cube.Define(stage, "/World/GroundVisual")
gp.CreateSizeAttr(1.0)
UsdGeom.Xformable(gp.GetPrim()).AddScaleOp().Set(Gf.Vec3f(MAP_W, MAP_H, 0.05))
UsdGeom.Xformable(gp.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0, 0, -0.03))
colorize(gp.GetPrim(), (0.25, 0.25, 0.25))



# Obstacles
for k, (xmin, ymin, xmax, ymax) in enumerate(OBST_AABBS):
    cx, cy = (xmin+xmax)/2.0, (ymin+ymax)/2.0
    sx, sy = abs(xmax-xmin), abs(ymax-ymin)
    p = UsdGeom.Cube.Define(stage, f"/World/Obs_{k}")
    p.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(p.GetPrim())
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, 0.5))
    xf.AddTranslateOp().Set(Gf.Vec3f(cx, cy, 0.25))
    colorize(p.GetPrim(), (0.8, 0.2, 0.2))

# Goal (its y will be clamped on use)
goal_prim = UsdGeom.Sphere.Define(stage, "/World/Goal")
goal_prim.CreateRadiusAttr(0.12)
UsdGeom.Xformable(goal_prim.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(GOAL_XY[0], GOAL_XY[1], 0.12))
colorize(goal_prim.GetPrim(), (0.2, 0.9, 0.2))
goal_api = UsdGeom.XformCommonAPI(goal_prim.GetPrim())
def set_goal_xy(x: float, y: float):
    global GOAL_XY
    if LOCK_Y: y = LOCK_Y_VAL
    GOAL_XY = (float(x), float(y))
    goal_api.SetTranslate(Gf.Vec3f(GOAL_XY[0], GOAL_XY[1], 0.12))
    print(f"[goal] Set to ({GOAL_XY[0]:.2f}, {GOAL_XY[1]:.2f})")

# ---------- Spawn TurtleBot ----------
TARGET_PRIM = TARGET_PRIM_IN_FILE
TURTLE_MOVE_PRIM  = "/World/TurtleRoot"
TURTLE_CHILD_PRIM = "/World/TurtleRoot/Model"
z_height = 0.0
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

def _ensure_xform(path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Imageable(prim).MakeVisible()
    return prim

try:
    root_xf  = _ensure_xform(TURTLE_MOVE_PRIM)
    child_xf = _ensure_xform(TURTLE_CHILD_PRIM)
    refs = child_xf.GetReferences()
    refs.ClearReferences()
    refs.AddReference(Sdf.Reference(assetPath=LOCAL_TURTLEBOT_USD, primPath=TARGET_PRIM))
    child_xf.Load(Usd.LoadWithDescendants)

    # place wrapper at START (y clamped if needed)
    init_y = LOCK_Y_VAL if LOCK_Y else START_XY[1]
    UsdGeom.XformCommonAPI(root_xf).SetTranslate(Gf.Vec3d(float(START_XY[0]), float(init_y), z_height))
    root_api = UsdGeom.XformCommonAPI(root_xf)
    root_api.SetScale(Gf.Vec3f(BOT_SCALE, BOT_SCALE, BOT_SCALE))

    kids = list(child_xf.GetChildren())
    print(f"[INFO] Referenced {LOCAL_TURTLEBOT_USD} @ {TARGET_PRIM} -> {TURTLE_CHILD_PRIM}")
    print(f"[INFO] Child count under Model: {len(kids)}")
    for c in kids[:12]:
        print("   [DEBUG]", c.GetPath())

except Exception as e:
    print("[WARN] Could not reference TurtleBot; using a cylinder placeholder. Error:", e)
    cyl = UsdGeom.Cylinder.Define(stage, f"{TURTLE_MOVE_PRIM}/Cylinder")
    cyl.CreateHeightAttr(0.2)
    cyl.CreateRadiusAttr(0.18)
    UsdGeom.Imageable(cyl.GetPrim()).MakeVisible()
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath(TURTLE_MOVE_PRIM)).SetTranslate(
        Gf.Vec3d(float(START_XY[0]), float(LOCK_Y_VAL if LOCK_Y else START_XY[1]), z_height)
    )

def _ensure_srt_ops(prim: Usd.Prim):
    xf = UsdGeom.Xformable(prim)
    sc = next((op for op in xf.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeScale), None)
    rz = next((op for op in xf.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeRotateZ), None)
    tr = next((op for op in xf.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate), None)
    if sc is None: sc = xf.AddScaleOp()
    if rz is None: rz = xf.AddRotateZOp()
    if tr is None: tr = xf.AddTranslateOp()
    xf.SetXformOpOrder([sc, rz, tr])
    return sc, rz, tr

# ================== Pose helpers ==================
def _get_srt():
    prim = stage.GetPrimAtPath(TURTLE_MOVE_PRIM)
    return _ensure_srt_ops(prim)

def get_agent_pose() -> Tuple[float, float, float]:
    _, rz, tr = _get_srt()
    t = tr.Get()
    if t is None:
        t = Gf.Vec3f(float(START_XY[0]), float(LOCK_Y_VAL if LOCK_Y else START_XY[1]), float(z_height))
        tr.Set(t)
    rz_val = rz.Get()
    if rz_val is None:
        rz_val = 0.0
        rz.Set(rz_val)
    yaw = math.radians(float(rz_val))
    x = float(t[0])
    y = LOCK_Y_VAL if LOCK_Y else float(t[1])
    return x, y, float(yaw)

def set_agent_pose(x: float, y: float, yaw: float):
    if LOCK_Y: y = LOCK_Y_VAL
    sc, rz, tr = _get_srt()
    rz.Set(math.degrees(float(yaw)))
    tr.Set(Gf.Vec3f(float(x), float(y), float(z_height)))

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

# ================== Planning grid ==================
grid = AStarGrid(MAP_W, MAP_H, RES, GRID_ORIGIN)
for (xmin, ymin, xmax, ymax) in OBST_AABBS:
    grid.mark_box_obstacle(xmin, ymin, xmax, ymax, val=100)

# ================== ROS init ==================
if not rclpy.ok():
    rclpy.init(args=None)

def _on_set_goal(x, y): set_goal_xy(x, y)
def _on_plan(): trigger_plan_and_follow()
def _on_cancel(): cancel_follow()

ros = ROSBridge(
    global_frame="map", base_frame="base_link", plan_topic="/plan",
    on_set_goal=_on_set_goal, on_plan=_on_plan, on_cancel=_on_cancel,
    twist_timeout=TWIST_TIMEOUT
)

# ================== Keyboard handling ==================
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
    sx, sy, _ = get_agent_pose()
    path_xy = grid.plan((sx, sy), GOAL_XY)
    ros.publish_path(path_xy)
    path_i = 0; following = bool(path_xy)
    print(f"[A*] Planned {len(path_xy)} waypoints." if path_xy else "[A*] No path (row blocked?).")

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

# ================== Driver state ==================
path_xy: List[Tuple[float,float]] = []
path_i = 0
following = False

print("[A*] Ready. Keys: ↑/↓ = +x/-x, ←/→ = rotate. Enter = plan+follow; Space = stop.")
print("[ROS] Send /cmd_vel (uses x only) or /move_base_simple/goal (y is clamped).")

# ================== Main loop ==================
try:
    x, y, yaw = get_agent_pose()
    while simulation_app.is_running():
        dt = world.get_physics_dt()
        rclpy.spin_once(ros, timeout_sec=0.0)

        if key_state["enter"]:
            trigger_plan_and_follow(); key_state["enter"] = False
        if key_state["space"]:
            cancel_follow(); key_state["space"] = False

        keys_active = key_state["up"] or key_state["down"] or key_state["left"] or key_state["right"]
        v_tw, w_tw, tw_active = ros.get_active_twist()

        if keys_active:
            # ---------- Manual: x-only + yaw ----------
            v_cmd = MAX_V_MANUAL * (float(key_state["up"]) - float(key_state["down"]))
            w_cmd = MAX_W_MANUAL * (float(key_state["left"]) - float(key_state["right"]))
            x  += v_cmd * dt                 # x-only
            yaw = _wrap_pi(yaw + w_cmd * dt)
            set_agent_pose(x, y, yaw)

        elif tw_active:
            # ---------- /cmd_vel: use linear.x for x-only, angular.z for yaw ----------
            x  += v_tw * dt
            yaw = _wrap_pi(yaw + w_tw * dt)
            set_agent_pose(x, y, yaw)

        elif following and path_xy and path_i < len(path_xy):
            # ---------- A* follower along x-only ----------
            tx, _ty = path_xy[path_i]
            dx = tx - x
            dist = abs(dx)

            # Point yaw either 0 (positive x) or pi (negative x) to align with x-axis
            desired_yaw = 0.0 if dx >= 0.0 else math.pi
            err_yaw = _wrap_pi(desired_yaw - yaw)
            w = max(-MAX_W_AUTO, min(MAX_W_AUTO, ANG_KP * err_yaw))

            # Gate linear motion on heading alignment
            heading_ok = abs(err_yaw) < 0.2
            v = AGENT_SPEED if heading_ok else 0.0

            if dist < max(ARRIVE_DIST, AGENT_SPEED * dt):
                path_i += 1
                if path_i >= len(path_xy):
                    following = False
            else:
                x  += v * dt
                yaw = _wrap_pi(yaw + w * dt)
                set_agent_pose(x, y, yaw)

        ros.publish_tf(x, y, yaw)
        world.step(render=True)

except Exception as e:
    print("Exception:", e)
finally:
    try:
        if kb_subscription:
            input_iface.unsubscribe_to_keyboard_events(keyboard, _on_key_event)
    except Exception:
        pass
    try:
        ros.destroy_node(); rclpy.shutdown()
    except Exception:
        pass
    simulation_app.close()
# ================== END ==================
