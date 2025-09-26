#!/usr/bin/env python3
# turtlebot_a_star_keys_ddrive_ros.py
import math, heapq
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

# ---------- ROS 2 ----------
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped, Twist
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header, String

# ========== Config ==========
MAP_W, MAP_H = 20.0, 20.0
RES = 0.10
GRID_ORIGIN = (-MAP_W/2, -MAP_H/2)
START_XY = (-3.0, -3.0)
GOAL_XY  = ( 3.0,  3.0)
ALLOW_DIAG = True
OCC_THRESH = 50

# Manual & auto drive tuning
AGENT_SPEED   = 0.6           # m/s when following path
MAX_V_MANUAL  = 1.2           # m/s   (↑/↓)
MAX_W_MANUAL  = 2.0           # rad/s (←/→)
MAX_W_AUTO    = 1.5           # rad/s when following path
ANG_KP        = 2.5           # P-gain for heading correction
ARRIVE_DIST   = 0.10          # waypoint reach threshold (m)
TWIST_TIMEOUT = 0.5           # seconds before /cmd_vel considered inactive

# Obstacles as AABBs in world XY (xmin, ymin, xmax, ymax)
OBST_AABBS = [
    (-1.5, -2.0,  1.5, -1.0),
    (-2.0,  1.0, -1.0,  4.0),
    ( 1.0,  1.1,  3.0,  1.9),
]

# Your working TurtleBot USD and its root prim
LOCAL_TURTLEBOT_USD = "/home/GTL/sgambhir/tmp_turtlebot3_src/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger/turtlebot3_burger.usd"
TARGET_PRIM         = "/turtlebot3_burger"   # root prim inside that USD

# We move this wrapper (pure Xform). The robot USD sits as its child.
TURTLE_MOVE_PRIM  = "/World/TurtleRoot"        # we move/read this
TURTLE_CHILD_PRIM = "/World/TurtleRoot/Model"  # holds the USD reference
z_height = 0.05                                 # slight lift off ground
BOT_SCALE = 2                              # ×100 if asset was authored in cm

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

# ===== ROS node: publish /plan + tf, subscribe to cmd_vel, goal, simple control =====
class ROSBridge(Node):
    def __init__(self, global_frame='map', base_frame='base_link', plan_topic='/plan',
                 on_set_goal=None, on_plan=None, on_cancel=None, on_speed=None,
                 auto_plan_on_goal=True, twist_timeout=TWIST_TIMEOUT):
        super().__init__('turtlebot_a_star_bridge')
        self.global_frame = global_frame
        self.base_frame   = base_frame
        self.plan_pub     = self.create_publisher(Path, plan_topic, 10)
        self.tf_br        = TransformBroadcaster(self)

        # Hooks into the sim (functions provided below)
        self.on_set_goal = on_set_goal
        self.on_plan     = on_plan
        self.on_cancel   = on_cancel
        self.on_speed    = on_speed
        self.auto_plan_on_goal = auto_plan_on_goal

        # Subscribers
        self.create_subscription(Twist, '/cmd_vel', self._twist_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self._goal_cb, 10)
        self.create_subscription(String, '/turtlebot_ctrl', self._ctrl_cb, 10)

        # Twist state
        self._twist_v = 0.0
        self._twist_w = 0.0
        self._last_twist = None
        self._twist_timeout = float(twist_timeout)

    # ----- publishers -----
    def publish_path(self, waypoints_xy: List[Tuple[float,float]]):
        if not waypoints_xy:
            return
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
        t.child_frame_id  = self.base_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        half = 0.5 * float(yaw)
        t.transform.rotation = Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))
        self.tf_br.sendTransform(t)

    # ----- subscribers -----
    def _twist_cb(self, msg: Twist):
        self._twist_v = float(msg.linear.x)
        self._twist_w = float(msg.angular.z)
        self._last_twist = self.get_clock().now()

    def _goal_cb(self, msg: PoseStamped):
        if self.on_set_goal:
            self.on_set_goal(float(msg.pose.position.x), float(msg.pose.position.y))
        if self.auto_plan_on_goal and self.on_plan:
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
                self.on_set_goal(x, y)
            elif cmd == "speed" and len(tokens) >= 2 and self.on_speed:
                self.on_speed(max(0.0, float(tokens[1])), None)
            elif cmd == "angspeed" and len(tokens) >= 2 and self.on_speed:
                self.on_speed(None, max(0.0, float(tokens[1])))
        except Exception as e:
            self.get_logger().warn(f"/turtlebot_ctrl parse failed: {e}")

    # ----- API for main loop -----
    def get_active_twist(self) -> Tuple[float,float,bool]:
        """Return (v, w, active) where active=False if last cmd too old."""
        if self._last_twist is None:
            return 0.0, 0.0, False
        age = (self.get_clock().now() - self._last_twist).nanoseconds * 1e-9
        if age > self._twist_timeout:
            return 0.0, 0.0, False
        return self._twist_v, self._twist_w, True

# ===== Build stage visuals =====
world = World()
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # ensure meters

# Ground
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
goal_api = UsdGeom.XformCommonAPI(goal_prim.GetPrim())

def set_goal_xy(x: float, y: float):
    global GOAL_XY
    GOAL_XY = (float(x), float(y))
    goal_api.SetTranslate(Gf.Vec3f(GOAL_XY[0], GOAL_XY[1], 0.12))
    print(f"[goal] Set to ({GOAL_XY[0]:.2f}, {GOAL_XY[1]:.2f})")

# ---------- Spawn TurtleBot (wrapper Xform + explicit primPath + payload load) ----------
def _ensure_xform(path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Imageable(prim).MakeVisible()
    return prim

try:
    root_xf  = _ensure_xform(TURTLE_MOVE_PRIM)
    child_xf = _ensure_xform(TURTLE_CHILD_PRIM)

    # add reference to explicit prim
    refs = child_xf.GetReferences()
    refs.ClearReferences()
    refs.AddReference(Sdf.Reference(assetPath=LOCAL_TURTLEBOT_USD, primPath=TARGET_PRIM))
    child_xf.Load(Usd.LoadWithDescendants)

    # place & scale wrapper
    root_api = UsdGeom.XformCommonAPI(root_xf)
    root_api.SetTranslate(Gf.Vec3d(float(START_XY[0]), float(START_XY[1]), z_height))
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
    UsdGeom.XformCommonAPI(root_xf).SetTranslate(Gf.Vec3d(float(START_XY[0]), float(START_XY[1]), z_height))

world.reset()
world.play()

# ===== Xform utilities (version-proof) =====
def _get_local_mat(prim) -> Gf.Matrix4d:
    xf = UsdGeom.Xformable(prim)
    res = xf.GetLocalTransformation()
    return res[0] if isinstance(res, tuple) else res

# ===== Pose helpers (x, y, yaw) on the wrapper prim =====
def get_agent_pose() -> Tuple[float, float, float]:
    prim = stage.GetPrimAtPath(TURTLE_MOVE_PRIM)
    mat = _get_local_mat(prim)
    x = float(mat[3][0]); y = float(mat[3][1])
    # yaw from rotation matrix (Z-up). Note: matrix layout is row-major in Gf.Matrix4d
    yaw = math.atan2(float(mat[1][0]), float(mat[0][0]))
    return x, y, yaw

def set_agent_pose(x: float, y: float, yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    m = Gf.Matrix4d(
        c, -s, 0.0, 0.0,
        s,  c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        float(x), float(y), float(z_height), 1.0
    )
    prim = stage.GetPrimAtPath(TURTLE_MOVE_PRIM)
    UsdGeom.Xformable(prim).SetXformMatrix(m)

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

# ===== Build planning grid =====
grid = AStarGrid(MAP_W, MAP_H, RES, GRID_ORIGIN)
for (xmin, ymin, xmax, ymax) in OBST_AABBS:
    grid.mark_box_obstacle(xmin, ymin, xmax, ymax, val=100)

# ===== Planner/driver state =====
path_xy: List[Tuple[float,float]] = []
path_i = 0
following = False

def cancel_follow():
    global following, path_i
    following = False; path_i = 0
    print("[keys] Follow cancelled. Waiting for a new goal to replan.")

def trigger_plan_and_follow():
    global path_xy, path_i, following
    start_xy = get_agent_pose()[:2]
    path_xy = grid.plan(start_xy, GOAL_XY)
    ros.publish_path(path_xy)
    path_i = 0; following = bool(path_xy)
    print(f"[A*] Planned {len(path_xy)} waypoints." if path_xy else "[A*] No path.")

# ===== ROS init =====
if not rclpy.ok():
    rclpy.init(args=None)

def _on_set_goal(x, y):
    set_goal_xy(x, y)

def _on_plan():
    trigger_plan_and_follow()

def _on_cancel():
    cancel_follow()

def _on_speed(v_linear: Optional[float], w_angular: Optional[float]):
    global AGENT_SPEED, MAX_W_AUTO
    if v_linear is not None:
        AGENT_SPEED = max(0.0, float(v_linear))
        print(f"[speed] AGENT_SPEED set to {AGENT_SPEED:.2f} m/s")
    if w_angular is not None:
        MAX_W_AUTO = max(0.0, float(w_angular))
        print(f"[speed] MAX_W_AUTO set to {MAX_W_AUTO:.2f} rad/s")

ros = ROSBridge(
    global_frame="map",
    base_frame="base_link",
    plan_topic="/plan",
    on_set_goal=_on_set_goal,
    on_plan=_on_plan,
    on_cancel=_on_cancel,
    on_speed=_on_speed,
    auto_plan_on_goal=True,
    twist_timeout=TWIST_TIMEOUT
)

# ===== Keyboard handling =====
key_state = { "up": False, "down": False, "left": False, "right": False, "enter": False, "space": False }
app_window = omni.appwindow.get_default_app_window()
keyboard   = app_window.get_keyboard()
input_iface = carb_input.acquire_input_interface()

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

# Initial path (optional): publish empty plan
ros.publish_path(path_xy)
print("[A*] Ready. Keys: ↑/↓ linear, ←/→ angular. Enter = plan+follow; Space = stop.")
print("[ROS] Send /cmd_vel, /move_base_simple/goal, or /turtlebot_ctrl commands as needed.")

# ===== Main loop =====
try:
    # robot state
    x, y, yaw = get_agent_pose()

    while simulation_app.is_running():
        dt = world.get_physics_dt()
        rclpy.spin_once(ros, timeout_sec=0.0)

        # Handle one-shot keys
        if key_state["enter"]:
            trigger_plan_and_follow(); key_state["enter"] = False
        if key_state["space"]:
            cancel_follow(); key_state["space"] = False

        # Priority: Keyboard > /cmd_vel > A* follow
        keys_active = key_state["up"] or key_state["down"] or key_state["left"] or key_state["right"]
        v_tw, w_tw, tw_active = ros.get_active_twist()

        if keys_active:
            v_cmd = MAX_V_MANUAL * (float(key_state["up"]) - float(key_state["down"]))
            w_cmd = MAX_W_MANUAL * (float(key_state["left"]) - float(key_state["right"]))
            x  += v_cmd * math.cos(yaw) * dt
            y  += v_cmd * math.sin(yaw) * dt
            yaw = _wrap_pi(yaw + w_cmd * dt)
            set_agent_pose(x, y, yaw)

        elif tw_active:
            # ROS /cmd_vel drives when no keys pressed
            x  += v_tw * math.cos(yaw) * dt
            y  += v_tw * math.sin(yaw) * dt
            yaw = _wrap_pi(yaw + w_tw * dt)
            set_agent_pose(x, y, yaw)

        elif following and path_xy and path_i < len(path_xy):
            # A* follow with heading control
            tx, ty = path_xy[path_i]
            dx, dy = tx - x, ty - y
            dist   = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            err_yaw    = _wrap_pi(target_yaw - yaw)
            w = max(-MAX_W_AUTO, min(MAX_W_AUTO, ANG_KP * err_yaw))
            v = AGENT_SPEED * max(0.2, 1.0 - abs(err_yaw)/1.5)

            if dist < max(ARRIVE_DIST, AGENT_SPEED * dt):
                path_i += 1
                if path_i >= len(path_xy):
                    following = False
            else:
                x  += v * math.cos(yaw) * dt
                y  += v * math.sin(yaw) * dt
                yaw = _wrap_pi(yaw + w * dt)
                set_agent_pose(x, y, yaw)

        # Publish TF for current pose
        ros.publish_tf(x, y, yaw)

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
