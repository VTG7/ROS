from controller import Robot, Supervisor
import math
import random


FIELD = {
    'GOAL_X':   7.0,   # x of the defended goal line
    'POST_Y':   1.35,  # |y| of the goal-post centres
    'CHARGE_X': 4.0,   # x the goalie charges to when cutting the angle
}


DEBUG_TRACE = False

class Observer:
    """Uses Supervisor God-Mode to track the ball perfectly without noise."""
    def __init__(self, supervisor):
        self.supervisor = supervisor
        self.ball_node = self.supervisor.getFromDef("BALL")
        if self.ball_node is None:
            print("ERROR: Could not find BALL.", flush=True)
            
    def get_ball_data(self):
        if self.ball_node is None: return None
        
        # Get perfect ground-truth simulation data
        pos = self.ball_node.getPosition()
        vel = self.ball_node.getVelocity()
        
        return {'x': pos[0], 'y': pos[1], 'vx': vel[0], 'vy': vel[1]}

class Strategist:
    """Calculates interception using bulletproof Spatial Geometry and resilient Kinematics."""
    def __init__(self, intercept_x):
        self.intercept_x = intercept_x
        self.goal_net_x = FIELD['GOAL_X']
        self.charge_x = FIELD['CHARGE_X']

        self.deceleration = 1.5
        self.is_charging = False


        self.v_y_max = 1.1


        self.release_threshold = 5
        self.release_counter = 0
        self.last_target_x = intercept_x
        self.last_target_y = 0.0

        # How far past the goalie's current x the ball is still considered
        # within last-ditch reach. Roughly the goalie's body radius. Past this
        # the goalie cannot physically intercept anymore (ball is ~10 m/s, goalie
        # ~0.6 m/s in x), so we stop tracking and go home.
        self.last_ditch_reach = 0.3


        self.past_goalie_done = False


        self.threat_half_width = FIELD['POST_Y']

        # Populated by calculate_interception so the main-loop trace can
        # report which branch produced the current target (see DEBUG_TRACE).
        self.last_branch = '?'

    def _get_intercept_data(self, ball, target_x):
        """Returns (Crossing_Y, Time_To_Intercept). Never aborts due to friction."""
        dx = target_x - ball['x']
        
        # Boomerang check (moving wrong way)
        if (dx > 0 and ball['vx'] <= 0) or (dx < 0 and ball['vx'] >= 0):
            return None, None
            
        v_mag = math.sqrt(ball['vx']**2 + ball['vy']**2)
        if v_mag < 0.05: 
            return None, None

        cross_y = ball['y'] + (ball['vy'] / ball['vx']) * dx
        

        tti = dx / ball['vx'] 
        
        ax = -self.deceleration * (ball['vx'] / v_mag)
        a = 0.5 * ax
        b = ball['vx']
        c = -dx
        
        discriminant = b**2 - (4 * a * c)
        
        # Only use the complex quadratic time if the discriminant is valid..
        if discriminant > 0 and a != 0: 
            t1 = (-b + math.sqrt(discriminant)) / (2 * a)
            t2 = (-b - math.sqrt(discriminant)) / (2 * a)
            valid_times = [t for t in (t1, t2) if t > 0]
            if valid_times: 
                tti = min(valid_times)
                
        return cross_y, tti

    def _hold_or_release(self):

        if not self.is_charging:
            return None
        self.release_counter += 1
        if self.release_counter >= self.release_threshold:
            self.is_charging = False
            self.release_counter = 0
            return None
        return {
            'is_threat': True,
            'target_x': self.last_target_x,
            'target_y': self.last_target_y,
        }

    def _commit(self, target_x, target_y):

        self.last_target_x = target_x
        self.last_target_y = target_y
        self.release_counter = 0
        return {'is_threat': True, 'target_x': target_x, 'target_y': target_y}

    def calculate_interception(self, ball, current_x, current_y):
        default_return = {'is_threat': False, 'target_x': self.intercept_x, 'target_y': 0.0}
        self.last_branch = '?'

        # Hard-stop conditions: ball gone / past goal line / not moving forward.
        # Force a full release here regardless of hysteresis — there is nothing
        # left to defend.
        if ball is None or ball['vx'] <= 0.01 or ball['x'] >= self.goal_net_x:
            self.is_charging = False
            self.past_goalie_done = False
            self.release_counter = 0
            self.last_branch = 'no-ball'
            return default_return


        final_y, _ = self._get_intercept_data(ball, self.goal_net_x)
        if final_y is None:
            final_y = self.last_target_y


        if ball['x'] > current_x:
            within_reach = ball['x'] - current_x < self.last_ditch_reach
            if within_reach and not self.past_goalie_done:
                self.last_branch = 'last-ditch'
                return self._commit(target_x=current_x, target_y=ball['y'])
            self.past_goalie_done = True
            self.is_charging = False
            self.release_counter = 0
            self.last_branch = 'past-goalie/release'
            return default_return

        # Ball is back upstream of us (rare: weird rebound that comes
        # back toward us). Reset the one-shot flag so legitimate fresh
        # threats can engage last-ditch again if needed.
        self.past_goalie_done = False

        off_target = abs(final_y) > self.threat_half_width
        if off_target and not self.is_charging:
            held = self._hold_or_release()
            self.last_branch = 'off-target/hold' if held else 'off-target'
            return held if held is not None else default_return

        # ENERGY CHECK: v² ≥ 2·a·d along the ball's path to the goal
        # line. Filters slow shots (e.g. 4 m/s diagonal) that project on-
        # target geometrically but actually stop short of us due to
        # friction. Same charging guard as the threat check — bailing
        # mid-charge produced the visible "moves and aborts" reversal on
        # Hard Left/Right, since Webots' real friction is significantly
        # higher than our model and v² drops below 2·a·d well before the
        # ball actually arrives.
        v_mag_sq = ball['vx']**2 + ball['vy']**2
        v_mag = math.sqrt(v_mag_sq)
        path_length = (self.goal_net_x - ball['x']) * v_mag / max(ball['vx'], 1e-3)
        if v_mag_sq < 2 * self.deceleration * path_length and not self.is_charging:
            held = self._hold_or_release()
            self.last_branch = 'low-energy/hold' if held else 'low-energy'
            return held if held is not None else default_return


        if self.is_charging:
            active_x = self.charge_x if ball['x'] < self.charge_x else current_x
        else:
            active_x = self.intercept_x
            intercept_y, tti = self._get_intercept_data(ball, self.intercept_x)
            if intercept_y is None:
                held = self._hold_or_release()
                self.last_branch = 'no-intercept/hold' if held else 'no-intercept'
                return held if held is not None else default_return
            time_to_reach = abs(intercept_y - current_y) / self.v_y_max
            if time_to_reach > tti and ball['x'] < self.charge_x:
                self.is_charging = True
                active_x = self.charge_x

        target_y, _ = self._get_intercept_data(ball, active_x)
        if target_y is None:
            # Geometry failed for the active line; fall back to the goal
            # line projection so we still defend something sensible.
            target_y = final_y

        self.last_branch = 'charge' if self.is_charging else 'lateral'
        return self._commit(target_x=active_x, target_y=target_y)

class Commander:
    def __init__(self, robot, timestep_ms):
        self.robot = robot
        self.dt = timestep_ms / 1000.0
        self.m0 = robot.getDevice("wheel0_joint")
        self.m1 = robot.getDevice("wheel1_joint")
        self.m2 = robot.getDevice("wheel2_joint")
        
        for m in [self.m0, self.m1, self.m2]:
            if m is not None:
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                
    
        self.Kp_y = 8.0
        self.Kp_x = 4.0

        self.Kd_v = 4.0

        self.prev_y = None
        self.prev_x = None

        self.last_trace = None

    def move_to_target(self, current_x, current_y, target_y, target_x):

        if self.prev_y is None:
            actual_vy = 0.0
            actual_vx = 0.0
        else:
            actual_vy = (current_y - self.prev_y) / self.dt
            actual_vx = (current_x - self.prev_x) / self.dt
        self.prev_y = current_y
        self.prev_x = current_x

        # 1. Y-axis PD with velocity feedback. Settle into the deadband only
        # when both the position error AND the actual speed are small;
        # otherwise we'd "settle" while still coasting through the target.
        error_y = target_y - current_y
        if abs(error_y) < 0.02 and abs(actual_vy) < 0.05:
            vy = 0.0
        else:
            vy = (self.Kp_y * error_y) - (self.Kd_v * actual_vy)
            vy = max(min(vy, 4.0), -4.0)

        # 2. X-axis PD with velocity feedback. Note: positive commanded vx
        # corresponds to -x world motion in this kinematic convention, hence
        # the outer negation.
        error_x = target_x - current_x
        if abs(error_x) < 0.02 and abs(actual_vx) < 0.05:
            vx = 0.0
        else:
            vx = -((self.Kp_x * error_x) - (self.Kd_v * actual_vx))
            vx = max(min(vx, 2.0), -2.0)

        # 3. Kinematics
        v0 = vy * 10.0
        v1 = (vx * 8.66) - (vy * 5.0)
        v2 = (-vx * 8.66) - (vy * 5.0)
        

        wheel_cap = 12.0

        max_wheel = max(abs(v0), abs(v1), abs(v2))
        if max_wheel > wheel_cap:
            scale = wheel_cap / max_wheel
            v0 *= scale
            v1 *= scale
            v2 *= scale

        if self.m0: self.m0.setVelocity(v0)
        if self.m1: self.m1.setVelocity(v1)
        if self.m2: self.m2.setVelocity(v2)

        self.last_trace = {
            'tx': target_x, 'ty': target_y,
            'ex': error_x, 'ey': error_y,
            'avx': actual_vx, 'avy': actual_vy,
            'vx': vx, 'vy': vy,
            'w0': v0, 'w1': v1, 'w2': v2,
        }

class AutoShooter:
    """
    Controls:
      N           next scenario (fires it)
      P           previous scenario (fires it)
      R / SPACE   repeat / fire the current scenario
      1-9, 0      jump directly to scenario 1-9 / 10 (fires it)
      C           clear the ball off the field
    """


    SCENARIOS = [
        {
            'name': 'Straight Center, Fast',
            'shots': [{'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'],  0.0),                          'speed': 10.0, 't': 0}],
        },
        {
            'name': 'Mid-Range Center (close, fast)',
            'shots': [{'spawn': (3.0,  0.0), 'aim': (FIELD['GOAL_X'],  0.0),                          'speed':  8.0, 't': 0}],
        },
        {
            'name': 'Hard Right-Corner Cut-Angle',
            'shots': [{'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'],  FIELD['POST_Y'] - 0.05),       'speed':  9.5, 't': 0}],
        },
        {
            'name': 'Cross From Right Wing (through centre)',
            'shots': [{'spawn': (2.0,  1.6), 'aim': (FIELD['GOAL_X'], -FIELD['POST_Y'] + 0.05),       'speed':  9.0, 't': 0}],
        },
        {
            'name': 'Slow Diagonal (friction prediction)',
            'shots': [{'spawn': (0.0, -1.5), 'aim': (FIELD['GOAL_X'],  1.0),                          'speed':  4.0, 't': 0}],
        },
        {
            'name': 'Off-Target Wide (should NOT save)',
            'shots': [{'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'],  FIELD['POST_Y'] + 0.45),       'speed':  8.0, 't': 0}],
        },
        {
            'name': 'Last-Ditch (close-range, sharp)',
            'shots': [{'spawn': (4.0,  1.0), 'aim': (FIELD['GOAL_X'], -0.6),                          'speed': 10.0, 't': 0}],
        },
    ]


    RANDOM_SPAWN = (0.0, 0.0)
    RANDOM_AIM_Y_RANGE = (-0.9, 0.9)
    RANDOM_SPEED_RANGE = (9.0, 12.0)

    # Frame gap between teleporting the ball and applying its velocity. Gives
    # the physics engine a moment to settle on the new spawn before launch.
    LAUNCH_DELAY_FRAMES = 3

    def __init__(self, supervisor, ball_node):
        self.supervisor = supervisor
        self.ball_node = ball_node
        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(int(self.supervisor.getBasicTimeStep()))

        self.trans_field = self.ball_node.getField("translation") if self.ball_node else None

        self.current_index = 0

        self.events = []
        self.last_key = -1
        self.tracking = False
        self.shot_on_target = False

        self._print_help()
        self._announce()

    def _print_help(self):
        print("=" * 60, flush=True)
        print("AutoShooter scenario player", flush=True)
        print("  N         next scenario (fires it)", flush=True)
        print("  P         previous scenario (fires it)", flush=True)
        print("  R / SPACE repeat / fire current scenario", flush=True)
        print("  1-9, 0    jump to scenario 1-9 / 10", flush=True)
        print("  X         random shot from centre (conservative angles)", flush=True)
        print("  C         clear the ball off the field", flush=True)
        print("=" * 60, flush=True)

    def _announce(self):
        s = self.SCENARIOS[self.current_index]
        n = len(self.SCENARIOS)
        print(f"\n[{self.current_index + 1}/{n}] {s['name']}", flush=True)

    def _schedule_scenario(self):
        """Build the event queue for the current scenario."""
        s = self.SCENARIOS[self.current_index]
        self.events = []
        for shot in s['shots']:
            t = shot.get('t', 0)
            spawn = shot['spawn']
            aim = shot['aim']
            speed = shot['speed']
            dx = aim[0] - spawn[0]
            dy = aim[1] - spawn[1]
            mag = math.sqrt(dx * dx + dy * dy) or 1.0
            vx = (dx / mag) * speed
            vy = (dy / mag) * speed
            self.events.append({
                'frames': t,
                'kind': 'teleport',
                'spawn': spawn,
            })
            self.events.append({
                'frames': t + self.LAUNCH_DELAY_FRAMES,
                'kind': 'velocity',
                'vx': vx,
                'vy': vy,
            })

    def _fire_current(self):
        self._schedule_scenario()
        print("  fire", flush=True)

    def _fire_random(self):
        spawn = self.RANDOM_SPAWN
        aim_y = random.uniform(*self.RANDOM_AIM_Y_RANGE)
        speed = random.uniform(*self.RANDOM_SPEED_RANGE)
        aim = (FIELD['GOAL_X'], aim_y)

        dx = aim[0] - spawn[0]
        dy = aim[1] - spawn[1]
        mag = math.sqrt(dx * dx + dy * dy) or 1.0
        vx = (dx / mag) * speed
        vy = (dy / mag) * speed

        self.events = [
            {'frames': 0, 'kind': 'teleport', 'spawn': spawn},
            {'frames': self.LAUNCH_DELAY_FRAMES, 'kind': 'velocity', 'vx': vx, 'vy': vy},
        ]
        n = len(self.SCENARIOS)
        print(
            f"\n[X/{n}] Random Shot  spawn=({spawn[0]:+.1f},{spawn[1]:+.1f}) "
            f"aim=({aim[0]:.1f},{aim_y:+.2f}) speed={speed:.2f}",
            flush=True,
        )
        print("  fire", flush=True)

    def _step(self, delta):
        self.current_index = (self.current_index + delta) % len(self.SCENARIOS)
        self._announce()
        self._fire_current()

    def _jump(self, idx):
        if 0 <= idx < len(self.SCENARIOS):
            self.current_index = idx
            self._announce()
            self._fire_current()

    def _clear_ball(self):
        if self.trans_field:
            self.trans_field.setSFVec3f([-5.0, 0.0, 0.1])
            self.ball_node.resetPhysics()
            self.ball_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.events = []
        self.tracking = False
        print("  ball cleared", flush=True)

    @staticmethod
    def _is_on_target(spawn, vx, vy):
        goal_x = FIELD['GOAL_X']
        post_y = FIELD['POST_Y']
        if vx <= 0.01:
            return False
        cross_y = spawn[1] + (vy / vx) * (goal_x - spawn[0])
        if abs(cross_y) > post_y:
            return False
        v_mag_sq = vx * vx + vy * vy
        v_mag = math.sqrt(v_mag_sq)
        path_length = (goal_x - spawn[0]) * v_mag / vx
        # Match Strategist.deceleration so the on-target judgement here
        # agrees with the strategist's own low-energy filter.
        deceleration = 1.5
        return v_mag_sq >= 2 * deceleration * path_length

    def _process_events(self):
        if not self.events:
            return
        still_pending = []
        for ev in self.events:
            ev['frames'] -= 1
            if ev['frames'] > 0:
                still_pending.append(ev)
                continue
            if ev['kind'] == 'teleport':
                spawn = ev['spawn']
                self.trans_field.setSFVec3f([spawn[0], spawn[1], 0.1])
                self.ball_node.resetPhysics()
            elif ev['kind'] == 'velocity':
                self.ball_node.setVelocity([ev['vx'], ev['vy'], 0.0, 0.0, 0.0, 0.0])
                # Arm outcome tracking from the launch frame. Use the
                # ball's spawn position (recorded by the prior teleport
                # event) as the geometric origin for the on-target check.
                spawn_pos = self.ball_node.getPosition()
                self.shot_on_target = self._is_on_target(
                    (spawn_pos[0], spawn_pos[1]), ev['vx'], ev['vy']
                )
                self.tracking = True
        self.events = still_pending

    def watch_outcome(self, ball):
        if not self.tracking or ball is None:
            return
        goal_x = FIELD['GOAL_X']
        post_y = FIELD['POST_Y']

        past_line = ball['x'] >= goal_x
        settled = abs(ball['vx']) < 0.05 and abs(ball['vy']) < 0.05

        if not (past_line or settled):
            return

        if past_line and abs(ball['y']) <= post_y:
            outcome = 'GOAL'
        elif self.shot_on_target:
            outcome = 'SAVE'
        else:
            outcome = 'LEAVE'

        print(f"  -> {outcome}", flush=True)
        self.tracking = False

    def check_and_shoot(self):
        if not self.ball_node:
            return

        self._process_events()

        # Press-edge debouncing: only act when the key value changes from
        # what it was last frame, so holding a key doesn't spam fire.
        key = self.keyboard.getKey()
        if key == self.last_key:
            return
        self.last_key = key
        if key == -1:
            return

        if key in (ord('N'), ord('n')):
            self._step(+1)
        elif key in (ord('P'), ord('p')):
            self._step(-1)
        elif key in (ord('R'), ord('r'), ord(' ')):
            self._announce()
            self._fire_current()
        elif key in (ord('C'), ord('c')):
            self._clear_ball()
        elif key in (ord('X'), ord('x')):
            self._fire_random()
        elif ord('1') <= key <= ord('9'):
            self._jump(key - ord('1'))
        elif key == ord('0'):
            self._jump(9)

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    observer = Observer(robot)
    
    robot_node = robot.getSelf()
    home_x = robot_node.getPosition()[0]
    
    strategist = Strategist(home_x)
    commander = Commander(robot, timestep)
    shooter = AutoShooter(robot, observer.ball_node)
    
    print("Goalie online. Use AutoShooter controls listed above to run scenarios.", flush=True)

    frame = 0
    while robot.step(timestep) != -1:
        shooter.check_and_shoot()

        ball_state = observer.get_ball_data()
        shooter.watch_outcome(ball_state)

        current_pos = robot_node.getPosition()
        current_x = current_pos[0]
        current_y = current_pos[1]

        target = strategist.calculate_interception(ball_state, current_x, current_y)

        if target['is_threat']:
            commander.move_to_target(current_x, current_y, target['target_y'], target['target_x'])
        else:
            commander.move_to_target(current_x, current_y, 0.0, target['target_x'])

        # Per-frame diagnostic trace. Only emits while a ball is in flight,
        # to keep the console quiet during idle waiting between scenarios.
        if DEBUG_TRACE and ball_state is not None and ball_state['vx'] > 0.05:
            t = commander.last_trace or {}
            print(
                f"[F{frame:04d}] "
                f"BALL=({ball_state['x']:+.2f},{ball_state['y']:+.2f}) "
                f"V=({ball_state['vx']:+.2f},{ball_state['vy']:+.2f}) "
                f"GK=({current_x:+.2f},{current_y:+.2f}) "
                f"BR={strategist.last_branch:<22s} "
                f"CHG={int(strategist.is_charging)} "
                f"TGT=({t.get('tx', 0):+.2f},{t.get('ty', 0):+.2f}) "
                f"ERR=({t.get('ex', 0):+.2f},{t.get('ey', 0):+.2f}) "
                f"GKv=({t.get('avx', 0):+.2f},{t.get('avy', 0):+.2f}) "
                f"CMD=({t.get('vx', 0):+.2f},{t.get('vy', 0):+.2f}) "
                f"W=({t.get('w0', 0):+5.1f},{t.get('w1', 0):+5.1f},{t.get('w2', 0):+5.1f})",
                flush=True,
            )
        frame += 1

if __name__ == "__main__":
    main()