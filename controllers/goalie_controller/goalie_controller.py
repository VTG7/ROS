from controller import Robot, Supervisor
import math

# ─── Field & Robot Geometry ──────────────────────────────────────────────────
# Single source of truth for everything that depends on the world setup.
# Values below match Webots' "adult" RobocupSoccerField + RobocupGoal protos:
#   - 14 m x 9 m playing area, goals at x = ±7
#   - goal width 2.7 m (post centres at |y| = 1.35), height 1.8 m
#
# To run on the "kid"-size proto instead, change GOAL_X to 4.5 and CHARGE_X
# accordingly, and move the goalie's `translation` in the world file.
# POST_Y is the same in both sizes (only field length and goal height change).
# The goalie's home x-position is read from the world at runtime (not here).
FIELD = {
    'GOAL_X':   7.0,   # x of the defended goal line
    'POST_Y':   1.35,  # |y| of the goal-post centres
    'CHARGE_X': 4.0,   # x the goalie charges to when cutting the angle
}

# Set to True to dump one line per frame to stdout describing the
# strategist's decision and the commander's command. Useful for diagnosing
# "why did the goalie do X on scenario Y?" without rerunning anything —
# fire the scenario, copy the [F####] lines, read them. Off by default;
# leave False unless you're actively debugging.
DEBUG_TRACE = True

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
        # Effective ball deceleration along its path. Calibrated against
        # observed Webots behaviour at this scale: 5 m/s straight shots
        # reach the goal (so a < ~1.8), and 4 m/s diagonal shots stop well
        # short (so a > ~1.1). 1.5 sits between the two, which lets the
        # energy check below distinguish "will reach" from "will stop short"
        # without rejecting normal save-worthy shots. Also feeds the
        # quadratic TTI estimate in _get_intercept_data; for shots that
        # can't physically reach the line the discriminant goes negative
        # and we fall back to linear TTI, which is harmless because the
        # energy check has already dropped the threat by then.
        self.deceleration = 1.5
        self.is_charging = False

        # Realistic robot lateral speed. Wheel cap is 12 rad/s and v0 = vy * 10,
        # so the actual achievable lateral speed is ~1.2 m/s. We use a slightly
        # conservative number so the "cut the angle" check tends to charge
        # rather than under-commit.
        self.v_y_max = 1.1

        # Hysteresis on charge release. After the strategist decides the ball
        # is no longer a threat we don't immediately drop charge / send the
        # goalie home; we hold the last commanded target for this many frames.
        # This prevents bailing out on transient blips (e.g. right after the
        # goalie deflects the ball and vx briefly goes near-zero or negative).
        self.release_threshold = 5
        self.release_counter = 0
        self.last_target_x = intercept_x
        self.last_target_y = 0.0

        # How far past the goalie's current x the ball is still considered
        # within last-ditch reach. Roughly the goalie's body radius. Past this
        # the goalie cannot physically intercept anymore (ball is ~10 m/s, goalie
        # ~0.6 m/s in x), so we stop tracking and go home.
        self.last_ditch_reach = 0.3

        # The strategist treats anything inside the post centres as a threat.
        # The actual post inner edge is at |y| ≈ POST_Y - post_radius (≈ 1.30);
        # using POST_Y itself gives a small grace margin for shots that would
        # graze the post rather than miss outright.
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
            
        # 1. BULLETPROOF SPATIAL GEOMETRY
        # We ALWAYS calculate where it will cross, regardless of friction stopping it early.
        cross_y = ball['y'] + (ball['vy'] / ball['vx']) * dx
        
        # 2. RESILIENT TEMPORAL KINEMATICS
        # Default to a simple linear time estimate if the friction math fails
        tti = dx / ball['vx'] 
        
        ax = -self.deceleration * (ball['vx'] / v_mag)
        a = 0.5 * ax
        b = ball['vx']
        c = -dx
        
        discriminant = b**2 - (4 * a * c)
        
        # Only use the complex quadratic time if the discriminant is valid!
        if discriminant > 0 and a != 0: 
            t1 = (-b + math.sqrt(discriminant)) / (2 * a)
            t2 = (-b - math.sqrt(discriminant)) / (2 * a)
            valid_times = [t for t in (t1, t2) if t > 0]
            if valid_times: 
                tti = min(valid_times)
                
        return cross_y, tti

    def _hold_or_release(self):
        """While charging, hold the last commanded target for a few frames
        before truly releasing. Returns a 'hold' threat dict if we should
        keep tracking, or None if we've fully released and should go home."""
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
        """Record the last commanded target so the hysteresis window can
        replay it if the threat momentarily disappears."""
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
            self.release_counter = 0
            self.last_branch = 'no-ball'
            return default_return

        # Goal-line projection — used by several blocks below. Falls back
        # to the last committed target when the geometry collapses (very
        # slow ball / nearly stopped), so downstream code always has a
        # sane y to work with.
        final_y, _ = self._get_intercept_data(ball, self.goal_net_x)
        if final_y is None:
            final_y = self.last_target_y

        # 1. PAST-GOALIE BLOCK. Run *before* the threat / energy gates.
        # Once the ball is at or past our body, geometric questions about
        # the goal line — "will it stay between the posts?", "does it
        # have the energy to reach?" — are irrelevant; the only thing
        # left to do is body contact at our current x. Putting this
        # ahead of the energy check also rescues the tail end of a
        # successful charge: by the time the ball arrives at our line,
        # friction has dropped v² below 2·a·d, and an energy-check
        # release at that moment would yank the goalie home exactly when
        # the ball is finally getting to us.
        #
        # Last-ditch target is ball['y'] (current lateral position) — not
        # final_y. For a nearly-stopped ball just past us, vy/vx explodes
        # as vx → 0 and the goal-line projection becomes meaningless,
        # producing extreme y targets that fling the goalie sideways for
        # nothing. ball['y'] is by definition where the body needs to be
        # to make contact, whether the ball is whizzing through or
        # rolling to a halt.
        if ball['x'] > current_x:
            if ball['x'] - current_x < self.last_ditch_reach:
                self.last_branch = 'last-ditch'
                return self._commit(target_x=current_x, target_y=ball['y'])
            self.is_charging = False
            self.release_counter = 0
            self.last_branch = 'past-goalie/release'
            return default_return

        # 2. THREAT CHECK at the goal line. Only releases when NOT mid-
        # charge: once committed, transient flicker in projected final_y
        # (Webots friction is high enough that vy wobbles noticeably
        # during flight) shouldn't cause us to reverse course.
        off_target = abs(final_y) > self.threat_half_width
        if off_target and not self.is_charging:
            held = self._hold_or_release()
            self.last_branch = 'off-target/hold' if held else 'off-target'
            return held if held is not None else default_return

        # 2b. ENERGY CHECK: v² ≥ 2·a·d along the ball's path to the goal
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

        # 3. CHARGE / INTERCEPT DECISION.
        # active_x — the line we commit to defend on this frame:
        #   * mid-charge AND ball still upstream of charge_x → charge_x
        #   * mid-charge AND ball has crossed charge_x but is still
        #     upstream of us → current_x. The charge "missed" the
        #     forward line geometrically (we couldn't slide there in
        #     time), but the ball is still in front of us, so the
        #     natural continuation is to defend at our own line with the
        #     cross_y at that x. Snapping back to intercept_x instead
        #     would force the goalie to reverse the -x momentum just
        #     built up during the charge — that's the second visible
        #     "abort" on hard corner shots, right at the charge_x
        #     boundary. With active_x = current_x, error_x ≈ 0, the x
        #     command is just braking momentum, and the y commitment
        #     stays intact all the way until past-goalie engages.
        #   * not charging → intercept_x, with possible promotion to
        #     charge_x if we can't make the lateral move in time and
        #     the ball is still upstream of charge_x.
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
    """Translates target coordinates into 2D omni-wheel commands (X and Y)."""
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
                
        # Position-feedback gains.
        self.Kp_y = 8.0
        self.Kp_x = 4.0
        # Velocity-feedback (damping) gain. With higher robot speeds the
        # goalie carries real momentum, and the previous "error - prev_error"
        # term effectively damped almost nothing (it was ~dy per frame, on the
        # order of 0.02). Damping against true velocity prevents the
        # post-save coast that drives the goalie past target_y to the edge.
        self.Kd_v = 4.0

        # Cache the previous position so we can numerically estimate the
        # goalie's actual world velocity each frame.
        self.prev_y = None
        self.prev_x = None

        # Last commanded values for the per-frame trace (DEBUG_TRACE).
        # None means move_to_target hasn't run yet this frame.
        self.last_trace = None

    def move_to_target(self, current_x, current_y, target_y, target_x):
        # Estimate the goalie's actual velocity from frame-to-frame position
        # change. This is what we damp against — far more effective than
        # damping against error-rate alone.
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
        
        # Wheel angular-velocity cap. With the kinematics above (v0 = vy*10,
        # v1 = vx*8.66 - vy*5) this directly governs the robot's top speed:
        #   max lateral ≈ wheel_cap / 10  m/s
        #   max forward ≈ wheel_cap / 8.66  m/s
        # 12 rad/s ⇒ ~1.2 m/s lateral, ~1.4 m/s forward. Keep the strategist's
        # v_y_max in sync if you change this.
        wheel_cap = 12.0
        v0_c = max(min(v0, wheel_cap), -wheel_cap)
        v1_c = max(min(v1, wheel_cap), -wheel_cap)
        v2_c = max(min(v2, wheel_cap), -wheel_cap)
        if self.m0: self.m0.setVelocity(v0_c)
        if self.m1: self.m1.setVelocity(v1_c)
        if self.m2: self.m2.setVelocity(v2_c)

        self.last_trace = {
            'tx': target_x, 'ty': target_y,
            'ex': error_x, 'ey': error_y,
            'avx': actual_vx, 'avy': actual_vy,
            'vx': vx, 'vy': vy,
            'w0': v0_c, 'w1': v1_c, 'w2': v2_c,
        }

class AutoShooter:
    """Plays a curated sequence of test scenarios to exercise the goalie.

    Each scenario is a list of one or more shots; each shot has a spawn
    position, an aim point on the goal line, a launch speed, and an absolute
    frame offset 't' inside the scenario (so chained / delayed sequences
    like rebound and volley work cleanly).

    Controls:
      N           next scenario (fires it)
      P           previous scenario (fires it)
      R / SPACE   repeat / fire the current scenario
      1-9, 0      jump directly to scenario 1-9 / 10 (fires it)
      C           clear the ball off the field
    """

    # Aim coordinates reference FIELD so scenarios automatically adapt if the
    # goal line moves (e.g. swapping to the kid-size field). Spawn coordinates
    # are scenario-specific design choices — calibrated for the adult field
    # (14 x 9 m). If switching to kid (9 x 6 m), rescale spawns in x.
    # Speeds are in m/s, 't' is frame offset within the scenario.
    #
    # _G  = goal line x. _P = |y| of post centre (used relative to the post
    # to express "just inside" / "well outside" without magic numbers).
    SCENARIOS = [
        {
            'name': 'Straight Center, Slow',
            'shots': [{'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'],  0.0),                          'speed':  5.0, 't': 0}],
        },
        {
            'name': 'Straight Center, Fast',
            'shots': [{'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'],  0.0),                          'speed': 10.0, 't': 0}],
        },
        {
            'name': 'Mid-Range Center (close, fast)',
            'shots': [{'spawn': (3.0,  0.0), 'aim': (FIELD['GOAL_X'],  0.0),                          'speed':  8.0, 't': 0}],
        },
        {
            'name': 'Hard Left-Corner Cut-Angle',
            'shots': [{'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'], -FIELD['POST_Y'] + 0.05),       'speed': 10.0, 't': 0}],
        },
        {
            'name': 'Hard Right-Corner Cut-Angle',
            'shots': [{'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'],  FIELD['POST_Y'] - 0.05),       'speed': 10.0, 't': 0}],
        },
        {
            'name': 'Sharp Angle from Right Wing',
            'shots': [{'spawn': (2.0,  1.6), 'aim': (FIELD['GOAL_X'], -FIELD['POST_Y'] + 0.05),       'speed':  9.0, 't': 0}],
        },
        {
            'name': 'Sharp Angle from Left Wing',
            'shots': [{'spawn': (2.0, -1.6), 'aim': (FIELD['GOAL_X'],  FIELD['POST_Y'] - 0.05),       'speed':  9.0, 't': 0}],
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
        {
            'name': 'Rebound Sequence (two shots)',
            'shots': [
                {'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'], -FIELD['POST_Y'] + 0.15),             'speed': 10.0, 't': 0},
                {'spawn': (3.0,  1.5), 'aim': (FIELD['GOAL_X'], -0.5),                                'speed':  9.0, 't': 80},
            ],
        },
        {
            'name': 'Pressure Volley (three quick shots)',
            'shots': [
                {'spawn': (0.0,  0.0), 'aim': (FIELD['GOAL_X'], -FIELD['POST_Y'] + 0.15),             'speed': 10.0, 't': 0},
                {'spawn': (1.0,  1.5), 'aim': (FIELD['GOAL_X'],  1.0),                                'speed':  9.0, 't': 70},
                {'spawn': (2.0, -1.5), 'aim': (FIELD['GOAL_X'], -0.8),                                'speed':  9.0, 't': 140},
            ],
        },
    ]

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
        # Scheduled events for the current scenario. Each event is a dict with
        # 'frames' (countdown) and 'kind' ('teleport' or 'velocity') plus its
        # own payload.
        self.events = []
        # Last raw key value seen, for press-edge debouncing (only fire on
        # the transition from "not pressed" / "different key" to "pressed").
        self.last_key = -1

        self._print_help()
        self._announce()

    def _print_help(self):
        print("=" * 60, flush=True)
        print("AutoShooter scenario player", flush=True)
        print("  N         next scenario (fires it)", flush=True)
        print("  P         previous scenario (fires it)", flush=True)
        print("  R / SPACE repeat / fire current scenario", flush=True)
        print("  1-9, 0    jump to scenario 1-9 / 10", flush=True)
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
        print("  ball cleared", flush=True)

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
        self.events = still_pending

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

        current_pos = robot_node.getPosition()
        current_x = current_pos[0]
        current_y = current_pos[1]

        # Pass current_x and current_y so the Strategist can both judge if we
        # are too slow laterally AND lock our x in last-ditch defense.
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