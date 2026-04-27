from controller import Robot, Supervisor
import random
import math

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
    """Calculates interception using Hybrid Kinematics (Spatial Geometry + Temporal Friction)."""
    def __init__(self, intercept_x):
        self.intercept_x = intercept_x 
        self.goal_net_x = 7.0          
        self.deceleration = 0.25  # The friction kinematic is back!
        self.is_charging = False  
        
    def _get_intercept_data(self, ball, target_x):
        """Returns (Crossing_Y, Time_To_Intercept) using a hybrid model."""
        dx = target_x - ball['x']
        
        # Boomerang check (moving wrong way)
        if (dx > 0 and ball['vx'] <= 0) or (dx < 0 and ball['vx'] >= 0):
            return None, None
            
        v_mag = math.sqrt(ball['vx']**2 + ball['vy']**2)
        if v_mag < 0.05: 
            return None, None
            
        # 1. SPATIAL GEOMETRY: Friction doesn't bend the path, it stays a straight line
        cross_y = ball['y'] + (ball['vy'] / ball['vx']) * dx
        
        # 2. TEMPORAL KINEMATICS: Friction heavily delays the arrival time! 
        ax = -self.deceleration * (ball['vx'] / v_mag)
        
        a = 0.5 * ax
        b = ball['vx']
        c = -dx
        
        discriminant = b**2 - (4 * a * c)
        
        # If discriminant < 0, the friction stops the ball before it reaches the line
        if discriminant < 0 or a == 0: 
            return None, None
            
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)
        
        valid_times = [t for t in (t1, t2) if t > 0]
        if not valid_times: 
            return None, None
            
        tti = min(valid_times)
        return cross_y, tti

    def calculate_interception(self, ball, current_y):
        default_return = {'is_threat': False, 'target_x': self.intercept_x, 'target_y': 0.0}
        
        # Safety checks
        if ball is None or ball['vx'] <= 0.01 or ball['x'] >= self.goal_net_x:
            self.is_charging = False
            return default_return
            
        # 1. THREAT CHECK: Does the kinematics prove it will reach the net at X=7.0?
        final_y, _ = self._get_intercept_data(ball, self.goal_net_x)
        if final_y is None or abs(final_y) > 1.4:
            self.is_charging = False 
            return default_return
            
        # 2. THE PASS-BY FIX: Are we defending the 5.0 line or the 4.0 line?
        active_x = 4.0 if self.is_charging else self.intercept_x
        
        if ball['x'] > active_x:
            # The ball is behind the robot! The play is dead. Return to center.
            self.is_charging = False
            return default_return
            
        # 3. KINEMATIC TIMING CHECK
        intercept_y, tti = self._get_intercept_data(ball, self.intercept_x)
        if intercept_y is None:
            self.is_charging = False
            return default_return
            
        # 4. CUT THE ANGLE LOGIC (Only calculate if we aren't already charging)
        if not self.is_charging:
            time_to_reach = abs(intercept_y - current_y) / 1.2
            
            # If kinematic time says we are too slow, commit to the charge!
            if time_to_reach > tti:
                self.is_charging = True
                active_x = 4.0
        
        # Get the spatial target for our active defense line
        target_y, _ = self._get_intercept_data(ball, active_x)
        
        if target_y is not None:
            return {'is_threat': True, 'target_x': active_x, 'target_y': target_y}
            
        self.is_charging = False
        return default_return
        
class Commander:
    """Translates target coordinates into 2D omni-wheel commands (X and Y)."""
    def __init__(self, robot):
        self.robot = robot
        self.m0 = robot.getDevice("wheel0_joint")
        self.m1 = robot.getDevice("wheel1_joint")
        self.m2 = robot.getDevice("wheel2_joint")
        
        for m in [self.m0, self.m1, self.m2]:
            if m is not None:
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                
        self.prev_error_y = 0.0
        self.Kp_y = 8.0  
        self.Kd_y = 2.0  
            
    def move_to_target(self, current_x, current_y, target_y, target_x):
        # 1. Y-axis PD Control
        error_y = target_y - current_y
        if abs(error_y) < 0.03:
            vy = 0.0
            self.prev_error_y = 0.0 
        else:
            derivative = error_y - self.prev_error_y
            vy = (self.Kp_y * error_y) + (self.Kd_y * derivative)
            self.prev_error_y = error_y
            vy = max(min(vy, 4.0), -4.0) 
            
        # 2. X-axis P Control (Dynamically holds home line OR charges forward)
        error_x = target_x - current_x
        if abs(error_x) < 0.03:
            vx = 0.0
        else:
            vx = -error_x * 4.0 
            vx = max(min(vx, 2.0), -2.0)

        # 3. Kinematics
        v0 = vy * 10.0
        v1 = (vx * 8.66) - (vy * 5.0)
        v2 = (-vx * 8.66) - (vy * 5.0)
        
        if self.m0: self.m0.setVelocity(max(min(v0, 6.0), -6.0))
        if self.m1: self.m1.setVelocity(max(min(v1, 6.0), -6.0))
        if self.m2: self.m2.setVelocity(max(min(v2, 6.0), -6.0))

class AutoShooter:
    """Spawns the ball at different locations based on key presses."""
    def __init__(self, supervisor, ball_node):
        self.supervisor = supervisor
        self.ball_node = ball_node
        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(int(self.supervisor.getBasicTimeStep()))
        
        if self.ball_node:
            self.trans_field = self.ball_node.getField("translation")
            
        self.frames_until_shoot = -1
        self.pending_vx = 0.0
        self.pending_vy = 0.0
        self.shot_type = ""
        
    def check_and_shoot(self):
        if not self.ball_node: return
        
        if self.frames_until_shoot > 0:
            self.frames_until_shoot -= 1
            return
        elif self.frames_until_shoot == 0:
            self.ball_node.setVelocity([self.pending_vx, self.pending_vy, 0.0, 0.0, 0.0, 0.0])
            print(f"💥 {self.shot_type} FIRED!", flush=True)
            self.frames_until_shoot = -1 
            return

        key = self.keyboard.getKey()
        
        # 1. SPACEBAR: Hard corner shot from the center to test "Cutting the Angle"
        if key == ord(' '):
            self.shot_type = "CORNER CUT-ANGLE SHOT"
            spawn_x = 0.0
            spawn_y = 0.0
            self.trans_field.setSFVec3f([spawn_x, spawn_y, 0.1])
            self.ball_node.resetPhysics()
            
            aim_x = 7.0
            # Force the ball to aim strictly at the extreme left or right edge of the net
            aim_y = random.choice([1.2, -1.2]) 
            
            dx = aim_x - spawn_x
            dy = aim_y - spawn_y
            magnitude = math.sqrt(dx**2 + dy**2)
            
            self.pending_vx = (dx / magnitude) * 10.0
            self.pending_vy = (dy / magnitude) * 10.0
                
            self.frames_until_shoot = 2
            
        # 2. 'R' KEY: Random dynamic shot from anywhere
        elif key == ord('R') or key == ord('r'):
            self.shot_type = "RANDOM DYNAMIC SHOT"
            spawn_x = random.uniform(0.0, 4.0)
            spawn_y = random.uniform(-2.0, 2.0)
            self.trans_field.setSFVec3f([spawn_x, spawn_y, 0.1])
            self.ball_node.resetPhysics()
            
            aim_x = 7.0
            aim_y = random.uniform(-1.2, 1.2)
            
            dx = aim_x - spawn_x
            dy = aim_y - spawn_y
            magnitude = math.sqrt(dx**2 + dy**2)
            
            self.pending_vx = (dx / magnitude) * 10.0
            self.pending_vy = (dy / magnitude) * 10.0
                
            self.frames_until_shoot = 2

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    observer = Observer(robot)
    
    robot_node = robot.getSelf()
    home_x = robot_node.getPosition()[0]
    
    strategist = Strategist(home_x)
    commander = Commander(robot)
    shooter = AutoShooter(robot, observer.ball_node)
    
    print("Omniscient Goalie Final Version Online. Press SPACEBAR for Dynamic Shots!", flush=True)
    
    while robot.step(timestep) != -1:
        shooter.check_and_shoot()
        
        ball_state = observer.get_ball_data()
        
        current_pos = robot_node.getPosition()
        current_x = current_pos[0]
        current_y = current_pos[1]
        
        # Pass current_y so the Strategist knows if we are too slow!
        target = strategist.calculate_interception(ball_state, current_y)
        
        if target['is_threat']:
            # Dynamically follows either home_x (5.0) or charge_x (4.0)
            commander.move_to_target(current_x, current_y, target['target_y'], target['target_x'])
        else:
            commander.move_to_target(current_x, current_y, 0.0, target['target_x']) 

if __name__ == "__main__":
    main()