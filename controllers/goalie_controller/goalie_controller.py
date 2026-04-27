from controller import Robot, Supervisor
import random
import math

class Observer:
    """Uses Supervisor God-Mode to track the ball."""
    def __init__(self, supervisor):
        self.supervisor = supervisor
        # Find the ball node using the DEF name we just set
        self.ball_node = self.supervisor.getFromDef("BALL")
        if self.ball_node is None:
            print("ERROR: Could not find BALL. Did you set the DEF name?", flush=True)
            
    def get_ball_data(self):
        if self.ball_node is None: return None
        
        pos = self.ball_node.getPosition() # Returns [x, y, z]
        vel = self.ball_node.getVelocity() # Returns [vx, vy, vz, wx, wy, wz]
        
        # We only care about 2D field data (X and Y)
        return {'x': pos[0], 'y': pos[1], 'vx': vel[0], 'vy': vel[1]}

class Strategist:
    """Calculates interception utilizing friction kinematics and the quadratic formula."""
    def __init__(self, intercept_x):
        self.intercept_x = intercept_x # Where the robot stands (dynamically 5.0)
        self.goal_net_x = 7.0          # Where the physical net is located
        self.deceleration = 1.5
        
    def _get_y_crossing(self, ball, target_x):
        """Helper function: Predicts the Y coordinate where the ball crosses a specific X line."""
        dx = target_x - ball['x']
        
        # Boomerang check
        if (dx > 0 and ball['vx'] <= 0) or (dx < 0 and ball['vx'] >= 0):
            return None
            
        v_mag = math.sqrt(ball['vx']**2 + ball['vy']**2)
        if v_mag < 0.05: 
            return None
            
        ax = -self.deceleration * (ball['vx'] / v_mag)
        ay = -self.deceleration * (ball['vy'] / v_mag)
        
        a = 0.5 * ax
        b = ball['vx']
        c = -dx
        
        discriminant = b**2 - (4 * a * c)
        if discriminant < 0 or a == 0: 
            return None
            
        t1 = (-b + math.sqrt(discriminant)) / (2 * a)
        t2 = (-b - math.sqrt(discriminant)) / (2 * a)
        
        valid_times = [t for t in (t1, t2) if t > 0]
        if not valid_times: 
            return None
            
        tti = min(valid_times)
        return ball['y'] + (ball['vy'] * tti) + (0.5 * ay * (tti**2))

    def calculate_interception(self, ball):
        if ball is None:
            return {'is_threat': False, 'target_y': 0.0}
            
        # 1. THREAT CHECK: Will it go in the actual net at X = 7.0?
        final_y = self._get_y_crossing(ball, self.goal_net_x)
        
        # If it doesn't reach the net, or goes wide of the 2.6m goal (-1.3 to 1.3)
        if final_y is None or abs(final_y) > 1.3:
            return {'is_threat': False, 'target_y': 0.0}
            
        # 2. INTERCEPT CHECK: Where should the robot block it at its current line?
        intercept_y = self._get_y_crossing(ball, self.intercept_x)
        
        if intercept_y is None:
            return {'is_threat': False, 'target_y': 0.0}
            
        return {'is_threat': True, 'target_y': intercept_y}

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
            
    def move_to_target(self, current_x, current_y, target_y, home_x):
        # 1. Y-axis PD Control (Lateral Strafing)
        error_y = target_y - current_y
        if abs(error_y) < 0.03:
            vy = 0.0
            self.prev_error_y = 0.0 
        else:
            derivative = error_y - self.prev_error_y
            vy = (self.Kp_y * error_y) + (self.Kd_y * derivative)
            self.prev_error_y = error_y
            vy = max(min(vy, 4.0), -4.0) 
            
        # 2. X-axis P Control (Stay on the goal line!)
        error_x = home_x - current_x
        
        # Add a deadzone so it doesn't panic over 1 millimeter drops
        if abs(error_x) < 0.03:
            vx = 0.0
        else:
            # FIX: Flipped the sign to negative to cure the positive feedback loop!
            vx = -error_x * 4.0 
            vx = max(min(vx, 2.0), -2.0)

        # 3. Full 2D Robotino Kinematics
        v0 = vy * 10.0
        v1 = (vx * 8.66) - (vy * 5.0)
        v2 = (-vx * 8.66) - (vy * 5.0)
        
        if self.m0: self.m0.setVelocity(max(min(v0, 6.0), -6.0))
        if self.m1: self.m1.setVelocity(max(min(v1, 6.0), -6.0))
        if self.m2: self.m2.setVelocity(max(min(v2, 6.0), -6.0))

class AutoShooter:
    """Instantly teleports the ball, waits for physics to settle, and shoots."""
    def __init__(self, supervisor, ball_node):
        self.supervisor = supervisor
        self.ball_node = ball_node
        
        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(int(self.supervisor.getBasicTimeStep()))
        
        if self.ball_node:
            self.trans_field = self.ball_node.getField("translation")
            
        self.frames_until_shoot = -1
        self.pending_vy = 0.0
        
    def check_and_shoot(self):
        if not self.ball_node: return
        
        if self.frames_until_shoot > 0:
            self.frames_until_shoot -= 1
            return
        elif self.frames_until_shoot == 0:
            # Fixed the velocity printout to match your 10.0m/s update
            self.ball_node.setVelocity([12.0, self.pending_vy, 0.0, 0.0, 0.0, 0.0])
            print(f"💥 CORNER SHOT FIRED! Speed: 15.0m/s | Angle Velocity: {self.pending_vy:.2f}m/s", flush=True)
            self.frames_until_shoot = -1 
            return

        key = self.keyboard.getKey()
        if key == ord(' '):
            self.trans_field.setSFVec3f([0.0, 0.0, 0.1])
            self.ball_node.resetPhysics()
            
            # This block now securely handles the hard corner angles
            if random.choice([True, False]):
                self.pending_vy = random.uniform(0.7, 1.9)   
            else:
                self.pending_vy = random.uniform(-1.9, -0.7) 
                
            self.frames_until_shoot = 2

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    observer = Observer(robot)
    
    robot_node = robot.getSelf()
    goal_x = robot_node.getPosition()[0]
    
    strategist = Strategist(goal_x)
    commander = Commander(robot)
    shooter = AutoShooter(robot, observer.ball_node)
    
    print("Omniscient Goalie Online. Click the 3D window and press SPACEBAR to fire a shot!", flush=True)
    
    while robot.step(timestep) != -1:
        shooter.check_and_shoot()
        
        ball_state = observer.get_ball_data()
        
        target = strategist.calculate_interception(ball_state)
        
        current_pos = robot_node.getPosition()
        current_x = current_pos[0]
        current_y = current_pos[1]
        
        if target['is_threat']:
            commander.move_to_target(current_x, current_y, target['target_y'], goal_x)
        else:
            commander.move_to_target(current_x, current_y, 0.0, goal_x)

if __name__ == "__main__":
    main()
