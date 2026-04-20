from controller import Robot, Supervisor
import random

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
    """Calculates exactly where the ball will cross the goal line."""
    def __init__(self, goal_x):
        self.goal_x = goal_x # The x-coordinate our goalie is standing on
        
    def calculate_interception(self, ball):
        # If ball isn't moving along the X-axis, it's not a threat
        if ball is None or ball['vx'] == 0:
            return {'is_threat': False, 'target_y': 0.0}
            
        # Calculate Time-to-Intercept (TTI): Time = Distance / Speed
        tti = (self.goal_x - ball['x']) / ball['vx']
        
        # If TTI is negative, the ball is moving away from our goal
        if tti < 0:
            return {'is_threat': False, 'target_y': 0.0}
            
        # Predict the exact Y-coordinate where the ball crosses the goal line
        target_y = ball['y'] + (ball['vy'] * tti)
        
        # Threat Check: Are those Y-coordinates within the goal posts?
        # A standard RoboCup goal is about 1.4m wide (-0.7 to 0.7)
        if abs(target_y) < 0.7:
            return {'is_threat': True, 'target_y': target_y}
        else:
            return {'is_threat': False, 'target_y': 0.0}

class Commander:
    """Translates the target coordinate into smooth omni-wheel motor commands using a PD controller."""
    def __init__(self, robot):
        self.robot = robot
        # Get the 3 omni-wheel motors of the Robotino
        self.m0 = robot.getDevice("wheel0_joint")
        self.m1 = robot.getDevice("wheel1_joint")
        self.m2 = robot.getDevice("wheel2_joint")
        
        # Set all motors to velocity control mode
        for m in [self.m0, self.m1, self.m2]:
            if m is not None:
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                
        # PD Controller variables
        self.prev_error = 0.0
        self.Kp = 8.0  # Proportional gain: How aggressively to move
        self.Kd = 2.0  # Derivative gain: How aggressively to brake/dampen
            
    def move_to_y(self, current_y, target_y):
        # Calculate distance to target
        error = target_y - current_y
        
        # Stop if we are within 3cm of the target
        if abs(error) < 0.03:
            vy = 0.0
            self.prev_error = 0.0 # Reset derivative when stopped
        else:
            # PD Control Math
            derivative = error - self.prev_error
            vy = (self.Kp * error) + (self.Kd * derivative)
            self.prev_error = error
            
            # Cap the maximum strafing speed to prevent physics glitches
            vy = max(min(vy, 4.0), -4.0) 
            
        # Robotino Kinematics for purely lateral movement
        v0 = vy * 10.0
        v1 = -vy * 5.0
        v2 = -vy * 5.0
        
        # Send speeds to motors, capped at max rad/s
        if self.m0: self.m0.setVelocity(max(min(v0, 6.0), -6.0))
        if self.m1: self.m1.setVelocity(max(min(v1, 6.0), -6.0))
        if self.m2: self.m2.setVelocity(max(min(v2, 6.0), -6.0))

class AutoShooter:
    """Instantly teleports and shoots the ball when Spacebar is pressed."""
    def __init__(self, supervisor, ball_node):
        self.supervisor = supervisor
        self.ball_node = ball_node
        
        self.keyboard = self.supervisor.getKeyboard()
        self.keyboard.enable(int(self.supervisor.getBasicTimeStep()))
        
        if self.ball_node:
            self.trans_field = self.ball_node.getField("translation")
        
    def check_and_shoot(self):
        if not self.ball_node: return
        
        key = self.keyboard.getKey()
        
        # ASCII 32 is the spacebar
        if key == ord(' '):
            self.trans_field.setSFVec3f([0.0, 0.0, 0.2])
            self.ball_node.resetPhysics()
            
            random_vy = random.uniform(-1.5, 1.5)
            
            # Assuming your robot is at positive X (e.g. 4.5). 
            # If your robot is at negative X (-4.5), change 5.0 to -5.0!
            self.ball_node.setVelocity([5.0, random_vy, 0.0, 0.0, 0.0, 0.0])
            
            print(f"💥 SHOT FIRED! Speed: 5.0m/s | Angle Velocity: {random_vy:.2f}m/s", flush=True)

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
        
        current_y = robot_node.getPosition()[1]
        if target['is_threat']:
            commander.move_to_y(current_y, target['target_y'])
        else:
            commander.move_to_y(current_y, 0.0) 

if __name__ == "__main__":
    main()

# def main():
#     robot = Supervisor()
#     timestep = int(robot.getBasicTimeStep())
    
#     observer = Observer(robot)
    
#     # Let's dynamically find out where our robot was placed on the field
#     robot_node = robot.getSelf()
#     goal_x = robot_node.getPosition()[0]
    
#     strategist = Strategist(goal_x)
#     commander = Commander(robot)
    
#     print("Omniscient Goalie Online. Ready for shots!", flush=True)
    
#     while robot.step(timestep) != -1:
#         ball_state = observer.get_ball_data()
        
#         # 1. Think
#         target = strategist.calculate_interception(ball_state)
        
#         # 2. Act
#         current_y = robot_node.getPosition()[1]
#         if target['is_threat']:
#             commander.move_to_y(current_y, target['target_y'])
#         else:
#             commander.move_to_y(current_y, 0.0) # Stay in the center

# if __name__ == "__main__":
#     main()
