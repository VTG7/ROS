from controller import Robot, Supervisor

class Observer:
    def __init__(self, supervisor):
        self.supervisor = supervisor

class Strategist:
    def __init__(self):
        pass

class Commander:
    def __init__(self, robot):
        self.robot = robot

def main():
    # Initialize the Supervisor
    robot = Supervisor()
    
    # Get the time step of the current world
    timestep = int(robot.getBasicTimeStep())
    
    print("Goalie Controller Initialized. Waiting for kickoff...")
    
    # Main simulation loop
    while robot.step(timestep) != -1:
        # We will put our logic here soon!
        print("vsr debug line")
        pass

if __name__ == "__main__":
    main()
