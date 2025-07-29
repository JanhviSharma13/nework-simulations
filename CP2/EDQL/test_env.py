print("👣 Step 0: Importing")
from sumo_env import SumoEnvironment
import os

print("👣 Step 1: Setting paths")
SUMO_BINARY = "sumo-gui"
ROOT = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(ROOT, "cp2.net.xml")
CONFIG_FILE = os.path.join(ROOT, "cp2.sumocfg")

print("👣 Step 2: Creating environment object")
env = SumoEnvironment(
    sumo_binary=SUMO_BINARY,
    config_file=CONFIG_FILE,
    net_file=NET_FILE,
    max_steps=500,
    gui=True
)

print("👣 Step 3: Calling reset()...")
state = env.reset()
print("✅ Reset returned:", state)

env.close()
