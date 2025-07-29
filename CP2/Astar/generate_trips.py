import subprocess
import os
import random

random.seed(42)

SUMO_TOOLS = r"C:\Users\Janhvi\Downloads\sumo\sumo-1.23.1\tools"

def generate_random_trips(net_file, output_trips, begin=0, end=1000, period=10):
    random_trips_py = os.path.join(SUMO_TOOLS, "randomTrips.py")

    result = subprocess.run([
        "python", random_trips_py,
        "-n", net_file,
        "-o", output_trips,
        "-b", str(begin),
        "-e", str(end),
        "-p", str(period),  # one vehicle every `period` seconds
        "--trip-attributes", "departLane='best' departSpeed='max' departPos='random'",
        "--validate",
        "--vehicle-class", "passenger",
        "--seed", "42"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Error running randomTrips.py:")
        print(result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args)
