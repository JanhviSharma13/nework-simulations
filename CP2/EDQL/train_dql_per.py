import os
from sumo_env import SumoEnvironment
from agent import DQLPERAgent

# === Setup Paths ===
SUMO_BINARY = "sumo-gui"  # Change to "sumo" for headless mode
ROOT = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(ROOT, "cp2.net.xml")
CONFIG_FILE = os.path.join(ROOT, "cp2.sumocfg")

# === Initialize Environment ===
env = SumoEnvironment(
    sumo_binary=SUMO_BINARY,
    config_file=CONFIG_FILE,
    net_file=NET_FILE,
    max_steps=500,
    gui=True  # Show SUMO GUI
)

# === Initialize Agent ===
agent = DQLPERAgent(
    actions_fn=env.get_valid_actions,
    alpha=0.5,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    buffer_size=2000,
    batch_size=32,
    priority_alpha=0.6
)

# === Training Parameters ===
n_episodes = 1000
log_every = 10
rewards = []

try:
    for ep in range(1, n_episodes + 1):
        try:
            state = env.reset()
            print(f"[EP {ep}] ✅ Reset complete. Start state: {state}")
        except Exception as e:
            print(f"[EP {ep}] ❌ Crash during env.reset(): {e}")
            continue

        if state is None:
            print(f"[EP {ep}] ⚠️ No valid route found. Skipping.")
            continue

        done = False
        ep_reward = 0
        steps = 0

        while not done:
            action = agent.select_action(state)

            if not action:
                print(f"[EP {ep}] ⚠️ No valid actions from state {state}. Skipping.")
                break

            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            steps += 1

        rewards.append(ep_reward)

        if ep % log_every == 0:
            avg = sum(rewards[-log_every:]) / log_every
            print(f"[EP {ep}] ✅ Total reward: {ep_reward:.2f} | Steps: {steps} | Avg (last {log_every}): {avg:.2f} | ε = {agent.epsilon:.4f}")

finally:
    print("[main] 🔚 Closing SUMO environment.")
    env.close()
