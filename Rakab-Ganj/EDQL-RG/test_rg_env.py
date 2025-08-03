import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edql_rg_env import EDQLRGEnv
import numpy as np

def test_rg_environment():
    """Test the Rakab-Ganj EDQL environment"""
    print("Testing Rakab-Ganj EDQL Environment...")
    
    try:
        # Initialize environment
        env = EDQLRGEnv(
            net_file="rg.net.xml",
            route_file="rg.rou.xml",
            use_gui=False,  # Set to True for visual debugging
            max_steps=100
        )
        
        print("Environment initialized successfully!")
        
        # Test reset
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        
        # Test a few steps
        for step in range(10):
            # Random action
            action = np.random.randint(0, 8)
            obs, reward, done = env.step(action)
            
            print(f"Step {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Observation: {obs}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            
            if done:
                print("Episode completed!")
                break
        
        env.close()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rg_environment() 