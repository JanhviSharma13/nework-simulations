# EDQL for Rakab-Ganj Network

This directory contains the Enhanced Double Q-Learning (EDQL) implementation specifically scaled for the Rakab-Ganj network, with TD error logging and histogram support.

## 📁 File Structure

```
EDQL-RG/
├── edql_rg_env.py          # EDQL environment for Rakab-Ganj
├── train_edql_rg.py        # Training script with TD error logging
├── test_rg_env.py          # Test script for environment
├── README.md               # This file
├── rg.net.xml             # SUMO network file
└── rg.rou.xml             # SUMO route file
```

## 🚀 Features

### ✅ TD Error Logging
- **Per-sample TD error calculation** during training
- **Episode-level statistics**: mean, std, min, max TD errors
- **CSV logging** with comprehensive TD error metrics
- **Real-time monitoring** of Q-value stability

### 📊 Histogram Support
- **Automatic histogram generation** every 50 episodes
- **Statistical annotations** on plots (mean, std, min, max)
- **High-resolution plots** saved to `td_error_plots/` directory
- **Visual verification** of PER behavior and Q-value differences

### 🔧 Network Scaling
- **7,663 edges** handled efficiently
- **Dynamic route validation** using available network edges
- **Enhanced observations** with edge progress tracking
- **Robust error handling** for large networks

## 🎯 Key Improvements

### 1. Enhanced Observation Space
```python
# Original: [speed, lane_pos, lane_index]
# Rakab-Ganj: [speed, lane_pos, lane_index, edge_progress]
```

### 2. TD Error Monitoring
```python
# Per-episode TD error statistics
td_stats = {
    'mean': float(np.mean(td_errors)),
    'std': float(np.std(td_errors)),
    'min': float(np.min(td_errors)),
    'max': float(np.max(td_errors))
}
```

### 3. Histogram Visualization
```python
# Automatic plotting every 50 episodes
plot_td_error_histogram(episode_td_errors, episode)
```

## 🚗 Usage

### 1. Test Environment
```bash
cd Rakab-Ganj/EDQL-RG
python test_rg_env.py
```

### 2. Start Training
```bash
cd Rakab-Ganj/EDQL-RG
python train_edql_rg.py
```

### 3. Monitor Progress
- **Console output**: Episode progress and TD error stats
- **CSV logs**: Detailed training metrics
- **Histogram plots**: Visual TD error distribution

## 📈 Output Files

### Training Logs
- `edql_rg_training_log_YYYYMMDD_HHMMSS.csv`
- Contains episode data with TD error statistics

### Model Files
- `trained_edql_rg_qnet1_YYYYMMDD_HHMMSS.pth`
- `trained_edql_rg_qnet2_YYYYMMDD_HHMMSS.pth`

### Histogram Plots
- `td_error_plots/td_error_episode_X.png`
- Generated every 50 episodes

## 🔍 TD Error Analysis

### What to Monitor
1. **TD Error Mean**: Should decrease over time (learning progress)
2. **TD Error Std**: Should stabilize (convergence)
3. **Q1/Q2 Differences**: Should show meaningful variations (EDQL working)
4. **Histogram Shape**: Should become more concentrated (learning stability)

### Red Flags
- **Increasing TD errors**: Learning instability
- **No Q1/Q2 differences**: EDQL not working properly
- **Wide histograms**: Poor convergence

## 🛠️ Configuration

### Environment Parameters
```python
env = EDQLRGEnv(
    net_file="rg.net.xml",
    route_file="rg.rou.xml",
    use_gui=False,      # Set to True for debugging
    max_steps=2000      # Increased for larger network
)
```

### Training Parameters
```python
episodes = 1000
batch_size = 64
epsilon_decay = 0.995
gamma = 0.99
tau = 0.005
```

## 📊 Network Statistics

- **Total Edges**: 7,663
- **Route Edges**: 8 (subset for training)
- **Observation Space**: 4-dimensional
- **Action Space**: 8 discrete actions
- **Max Steps**: 2,000 (scaled for larger network)

## 🔧 Troubleshooting

### Common Issues
1. **Vehicle not found**: Check route edge validity
2. **TD errors too high**: Reduce learning rate or increase batch size
3. **No convergence**: Check reward function and network architecture

### Debug Mode
```python
env = EDQLRGEnv(use_gui=True)  # Visual debugging
```

## 📚 Next Steps

1. **Run training** and monitor TD error trends
2. **Analyze histograms** for learning stability
3. **Compare Q1/Q2** differences to verify EDQL
4. **Scale to other networks** using this framework

## 🎯 Success Metrics

- **TD Error Mean**: Decreasing trend
- **TD Error Std**: Stabilizing over time
- **Q1/Q2 Differences**: Meaningful variations
- **Histogram Concentration**: More focused distributions
- **Episode Rewards**: Increasing trend 