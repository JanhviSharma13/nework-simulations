# 🚀 Greedy Travel-Time Routing Agent

## 📋 Overview

This implementation provides a **Greedy Travel-Time Routing Agent** that selects the next edge at each decision point based on minimum estimated travel time to the destination, using SUMO and TraCI.

The agent is tested across **4 differently dense subregions of Delhi**:
- **Rakab Ganj** - Mixed residential-commercial area
- **Safdarjung** - Residential area with moderate density  
- **CP2** - High-density commercial area
- **Chandni Chowk** - Historic area with complex traffic patterns

## 🏗️ Architecture

### Core Components

1. **`greedy_routing_agent.py`** - Main agent implementation
2. **`run_greedy.py`** - Testing framework for all areas
3. **`test_greedy_simple.py`** - Simple test script for single area

### Agent Features

- ✅ **Travel-time based routing** using TraCI's real-time data
- ✅ **Congestion detection** and penalty application
- ✅ **Dynamic rerouting** at decision points
- ✅ **Comprehensive logging** of all metrics
- ✅ **Success/failure tracking** with detailed statistics

## 📊 Logged Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `episode` | int | Episode number (fixed at 1 for deterministic runs) |
| `map_name` | str | Name of the subregion |
| `start_edge` | str | Where the vehicle started from |
| `goal_edge` | str | Final destination |
| `success` | bool/int | 1 if agent reached goal, 0 otherwise |
| `total_time` | float | Total time taken from start to destination (seconds) |
| `route_length` | int | Number of edges visited in total |
| `total_distance` | float | Total physical distance traveled |
| `decision_count` | int | How many greedy decisions (reroutes) were made |
| `final_edge` | str | Final edge reached (to confirm success) |
| `stuck_time` | float | Time agent spent without progress |
| `congestion_event` | bool/int | 1 if a mid-episode blockage was active |
| `route_edges` | list | List of all edge IDs traversed |
| `reroute_timestamps` | list | Time steps when the agent rerouted |
| `decision_latency_avg` | float | Average time per routing decision (ms) |
| `internal_edges_skipped` | int | How many internal edges were skipped |

## 🚀 Usage

### Phase 1: Simple Test (Single Area)

```bash
python test_greedy_simple.py
```

This will test the agent on **Rakab Ganj** with a simple start-goal pair.

### Phase 2: Full Testing (All Areas)

```bash
python run_greedy.py
```

This will:
1. Test on **Rakab Ganj** first (3 tests)
2. Test on all **4 areas** (3 tests each)
3. Generate comprehensive results analysis

## 📁 Output Files

- **`greedy_results_YYYYMMDD_HHMMSS.csv`** - Main results file
- **`test_greedy_results.csv`** - Simple test results

## 🎯 Workflow

### Agent Decision Process

1. **Load SUMO config** per map (*.sumocfg)
2. **Insert greedy agent** with `traci.vehicle.add()`
3. **At each timestep**:
   - Check if agent is near end of edge
   - Call `greedy_decision_step()`
   - Rebuild route using current → best → final leg
4. **Log stats**: total travel time, route length, edge list, success/failure

### Testing Strategy

#### Stage 1: Single Area Testing
- Test on **Rakab Ganj** only
- Verify agent reaches destination
- Check route adaptation per step
- Validate logging functionality

#### Stage 2: Multi-Area Testing
- Test on all **4 Delhi areas**
- Compare performance across different densities
- Generate area-specific analysis

## 📈 Performance Analysis

The implementation provides:

### Success Metrics
- **Success Rate**: Percentage of successful goal completions
- **Average Travel Time**: Mean time to reach destination
- **Route Efficiency**: Average route length vs optimal

### Decision Quality
- **Decision Count**: Number of reroutes made
- **Decision Latency**: Average time per routing decision
- **Congestion Handling**: Detection and response to traffic

### Area Comparison
- **Per-area statistics** for all 4 Delhi subregions
- **Density impact analysis** on routing performance
- **Network topology effects** on greedy decisions

## 🔧 Configuration

### Agent Parameters

```python
agent = GreedyTravelTimeAgent(
    vehicle_id="greedy_agent",
    reroute_threshold=50  # meters from edge end to trigger reroute
)
```

### Testing Parameters

```python
# In run_greedy.py
tester.run_all_areas(tests_per_area=3)  # 3 tests per area
```

## 🐛 Troubleshooting

### Common Issues

1. **"Not enough valid edges"**
   - Check network file exists and is valid
   - Verify SUMO installation

2. **"Vehicle stuck"**
   - Increase `max_steps` parameter
   - Check network connectivity

3. **"Error in test"**
   - Verify SUMO configuration files
   - Check file paths are correct

### Debug Mode

Add debug prints in `greedy_routing_agent.py`:

```python
print(f"Current edge: {current_edge}")
print(f"Outgoing edges: {outgoing_edges}")
print(f"Estimated time: {estimated_time}")
```

## 📊 Expected Results

### Success Rates
- **Rakab Ganj**: 80-90% (moderate density)
- **Safdarjung**: 85-95% (residential, predictable)
- **CP2**: 70-85% (high density, complex)
- **Chandni Chowk**: 75-90% (historic, variable)

### Performance Metrics
- **Average Travel Time**: 100-300 seconds
- **Route Length**: 5-15 edges
- **Decision Count**: 2-8 reroutes per trip

## 🔄 Future Enhancements

1. **Advanced Routing Algorithms**
   - A* integration
   - Multi-criteria optimization
   - Real-time traffic prediction

2. **Enhanced Metrics**
   - Fuel consumption estimation
   - CO2 emission tracking
   - Economic cost analysis

3. **Visualization**
   - Route heatmaps
   - Decision point analysis
   - Performance comparison charts

## 📝 Dependencies

- **SUMO** (Simulation of Urban MObility)
- **TraCI** (Traffic Control Interface)
- **Python 3.7+**
- **NumPy** (for calculations)
- **Pandas** (for analysis, optional)

## 🎯 Quick Start

```bash
# 1. Test single area
python test_greedy_simple.py

# 2. Run full testing
python run_greedy.py

# 3. Check results
cat greedy_results_*.csv
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify SUMO installation
3. Test with simple configuration first
4. Review log files for error details 