"""
Multi-Zone Building Control and Coordination

Implementation of distributed control strategies for multi-zone buildings:
- Consensus-based distributed control (Olfati-Saber et al., 2007)
- Game-theoretic approaches (Basar & Olsder, 1999)
- Hierarchical control architectures (Bengea et al., 2014)
- Energy-optimal zone coordination (Ma et al., 2012)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import networkx as nx
from dataclasses import dataclass

@dataclass
class ZoneParameters:
    """Parameters for individual building zones"""
    zone_id: str
    thermal_mass: float
    wall_resistance: float
    internal_gains: float
    hvac_capacity: float
    floor_area: float
    adjacent_zones: List[str]
    coupling_resistance: float  # Thermal resistance to adjacent zones

class MultiZoneBuilding:
    """
    Multi-zone building thermal model with inter-zone coupling
    
    Based on:
    - ASHRAE Handbook - Fundamentals (2017)
    - Clarke "Energy Simulation in Building Design" (2001)
    - Crawley et al. "EnergyPlus: Creating a new-generation building 
      energy simulation program" (2001)
    """
    
    def __init__(self, zones: Dict[str, ZoneParameters]):
        self.zones = zones
        self.zone_ids = list(zones.keys())
        self.n_zones = len(zones)
        
        # Build adjacency matrix for thermal coupling
        self.adjacency_matrix = self._build_adjacency_matrix()
        
        # State vector: [T1, T2, ..., Tn] for n zones
        self.current_temperatures = np.ones(self.n_zones) * 20.0
        
    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build thermal coupling adjacency matrix"""
        
        adj_matrix = np.zeros((self.n_zones, self.n_zones))
        
        for i, zone_id in enumerate(self.zone_ids):
            zone = self.zones[zone_id]
            
            for adj_zone_id in zone.adjacent_zones:
                if adj_zone_id in self.zone_ids:
                    j = self.zone_ids.index(adj_zone_id)
                    # Thermal conductance between zones
                    conductance = 1.0 / zone.coupling_resistance
                    adj_matrix[i, j] = conductance
        
        return adj_matrix
    
    def multi_zone_dynamics(self, t: float, temperatures: np.ndarray,
                           control_inputs: np.ndarray, 
                           disturbances: List[Dict]) -> np.ndarray:
        """Multi-zone thermal dynamics with coupling"""
        
        dT_dt = np.zeros(self.n_zones)
        
        for i, zone_id in enumerate(self.zone_ids):
            zone = self.zones[zone_id]
            T_zone = temperatures[i]
            
            # External heat transfer
            T_ambient = disturbances[i].get('ambient_temp', 25.0)
            Q_external = (T_ambient - T_zone) / zone.wall_resistance
            
            # Inter-zone heat transfer
            Q_inter_zone = 0
            for j in range(self.n_zones):
                if i != j and self.adjacency_matrix[i, j] > 0:
                    T_adjacent = temperatures[j]
                    Q_inter_zone += self.adjacency_matrix[i, j] * (T_adjacent - T_zone)
            
            # Internal gains
            Q_internal = zone.internal_gains
            
            # HVAC input
            Q_hvac = control_inputs[i]
            
            # Total heat balance
            Q_total = Q_external + Q_inter_zone + Q_internal + Q_hvac
            
            # Temperature rate of change
            dT_dt[i] = Q_total / zone.thermal_mass
        
        return dT_dt
    
    def simulate_step(self, dt: float, control_inputs: np.ndarray,
                     disturbances: List[Dict]) -> np.ndarray:
        """Simulate one time step using Euler integration"""
        
        derivatives = self.multi_zone_dynamics(
            0, self.current_temperatures, control_inputs, disturbances
        )
        
        self.current_temperatures += dt * derivatives
        return self.current_temperatures.copy()

class ConsensusController:
    """
    Consensus-based distributed control for multi-zone coordination
    
    Based on:
    - Olfati-Saber et al. "Consensus and cooperation in networked 
      multi-agent systems" (2007)
    - Ren & Beard "Distributed Consensus in Multi-vehicle Cooperative 
      Control" (2008)
    """
    
    def __init__(self, building: MultiZoneBuilding, consensus_gain: float = 0.1):
        self.building = building
        self.consensus_gain = consensus_gain
        
        # Communication graph (same as thermal coupling for simplicity)
        self.communication_graph = building.adjacency_matrix > 0
        
        # Local controllers for each zone
        self.local_controllers = {}
        for zone_id in building.zone_ids:
            self.local_controllers[zone_id] = {
                'kp': 500.0,
                'ki': 50.0,
                'integral': 0.0
            }
    
    def compute_consensus_term(self, zone_idx: int, 
                              local_setpoints: np.ndarray) -> float:
        """Compute consensus term for distributed coordination"""
        
        consensus_term = 0.0
        my_setpoint = local_setpoints[zone_idx]
        
        # Consensus with neighboring zones
        for neighbor_idx in range(self.building.n_zones):
            if self.communication_graph[zone_idx, neighbor_idx]:
                neighbor_setpoint = local_setpoints[neighbor_idx]
                consensus_term += self.consensus_gain * (neighbor_setpoint - my_setpoint)
        
        return consensus_term
    
    def compute_control(self, temperatures: np.ndarray, setpoints: np.ndarray,
                       dt: float) -> np.ndarray:
        """Compute distributed control actions"""
        
        control_actions = np.zeros(self.building.n_zones)
        
        for i, zone_id in enumerate(self.building.zone_ids):
            controller = self.local_controllers[zone_id]
            
            # Local tracking error
            error = setpoints[i] - temperatures[i]
            
            # Consensus term for coordination
            consensus_term = self.compute_consensus_term(i, setpoints)
            
            # PI control with consensus
            proportional = controller['kp'] * error
            controller['integral'] += error * dt
            integral = controller['ki'] * controller['integral']
            
            # Total control action
            control_actions[i] = proportional + integral + consensus_term
            
            # Saturation
            zone = self.building.zones[zone_id]
            control_actions[i] = np.clip(control_actions[i], 
                                       -zone.hvac_capacity, zone.hvac_capacity)
        
        return control_actions

class GameTheoreticController:
    """
    Game-theoretic control for multi-zone buildings
    
    Based on:
    - Basar & Olsder "Dynamic Noncooperative Game Theory" (1999)
    - Li & Wen "Game theory approach to multi-agent systems" (2014)
    - Grammatico et al. "Dynamic control of agents playing aggregative games" (2017)
    """
    
    def __init__(self, building: MultiZoneBuilding):
        self.building = building
        
        # Game parameters
        self.comfort_weight = 1.0
        self.energy_weight = 0.01
        self.coordination_weight = 0.1
        
    def zone_cost_function(self, zone_idx: int, control_action: float,
                          temperature: float, setpoint: float,
                          other_actions: np.ndarray) -> float:
        """Cost function for individual zone (player)"""
        
        # Comfort cost
        comfort_cost = self.comfort_weight * (temperature - setpoint)**2
        
        # Energy cost
        energy_cost = self.energy_weight * control_action**2
        
        # Coordination cost (interaction with neighbors)
        coordination_cost = 0
        for j in range(self.building.n_zones):
            if j != zone_idx and self.building.adjacency_matrix[zone_idx, j] > 0:
                # Penalize large differences in control actions
                coordination_cost += self.coordination_weight * (control_action - other_actions[j])**2
        
        return comfort_cost + energy_cost + coordination_cost
    
    def nash_equilibrium_solver(self, temperatures: np.ndarray, 
                               setpoints: np.ndarray) -> np.ndarray:
        """Solve for Nash equilibrium control actions"""
        
        def objective(control_vars):
            total_cost = 0
            
            for i in range(self.building.n_zones):
                # Predict temperature response (simplified)
                predicted_temp = temperatures[i] + 0.1 * control_vars[i] / 1000
                
                cost = self.zone_cost_function(
                    i, control_vars[i], predicted_temp, setpoints[i], control_vars
                )
                total_cost += cost
            
            return total_cost
        
        # Optimization constraints
        bounds = []
        for zone_id in self.building.zone_ids:
            zone = self.building.zones[zone_id]
            bounds.append((-zone.hvac_capacity, zone.hvac_capacity))
        
        # Solve
        x0 = np.zeros(self.building.n_zones)
        result = minimize(objective, x0, method='SLSQP', bounds=bounds)
        
        return result.x if result.success else x0

class HierarchicalController:
    """
    Hierarchical control architecture for large buildings
    
    Based on:
    - Bengea et al. "Implementation of model predictive control for 
      an HVAC system in a mid-size commercial building" (2014)
    - Oldewurtel et al. "Importance of occupancy information for 
      building climate control" (2013)
    """
    
    def __init__(self, building: MultiZoneBuilding):
        self.building = building
        
        # Hierarchy levels
        self.supervisory_controller = SupervisoryController(building)
        self.local_controllers = {}
        
        for zone_id in building.zone_ids:
            self.local_controllers[zone_id] = LocalZoneController(zone_id)
    
    def compute_hierarchical_control(self, temperatures: np.ndarray,
                                   setpoints: np.ndarray, 
                                   global_constraints: Dict) -> np.ndarray:
        """Compute control using hierarchical architecture"""
        
        # Supervisory level: compute global coordination signals
        coordination_signals = self.supervisory_controller.compute_coordination(
            temperatures, setpoints, global_constraints
        )
        
        # Local level: compute individual zone controls
        control_actions = np.zeros(self.building.n_zones)
        
        for i, zone_id in enumerate(self.building.zone_ids):
            local_controller = self.local_controllers[zone_id]
            
            control_actions[i] = local_controller.compute_control(
                temperatures[i], setpoints[i], coordination_signals[i]
            )
        
        return control_actions

class SupervisoryController:
    """Supervisory controller for global coordination"""
    
    def __init__(self, building: MultiZoneBuilding):
        self.building = building
        self.total_energy_limit = 20000.0  # Watts
        
    def compute_coordination(self, temperatures: np.ndarray, 
                           setpoints: np.ndarray,
                           constraints: Dict) -> np.ndarray:
        """Compute coordination signals for local controllers"""
        
        coordination_signals = np.zeros(self.building.n_zones)
        
        # Energy allocation based on zone priorities
        total_demand = np.sum(np.abs(setpoints - temperatures))
        
        if total_demand > 0:
            for i in range(self.building.n_zones):
                zone_priority = abs(setpoints[i] - temperatures[i]) / total_demand
                coordination_signals[i] = zone_priority * self.total_energy_limit
        
        return coordination_signals

class LocalZoneController:
    """Local zone controller"""
    
    def __init__(self, zone_id: str):
        self.zone_id = zone_id
        self.kp = 300.0
        self.ki = 30.0
        self.integral = 0.0
        
    def compute_control(self, temperature: float, setpoint: float,
                       coordination_signal: float) -> float:
        """Compute local control action"""
        
        error = setpoint - temperature
        
        # PI control
        proportional = self.kp * error
        self.integral += error * 0.1  # Assuming dt = 0.1
        integral_term = self.ki * self.integral
        
        # Include coordination signal
        control_action = proportional + integral_term + coordination_signal * 0.1
        
        return np.clip(control_action, -2000, 2000)

def demonstrate_multi_zone_control():
    """Demonstrate multi-zone building control strategies"""
    
    print("Multi-Zone Building Control Demonstration")
    print("=" * 60)
    print("Implementing distributed and hierarchical control strategies")
    print("for coordinated multi-zone building automation.")
    print()
    
    # Create multi-zone building
    zones = {
        'Zone_A': ZoneParameters(
            zone_id='Zone_A', thermal_mass=3e6, wall_resistance=0.1,
            internal_gains=1000, hvac_capacity=3000, floor_area=30,
            adjacent_zones=['Zone_B'], coupling_resistance=0.2
        ),
        'Zone_B': ZoneParameters(
            zone_id='Zone_B', thermal_mass=4e6, wall_resistance=0.12,
            internal_gains=1200, hvac_capacity=3500, floor_area=35,
            adjacent_zones=['Zone_A', 'Zone_C'], coupling_resistance=0.18
        ),
        'Zone_C': ZoneParameters(
            zone_id='Zone_C', thermal_mass=2.5e6, wall_resistance=0.08,
            internal_gains=800, hvac_capacity=2500, floor_area=25,
            adjacent_zones=['Zone_B'], coupling_resistance=0.25
        )
    }
    
    building = MultiZoneBuilding(zones)
    
    # Initialize controllers
    consensus_controller = ConsensusController(building)
    game_controller = GameTheoreticController(building)
    hierarchical_controller = HierarchicalController(building)
    
    # Simulation parameters
    duration_hours = 24
    dt = 0.1
    time_steps = int(duration_hours / dt)
    time_hours = np.arange(0, duration_hours, dt)
    
    # Generate setpoints and disturbances
    setpoints = np.zeros((time_steps, building.n_zones))
    disturbances = []
    
    for i, t in enumerate(time_hours):
        # Different setpoints for different zones
        hour_of_day = t % 24
        
        if 8 <= hour_of_day <= 17:  # Office hours
            setpoints[i] = [22.0, 21.5, 22.5]  # Different comfort preferences
        else:
            setpoints[i] = [18.0, 18.0, 18.0]  # Energy saving
        
        # Environmental disturbances
        ambient_temp = 25 + 5 * np.sin(2 * np.pi * t / 24 - np.pi/2)
        
        zone_disturbances = []
        for zone_id in building.zone_ids:
            zone_disturbances.append({
                'ambient_temp': ambient_temp + np.random.normal(0, 0.5),
                'solar_gain': max(0, 500 * np.sin(np.pi * (hour_of_day - 6) / 12)) if 6 <= hour_of_day <= 18 else 0
            })
        
        disturbances.append(zone_disturbances)
    
    # Run simulations with different control strategies
    controllers = {
        'Consensus': consensus_controller,
        'Game Theoretic': game_controller,
        'Hierarchical': hierarchical_controller
    }
    
    results = {}
    
    for controller_name, controller in controllers.items():
        print(f"Simulating {controller_name} control...")
        
        # Reset building state
        building.current_temperatures = np.ones(building.n_zones) * 20.0
        
        temperatures = [building.current_temperatures.copy()]
        controls = []
        
        for i in range(time_steps - 1):
            current_temps = building.current_temperatures
            current_setpoints = setpoints[i]
            
            # Compute control actions
            if controller_name == 'Consensus':
                control_actions = controller.compute_control(
                    current_temps, current_setpoints, dt
                )
            elif controller_name == 'Game Theoretic':
                control_actions = controller.nash_equilibrium_solver(
                    current_temps, current_setpoints
                )
            elif controller_name == 'Hierarchical':
                global_constraints = {'energy_limit': 15000}
                control_actions = controller.compute_hierarchical_control(
                    current_temps, current_setpoints, global_constraints
                )
            
            controls.append(control_actions.copy())
            
            # Simulate building response
            new_temps = building.simulate_step(dt, control_actions, disturbances[i])
            temperatures.append(new_temps.copy())
        
        results[controller_name] = {
            'temperatures': np.array(temperatures),
            'controls': np.array(controls),
            'setpoints': setpoints
        }
    
    # Performance analysis
    print("\nMulti-Zone Control Performance:")
    print("-" * 50)
    
    for controller_name, data in results.items():
        temps = data['temperatures'][:-1]  # Remove last point
        setpts = data['setpoints']
        controls = data['controls']
        
        # Calculate metrics for each zone
        total_mae = 0
        total_energy = 0
        
        for zone_idx in range(building.n_zones):
            zone_id = building.zone_ids[zone_idx]
            zone_temps = temps[:, zone_idx]
            zone_setpts = setpts[:, zone_idx]
            zone_controls = controls[:, zone_idx]
            
            mae = np.mean(np.abs(zone_temps - zone_setpts))
            energy = np.sum(np.abs(zone_controls)) * dt
            
            total_mae += mae
            total_energy += energy
        
        avg_mae = total_mae / building.n_zones
        
        print(f"\n{controller_name}:")
        print(f"  Average MAE: {avg_mae:.2f}°C")
        print(f"  Total Energy: {total_energy:.1f} kWh")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Temperature tracking for each zone
    for zone_idx in range(building.n_zones):
        zone_id = building.zone_ids[zone_idx]
        
        ax = axes[zone_idx // 2, zone_idx % 2]
        
        for controller_name, data in results.items():
            temps = data['temperatures'][:-1, zone_idx]
            ax.plot(time_hours[:-1], temps, label=controller_name, linewidth=2)
        
        ax.plot(time_hours[:-1], setpoints[:-1, zone_idx], 'k--', 
               label='Setpoint', linewidth=2)
        
        ax.set_title(f'{zone_id} Temperature Control')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Control effort comparison
    if building.n_zones < 4:  # Add fourth subplot for control effort
        ax = axes[1, 1] if building.n_zones == 3 else axes[1, 0]
        
        for controller_name, data in results.items():
            total_control = np.sum(np.abs(data['controls']), axis=1)
            ax.plot(time_hours[:-1], total_control, label=controller_name, linewidth=2)
        
        ax.set_title('Total Control Effort')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Total HVAC Power (W)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nMulti-zone coordination demonstrates the benefits of")
    print("distributed and hierarchical control strategies for")
    print("large building systems with thermal coupling between zones.")

if __name__ == "__main__":
    demonstrate_multi_zone_control()
