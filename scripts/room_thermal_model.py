"""
Room Thermal Dynamics and Control System Simulation

Based on research from:
- "Building Energy Management Systems" by Levermore (2000)
- "Model Predictive Control for Energy Efficient Buildings" by Oldewurtel et al. (2012)
- RC thermal network models from ASHRAE standards
- Control strategies from IEEE Control Systems Magazine papers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

@dataclass
class RoomParameters:
    """Physical parameters of the room based on building physics research"""
    # Thermal capacitance (J/K) - from ASHRAE Handbook
    thermal_mass: float = 5e6  # Typical office room
    
    # Thermal resistances (K/W) - RC network model
    wall_resistance: float = 0.1    # Wall to ambient
    window_resistance: float = 0.05  # Window to ambient
    internal_resistance: float = 0.02  # Internal thermal resistance
    
    # Areas (m²)
    wall_area: float = 50.0
    window_area: float = 10.0
    floor_area: float = 25.0
    
    # HVAC system parameters
    hvac_max_power: float = 5000.0  # Watts
    hvac_efficiency: float = 0.9
    
    # Internal heat gains (W) - ASHRAE 90.1 standard
    occupancy_gain: float = 100.0  # Per person
    lighting_gain: float = 10.0    # Per m²
    equipment_gain: float = 15.0   # Per m²

@dataclass
class EnvironmentalConditions:
    """External environmental conditions"""
    ambient_temperature: float = 25.0  # Celsius
    solar_irradiance: float = 500.0    # W/m²
    wind_speed: float = 2.0            # m/s
    humidity: float = 50.0             # %

class ThermalModel:
    """
    RC thermal network model for room dynamics
    
    Based on the lumped capacitance method from:
    - Gouda et al. "Building thermal model reduction using nonlinear 
      constrained optimization" (2002)
    - Underwood & Yik "Modelling Methods for Energy in Buildings" (2004)
    """
    
    def __init__(self, params: RoomParameters):
        self.params = params
        self.state_history = []
        
    def thermal_dynamics(self, t: float, state: np.ndarray, 
                        control_input: float, disturbances: Dict) -> np.ndarray:
        """
        Thermal dynamics differential equation
        
        dT/dt = (1/C) * [Q_hvac + Q_solar + Q_internal + Q_conduction]
        
        Where:
        - C: thermal capacitance
        - Q_hvac: HVAC heat input
        - Q_solar: solar heat gain
        - Q_internal: internal heat gains
        - Q_conduction: heat transfer through building envelope
        """
        T_room = state[0]
        T_ambient = disturbances.get('ambient_temp', 25.0)
        solar_gain = disturbances.get('solar_gain', 0.0)
        occupancy = disturbances.get('occupancy', 1.0)
        
        # Heat transfer through building envelope (W)
        Q_walls = (T_ambient - T_room) / self.params.wall_resistance
        Q_windows = (T_ambient - T_room) / self.params.window_resistance
        Q_conduction = Q_walls + Q_windows
        
        # Solar heat gain (W)
        Q_solar = solar_gain * self.params.window_area * 0.7  # SHGC = 0.7
        
        # Internal heat gains (W)
        Q_occupancy = self.params.occupancy_gain * occupancy
        Q_lighting = self.params.lighting_gain * self.params.floor_area
        Q_equipment = self.params.equipment_gain * self.params.floor_area
        Q_internal = Q_occupancy + Q_lighting + Q_equipment
        
        # HVAC heat input (W)
        Q_hvac = control_input * self.params.hvac_efficiency
        
        # Total heat balance
        Q_total = Q_hvac + Q_solar + Q_internal + Q_conduction
        
        # Temperature rate of change
        dT_dt = Q_total / self.params.thermal_mass
        
        return np.array([dT_dt])
    
    def simulate(self, time_span: Tuple[float, float], initial_temp: float,
                control_sequence: np.ndarray, disturbance_sequence: List[Dict],
                time_points: np.ndarray) -> Dict:
        """Simulate room thermal response"""
        
        def ode_func(t, state):
            # Interpolate control input and disturbances
            idx = min(int(t), len(control_sequence) - 1)
            control = control_sequence[idx]
            disturbances = disturbance_sequence[idx] if idx < len(disturbance_sequence) else {}
            
            return self.thermal_dynamics(t, state, control, disturbances)
        
        # Solve ODE
        sol = solve_ivp(ode_func, time_span, [initial_temp], 
                       t_eval=time_points, method='RK45', rtol=1e-6)
        
        return {
            'time': sol.t,
            'temperature': sol.y[0],
            'success': sol.success
        }

class PIDController:
    """
    PID controller implementation based on:
    - Astrom & Hagglund "PID Controllers: Theory, Design and Tuning" (1995)
    - Franklin et al. "Feedback Control of Dynamic Systems" (2019)
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-5000, 5000)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0
        
    def compute(self, setpoint: float, measurement: float, 
                current_time: float) -> float:
        """Compute PID control output"""
        
        # Error calculation
        error = setpoint - measurement
        
        # Time step
        dt = current_time - self.previous_time if self.previous_time > 0 else 0.1
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        derivative_term = self.kd * derivative
        
        # PID output
        output = proportional + integral_term + derivative_term
        
        # Apply output limits
        output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Anti-windup: adjust integral if output is saturated
        if output == self.output_limits[1] or output == self.output_limits[0]:
            self.integral -= error * dt
        
        # Update state
        self.previous_error = error
        self.previous_time = current_time
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0

class ModelPredictiveController:
    """
    Model Predictive Control implementation based on:
    - Camacho & Bordons "Model Predictive Control" (2007)
    - Oldewurtel et al. "Use of model predictive control and weather 
      forecasts for energy efficient building climate control" (2012)
    """
    
    def __init__(self, model: ThermalModel, prediction_horizon: int = 10,
                 control_horizon: int = 5):
        self.model = model
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        
    def predict_temperature(self, initial_temp: float, control_sequence: np.ndarray,
                          disturbance_forecast: List[Dict]) -> np.ndarray:
        """Predict temperature evolution over prediction horizon"""
        
        time_points = np.arange(len(control_sequence))
        result = self.model.simulate(
            (0, len(control_sequence) - 1),
            initial_temp,
            control_sequence,
            disturbance_forecast,
            time_points
        )
        
        return result['temperature']
    
    def optimize_control(self, current_temp: float, setpoint_sequence: np.ndarray,
                        disturbance_forecast: List[Dict]) -> np.ndarray:
        """Optimize control sequence using MPC"""
        
        def objective(control_vars):
            # Extend control sequence
            control_sequence = np.zeros(self.prediction_horizon)
            control_sequence[:len(control_vars)] = control_vars
            
            # Predict temperatures
            predicted_temps = self.predict_temperature(
                current_temp, control_sequence, disturbance_forecast
            )
            
            # Cost function: tracking error + control effort
            tracking_cost = np.sum((predicted_temps - setpoint_sequence)**2)
            control_cost = 0.01 * np.sum(control_vars**2)  # Energy penalty
            
            return tracking_cost + control_cost
        
        # Optimization constraints
        bounds = [(-5000, 5000)] * self.control_horizon  # HVAC power limits
        
        # Initial guess
        x0 = np.zeros(self.control_horizon)
        
        # Solve optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds)
        
        return result.x if result.success else x0

class BuildingSimulation:
    """
    Complete building simulation environment
    
    Integrates thermal model with control systems and disturbances
    Based on EnergyPlus modeling principles and ASHRAE standards
    """
    
    def __init__(self, room_params: RoomParameters):
        self.room_params = room_params
        self.thermal_model = ThermalModel(room_params)
        
        # Controllers
        self.pid_controller = PIDController(kp=500, ki=50, kd=100)
        self.mpc_controller = ModelPredictiveController(self.thermal_model)
        
        # Simulation state
        self.current_time = 0.0
        self.simulation_data = []
        
    def generate_disturbances(self, time_hours: np.ndarray) -> List[Dict]:
        """Generate realistic disturbance profiles"""
        disturbances = []
        
        for t in time_hours:
            # Daily temperature variation (sinusoidal)
            ambient_temp = 25 + 5 * np.sin(2 * np.pi * t / 24 - np.pi/2)
            
            # Solar gain profile (peak at noon)
            hour_of_day = t % 24
            if 6 <= hour_of_day <= 18:
                solar_gain = 800 * np.sin(np.pi * (hour_of_day - 6) / 12)
            else:
                solar_gain = 0
            
            # Occupancy profile (office hours)
            if 8 <= hour_of_day <= 17:
                occupancy = 1.0 + 0.3 * np.random.normal()  # Some variation
            else:
                occupancy = 0.1  # Minimal occupancy
            
            disturbances.append({
                'ambient_temp': ambient_temp,
                'solar_gain': max(0, solar_gain),
                'occupancy': max(0, occupancy)
            })
        
        return disturbances
    
    def generate_setpoint_profile(self, time_hours: np.ndarray) -> np.ndarray:
        """Generate realistic temperature setpoint profile"""
        setpoints = []
        
        for t in time_hours:
            hour_of_day = t % 24
            
            # Office temperature schedule
            if 6 <= hour_of_day <= 18:
                setpoint = 22.0  # Comfort temperature during occupied hours
            else:
                setpoint = 18.0  # Energy saving during unoccupied hours
            
            setpoints.append(setpoint)
        
        return np.array(setpoints)
    
    def run_simulation(self, duration_hours: float = 48, dt_hours: float = 0.1,
                      control_type: str = 'pid') -> Dict:
        """Run complete building simulation"""
        
        # Time vector
        time_hours = np.arange(0, duration_hours, dt_hours)
        
        # Generate profiles
        disturbances = self.generate_disturbances(time_hours)
        setpoints = self.generate_setpoint_profile(time_hours)
        
        # Initialize
        current_temp = 20.0  # Initial room temperature
        control_sequence = []
        temperature_sequence = [current_temp]
        
        # Reset controllers
        self.pid_controller.reset()
        
        # Simulation loop
        for i, t in enumerate(time_hours[1:], 1):
            
            if control_type == 'pid':
                # PID control
                control_output = self.pid_controller.compute(
                    setpoints[i], current_temp, t
                )
                
            elif control_type == 'mpc':
                # MPC control
                horizon_end = min(i + self.mpc_controller.prediction_horizon, len(setpoints))
                setpoint_horizon = setpoints[i:horizon_end]
                disturbance_horizon = disturbances[i:horizon_end]
                
                if len(setpoint_horizon) < self.mpc_controller.prediction_horizon:
                    # Pad with last values
                    last_setpoint = setpoint_horizon[-1]
                    last_disturbance = disturbance_horizon[-1]
                    while len(setpoint_horizon) < self.mpc_controller.prediction_horizon:
                        setpoint_horizon = np.append(setpoint_horizon, last_setpoint)
                        disturbance_horizon.append(last_disturbance)
                
                optimal_controls = self.mpc_controller.optimize_control(
                    current_temp, setpoint_horizon, disturbance_horizon
                )
                control_output = optimal_controls[0]
                
            else:
                control_output = 0.0  # No control
            
            control_sequence.append(control_output)
            
            # Simulate one time step
            result = self.thermal_model.simulate(
                (0, dt_hours),
                current_temp,
                np.array([control_output]),
                [disturbances[i]],
                np.array([0, dt_hours])
            )
            
            current_temp = result['temperature'][-1]
            temperature_sequence.append(current_temp)
        
        return {
            'time': time_hours,
            'temperature': np.array(temperature_sequence),
            'setpoint': setpoints,
            'control': np.array([0] + control_sequence),
            'disturbances': disturbances,
            'control_type': control_type
        }
    
    def analyze_performance(self, simulation_results: Dict) -> Dict:
        """Analyze control system performance"""
        
        temp = simulation_results['temperature']
        setpoint = simulation_results['setpoint']
        control = simulation_results['control']
        
        # Performance metrics
        tracking_error = temp - setpoint
        mae = np.mean(np.abs(tracking_error))  # Mean Absolute Error
        rmse = np.sqrt(np.mean(tracking_error**2))  # Root Mean Square Error
        
        # Energy consumption
        energy_consumption = np.sum(np.abs(control)) * 0.1  # kWh (assuming dt=0.1h)
        
        # Comfort metrics (percentage of time within comfort band)
        comfort_band = 1.0  # ±1°C
        comfort_violations = np.sum(np.abs(tracking_error) > comfort_band)
        comfort_percentage = (1 - comfort_violations / len(tracking_error)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'energy_consumption': energy_consumption,
            'comfort_percentage': comfort_percentage,
            'max_error': np.max(np.abs(tracking_error)),
            'control_effort': np.sum(np.abs(np.diff(control)))
        }

def main():
    """Demonstrate room simulation and control systems"""
    
    print("Building Thermal Simulation and Control System")
    print("=" * 60)
    print("Based on established research in building automation")
    print("and control theory from academic literature.")
    print()
    
    # Create room parameters
    room_params = RoomParameters()
    
    # Initialize simulation
    building_sim = BuildingSimulation(room_params)
    
    # Run simulations with different control strategies
    control_strategies = ['pid', 'mpc', 'none']
    results = {}
    
    for strategy in control_strategies:
        print(f"Running simulation with {strategy.upper()} control...")
        results[strategy] = building_sim.run_simulation(
            duration_hours=48, 
            control_type=strategy
        )
    
    # Analyze performance
    print("\nPerformance Analysis:")
    print("-" * 40)
    
    for strategy in control_strategies:
        performance = building_sim.analyze_performance(results[strategy])
        print(f"\n{strategy.upper()} Controller:")
        print(f"  Mean Absolute Error: {performance['mae']:.2f}°C")
        print(f"  RMSE: {performance['rmse']:.2f}°C")
        print(f"  Energy Consumption: {performance['energy_consumption']:.1f} kWh")
        print(f"  Comfort Percentage: {performance['comfort_percentage']:.1f}%")
        print(f"  Max Error: {performance['max_error']:.2f}°C")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Temperature tracking
    for strategy in control_strategies:
        data = results[strategy]
        axes[0].plot(data['time'], data['temperature'], 
                    label=f'{strategy.upper()} Control', linewidth=2)
    
    axes[0].plot(results['pid']['time'], results['pid']['setpoint'], 
                'k--', label='Setpoint', linewidth=2)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Room Temperature Control Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Control effort
    for strategy in ['pid', 'mpc']:
        data = results[strategy]
        axes[1].plot(data['time'], data['control'], 
                    label=f'{strategy.upper()} Control', linewidth=2)
    
    axes[1].set_ylabel('HVAC Power (W)')
    axes[1].set_title('Control Effort')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Disturbances
    time = results['pid']['time']
    ambient_temps = [d['ambient_temp'] for d in results['pid']['disturbances']]
    solar_gains = [d['solar_gain'] for d in results['pid']['disturbances']]
    
    ax2_twin = axes[2].twinx()
    axes[2].plot(time, ambient_temps, 'r-', label='Ambient Temperature', linewidth=2)
    ax2_twin.plot(time, solar_gains, 'orange', label='Solar Gain', linewidth=2)
    
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_ylabel('Ambient Temperature (°C)', color='r')
    ax2_twin.set_ylabel('Solar Gain (W/m²)', color='orange')
    axes[2].set_title('Environmental Disturbances')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSimulation completed. Results show comparative performance")
    print("of different control strategies for building thermal management.")

if __name__ == "__main__":
    main()
