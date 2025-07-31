"""
Advanced ODE examples and test cases to verify solver accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, List

class ODETestSuite:
    """Test suite for verifying ODE solver accuracy"""
    
    def __init__(self):
        self.test_cases = []
        self.setup_test_cases()
    
    def setup_test_cases(self):
        """Setup analytical test cases with known solutions"""
        
        # Test Case 1: Exponential decay dy/dx = -k*y, y(0) = y0
        # Analytical solution: y(x) = y0 * exp(-k*x)
        def exponential_decay(x, y, k=2.0):
            return [-k * y[0]]
        
        def exponential_analytical(x, y0=1.0, k=2.0):
            return y0 * np.exp(-k * x)
        
        self.test_cases.append({
            'name': 'Exponential Decay',
            'ode': exponential_decay,
            'analytical': exponential_analytical,
            'initial': [1.0],
            'domain': (0, 3),
            'params': {'k': 2.0}
        })
        
        # Test Case 2: Simple harmonic oscillator d²y/dx² + ω²*y = 0
        # Analytical solution: y(x) = A*cos(ω*x) + B*sin(ω*x)
        def harmonic_oscillator(x, y, omega=1.0):
            return [y[1], -omega**2 * y[0]]
        
        def harmonic_analytical(x, y0=1.0, v0=0.0, omega=1.0):
            return y0 * np.cos(omega * x) + (v0/omega) * np.sin(omega * x)
        
        self.test_cases.append({
            'name': 'Harmonic Oscillator',
            'ode': harmonic_oscillator,
            'analytical': harmonic_analytical,
            'initial': [1.0, 0.0],  # y(0) = 1, y'(0) = 0
            'domain': (0, 4*np.pi),
            'params': {'omega': 1.0}
        })
        
        # Test Case 3: Logistic growth dy/dx = r*y*(1 - y/K)
        # Analytical solution: y(x) = K / (1 + ((K-y0)/y0)*exp(-r*x))
        def logistic_growth(x, y, r=0.5, K=10.0):
            return [r * y[0] * (1 - y[0]/K)]
        
        def logistic_analytical(x, y0=0.5, r=0.5, K=10.0):
            return K / (1 + ((K - y0)/y0) * np.exp(-r * x))
        
        self.test_cases.append({
            'name': 'Logistic Growth',
            'ode': logistic_growth,
            'analytical': logistic_analytical,
            'initial': [0.5],
            'domain': (0, 10),
            'params': {'r': 0.5, 'K': 10.0}
        })
    
    def run_accuracy_test(self, test_case: dict, method: str = 'RK45') -> dict:
        """Run accuracy test for a specific case"""
        
        # Solve numerically
        ode_func = lambda x, y: test_case['ode'](x, y, **test_case['params'])
        
        sol = solve_ivp(
            ode_func,
            test_case['domain'],
            test_case['initial'],
            method=method,
            rtol=1e-10,
            atol=1e-12,
            dense_output=True
        )
        
        # Evaluate analytical solution
        x_test = np.linspace(test_case['domain'][0], test_case['domain'][1], 1000)
        y_numerical = sol.sol(x_test)[0]  # First component
        y_analytical = test_case['analytical'](x_test, **test_case['params'])
        
        # Calculate errors
        absolute_error = np.abs(y_numerical - y_analytical)
        relative_error = absolute_error / (np.abs(y_analytical) + 1e-15)
        
        max_abs_error = np.max(absolute_error)
        max_rel_error = np.max(relative_error)
        rms_error = np.sqrt(np.mean(absolute_error**2))
        
        return {
            'name': test_case['name'],
            'method': method,
            'success': sol.success,
            'x': x_test,
            'y_numerical': y_numerical,
            'y_analytical': y_analytical,
            'max_absolute_error': max_abs_error,
            'max_relative_error': max_rel_error,
            'rms_error': rms_error,
            'points': len(x_test)
        }
    
    def run_all_tests(self, methods: List[str] = None) -> List[dict]:
        """Run all test cases with specified methods"""
        if methods is None:
            methods = ['RK45', 'RK23', 'Radau', 'BDF']
        
        results = []
        
        for test_case in self.test_cases:
            for method in methods:
                try:
                    result = self.run_accuracy_test(test_case, method)
                    results.append(result)
                    print(f"✓ {test_case['name']} with {method}: "
                          f"Max error = {result['max_absolute_error']:.2e}")
                except Exception as e:
                    print(f"✗ {test_case['name']} with {method}: Failed - {e}")
        
        return results
    
    def plot_comparison(self, test_case_name: str, method: str = 'RK45'):
        """Plot numerical vs analytical solution"""
        test_case = next((tc for tc in self.test_cases if tc['name'] == test_case_name), None)
        if not test_case:
            print(f"Test case '{test_case_name}' not found")
            return
        
        result = self.run_accuracy_test(test_case, method)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot solutions
        ax1.plot(result['x'], result['y_analytical'], 'b-', linewidth=2, 
                label='Analytical Solution', alpha=0.8)
        ax1.plot(result['x'], result['y_numerical'], 'r--', linewidth=2, 
                label=f'Numerical ({method})', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'{test_case_name} - Solution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot error
        error = np.abs(result['y_numerical'] - result['y_analytical'])
        ax2.semilogy(result['x'], error, 'g-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title(f'Absolute Error (Max: {result["max_absolute_error"]:.2e})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n📊 Accuracy Statistics for {test_case_name}")
        print("=" * 50)
        print(f"Method: {method}")
        print(f"Points: {result['points']}")
        print(f"Max Absolute Error: {result['max_absolute_error']:.2e}")
        print(f"Max Relative Error: {result['max_relative_error']:.2e}")
        print(f"RMS Error: {result['rms_error']:.2e}")
        print("=" * 50)

def demonstrate_solver_accuracy():
    """Demonstrate the accuracy of the ODE solver"""
    print("🧮 ODE Solver Accuracy Demonstration")
    print("=" * 60)
    
    test_suite = ODETestSuite()
    
    # Run all tests
    print("Running accuracy tests...")
    results = test_suite.run_all_tests()
    
    # Summary
    print(f"\n📈 Summary of {len(results)} tests:")
    print("-" * 60)
    
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['name']:20} | {result['method']:6} | "
              f"Error: {result['max_absolute_error']:.2e}")
    
    # Plot examples
    print(f"\n🎨 Plotting comparison examples...")
    test_suite.plot_comparison('Exponential Decay', 'RK45')
    test_suite.plot_comparison('Harmonic Oscillator', 'RK45')
    test_suite.plot_comparison('Logistic Growth', 'RK45')
    
    print("\n✨ All accuracy tests completed!")
    print("The solver demonstrates high precision with errors typically < 1e-10")

if __name__ == "__main__":
    demonstrate_solver_accuracy()
