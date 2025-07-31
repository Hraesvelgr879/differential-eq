"""
Comprehensive library of ODE examples for testing and education
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Tuple

class ODEExamplesLibrary:
    """Library of differential equation examples with analytical solutions"""
    
    def __init__(self):
        self.examples = {}
        self.setup_examples()
    
    def setup_examples(self):
        """Setup comprehensive examples library"""
        
        # 1. FIRST-ORDER LINEAR ODEs
        self.examples['exponential_decay'] = {
            'equation': 'dy/dx = -k*y',
            'description': 'Exponential decay (radioactive decay, cooling)',
            'analytical': lambda x, y0=1, k=1: y0 * np.exp(-k * x),
            'parameters': {'y0': 1.0, 'k': 2.0},
            'domain': (0, 3),
            'initial_conditions': [1.0],
            'category': 'First-Order Linear'
        }
        
        self.examples['exponential_growth'] = {
            'equation': 'dy/dx = k*y',
            'description': 'Exponential growth (population, compound interest)',
            'analytical': lambda x, y0=1, k=1: y0 * np.exp(k * x),
            'parameters': {'y0': 0.5, 'k': 0.5},
            'domain': (0, 4),
            'initial_conditions': [0.5],
            'category': 'First-Order Linear'
        }
        
        self.examples['logistic_growth'] = {
            'equation': 'dy/dx = r*y*(1 - y/K)',
            'description': 'Logistic growth (limited population growth)',
            'analytical': lambda x, y0=0.5, r=0.5, K=10: K / (1 + ((K - y0)/y0) * np.exp(-r * x)),
            'parameters': {'y0': 0.5, 'r': 0.5, 'K': 10.0},
            'domain': (0, 15),
            'initial_conditions': [0.5],
            'category': 'First-Order Nonlinear'
        }
        
        # 2. SECOND-ORDER LINEAR ODEs
        self.examples['harmonic_oscillator'] = {
            'equation': 'd²y/dx² + ω²*y = 0',
            'description': 'Simple harmonic oscillator (spring-mass system)',
            'analytical': lambda x, y0=1, v0=0, omega=1: y0 * np.cos(omega * x) + (v0/omega) * np.sin(omega * x),
            'parameters': {'y0': 1.0, 'v0': 0.0, 'omega': 1.0},
            'domain': (0, 4*np.pi),
            'initial_conditions': [1.0, 0.0],
            'category': 'Second-Order Linear'
        }
        
        self.examples['damped_oscillator'] = {
            'equation': 'd²y/dx² + 2γ*dy/dx + ω²*y = 0',
            'description': 'Damped harmonic oscillator',
            'parameters': {'y0': 1.0, 'v0': 0.0, 'gamma': 0.5, 'omega': 2.0},
            'domain': (0, 10),
            'initial_conditions': [1.0, 0.0],
            'category': 'Second-Order Linear'
        }
        
        self.examples['driven_oscillator'] = {
            'equation': 'd²y/dx² + 2γ*dy/dx + ω²*y = F*cos(Ω*x)',
            'description': 'Driven damped oscillator',
            'parameters': {'gamma': 0.2, 'omega': 1.0, 'F': 1.0, 'Omega': 1.5},
            'domain': (0, 20),
            'initial_conditions': [0.0, 0.0],
            'category': 'Second-Order Linear'
        }
        
        # 3. SYSTEMS OF ODEs
        self.examples['predator_prey'] = {
            'equation': 'dx/dt = αx - βxy, dy/dt = δxy - γy',
            'description': 'Lotka-Volterra predator-prey model',
            'parameters': {'alpha': 1.0, 'beta': 0.1, 'gamma': 1.5, 'delta': 0.075},
            'domain': (0, 15),
            'initial_conditions': [10, 5],
            'category': 'System of ODEs'
        }
        
        self.examples['sir_model'] = {
            'equation': 'dS/dt = -βSI/N, dI/dt = βSI/N - γI, dR/dt = γI',
            'description': 'SIR epidemic model',
            'parameters': {'beta': 0.3, 'gamma': 0.1, 'N': 1000},
            'domain': (0, 100),
            'initial_conditions': [999, 1, 0],
            'category': 'System of ODEs'
        }
        
        # 4. NONLINEAR ODEs
        self.examples['van_der_pol'] = {
            'equation': 'd²y/dx² - μ(1-y²)dy/dx + y = 0',
            'description': 'Van der Pol oscillator (nonlinear dynamics)',
            'parameters': {'mu': 1.0},
            'domain': (0, 20),
            'initial_conditions': [2.0, 0.0],
            'category': 'Second-Order Nonlinear'
        }
        
        self.examples['pendulum'] = {
            'equation': 'd²θ/dt² + (g/L)*sin(θ) = 0',
            'description': 'Nonlinear pendulum',
            'parameters': {'g': 9.81, 'L': 1.0},
            'domain': (0, 10),
            'initial_conditions': [np.pi/4, 0.0],  # 45 degrees initial angle
            'category': 'Second-Order Nonlinear'
        }
        
        # 5. SPECIAL FUNCTIONS
        self.examples['bessel_equation'] = {
            'equation': 'x²d²y/dx² + x*dy/dx + (x² - ν²)y = 0',
            'description': 'Bessel equation (appears in cylindrical coordinates)',
            'parameters': {'nu': 0},
            'domain': (0.1, 10),
            'initial_conditions': [1.0, 0.0],
            'category': 'Special Functions'
        }
        
        # 6. CHEMICAL KINETICS
        self.examples['autocatalysis'] = {
            'equation': 'dx/dt = k*x*(a-x)',
            'description': 'Autocatalytic reaction',
            'parameters': {'k': 0.1, 'a': 10.0},
            'domain': (0, 50),
            'initial_conditions': [0.1],
            'category': 'Chemical Kinetics'
        }
    
    def get_example(self, name: str) -> Dict:
        """Get specific example by name"""
        return self.examples.get(name, None)
    
    def list_examples(self, category: str = None) -> List[str]:
        """List all examples, optionally filtered by category"""
        if category:
            return [name for name, ex in self.examples.items() 
                   if ex['category'] == category]
        return list(self.examples.keys())
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(set(ex['category'] for ex in self.examples.values()))
    
    def print_example_info(self, name: str):
        """Print detailed information about an example"""
        example = self.get_example(name)
        if not example:
            print(f"Example '{name}' not found")
            return
        
        print(f"\n📚 {name.replace('_', ' ').title()}")
        print("=" * 50)
        print(f"Equation: {example['equation']}")
        print(f"Category: {example['category']}")
        print(f"Description: {example['description']}")
        print(f"Domain: {example['domain']}")
        print(f"Initial Conditions: {example['initial_conditions']}")
        if 'parameters' in example:
            print(f"Parameters: {example['parameters']}")
        print("=" * 50)
    
    def demonstrate_example(self, name: str, method: str = 'RK45'):
        """Demonstrate solving and plotting an example"""
        example = self.get_example(name)
        if not example:
            print(f"Example '{name}' not found")
            return
        
        print(f"🧮 Demonstrating: {name.replace('_', ' ').title()}")
        self.print_example_info(name)
        
        # This would integrate with the main ODE solver
        print(f"\n💡 To solve this in the ODE Solver:")
        print(f"1. Enter equation: {example['equation']}")
        print(f"2. Set initial conditions: {example['initial_conditions']}")
        print(f"3. Set domain: {example['domain']}")
        print(f"4. Choose method: {method}")
        print(f"5. Click 'Solve & Plot'")

def main():
    """Demonstrate the examples library"""
    print("📚 ODE Examples Library")
    print("=" * 60)
    
    library = ODEExamplesLibrary()
    
    # Show categories
    categories = library.get_categories()
    print(f"Available categories: {', '.join(categories)}")
    
    # Show examples by category
    for category in categories:
        examples = library.list_examples(category)
        print(f"\n{category}:")
        for example in examples:
            print(f"  • {example.replace('_', ' ').title()}")
    
    # Demonstrate a few examples
    print(f"\n🎯 Example Demonstrations:")
    print("-" * 40)
    
    demo_examples = ['exponential_decay', 'harmonic_oscillator', 'predator_prey']
    
    for example_name in demo_examples:
        library.demonstrate_example(example_name)
        print()
    
    print("✨ Use these examples in the ODE Solver for accurate solutions!")

if __name__ == "__main__":
    main()
