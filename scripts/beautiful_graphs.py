import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import seaborn as sns
from typing import List, Tuple, Optional, Union
import pandas as pd

class BeautifulGraphs:
    """
    A matplotlib extension for creating beautiful graphs with minimal code.
    Pre-configured with aesthetic styling and easy-to-use methods.
    """
    
    def __init__(self, style='seaborn-v0_8', figsize=(10, 6), dpi=100):
        """Initialize with beautiful default settings."""
        # Set the style
        plt.style.use('default')  # Reset to default first
        sns.set_palette("husl")
        
        # Configure matplotlib for better aesthetics
        plt.rcParams.update({
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        self.current_color_idx = 0
    
    def _get_next_color(self):
        """Get the next color in the palette."""
        color = self.colors[self.current_color_idx % len(self.colors)]
        self.current_color_idx += 1
        return color
    
    def line_plot(self, x_data, y_data, title="Beautiful Line Plot", 
                  xlabel="X-axis", ylabel="Y-axis", label=None, 
                  color=None, linewidth=2.5, alpha=0.8):
        """Create a beautiful line plot."""
        if color is None:
            color = self._get_next_color()
        
        plt.figure(figsize=plt.rcParams['figure.figsize'])
        plt.plot(x_data, y_data, color=color, linewidth=linewidth, 
                alpha=alpha, label=label, marker='o', markersize=4, 
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
        
        plt.title(title, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        
        if label:
            plt.legend(frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        return plt.gcf()
    
    def multi_line_plot(self, x_data, y_data_list, labels=None, 
                       title="Beautiful Multi-Line Plot", xlabel="X-axis", ylabel="Y-axis"):
        """Create a beautiful multi-line plot."""
        plt.figure(figsize=plt.rcParams['figure.figsize'])
        
        if labels is None:
            labels = [f"Series {i+1}" for i in range(len(y_data_list))]
        
        for i, y_data in enumerate(y_data_list):
            color = self._get_next_color()
            plt.plot(x_data, y_data, color=color, linewidth=2.5, 
                    alpha=0.8, label=labels[i], marker='o', markersize=4,
                    markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
        
        plt.title(title, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        return plt.gcf()
    
    def scatter_plot(self, x_data, y_data, title="Beautiful Scatter Plot",
                    xlabel="X-axis", ylabel="Y-axis", size=60, alpha=0.7):
        """Create a beautiful scatter plot."""
        plt.figure(figsize=plt.rcParams['figure.figsize'])
        color = self._get_next_color()
        
        plt.scatter(x_data, y_data, c=color, s=size, alpha=alpha, 
                   edgecolors='white', linewidth=1.5)
        
        plt.title(title, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        plt.tight_layout()
        return plt.gcf()
    
    def bar_plot(self, categories, values, title="Beautiful Bar Plot",
                xlabel="Categories", ylabel="Values", horizontal=False):
        """Create a beautiful bar plot."""
        plt.figure(figsize=plt.rcParams['figure.figsize'])
        colors = [self._get_next_color() for _ in range(len(categories))]
        
        if horizontal:
            bars = plt.barh(categories, values, color=colors, alpha=0.8, 
                           edgecolor='white', linewidth=1.5)
            plt.xlabel(ylabel, fontweight='bold')
            plt.ylabel(xlabel, fontweight='bold')
        else:
            bars = plt.bar(categories, values, color=colors, alpha=0.8, 
                          edgecolor='white', linewidth=1.5)
            plt.xlabel(xlabel, fontweight='bold')
            plt.ylabel(ylabel, fontweight='bold')
        
        plt.title(title, fontweight='bold', pad=20)
        plt.tight_layout()
        return plt.gcf()
    
    def histogram(self, data, bins=30, title="Beautiful Histogram",
                 xlabel="Values", ylabel="Frequency", alpha=0.7):
        """Create a beautiful histogram."""
        plt.figure(figsize=plt.rcParams['figure.figsize'])
        color = self._get_next_color()
        
        plt.hist(data, bins=bins, color=color, alpha=alpha, 
                edgecolor='white', linewidth=1.5)
        
        plt.title(title, fontweight='bold', pad=20)
        plt.xlabel(xlabel, fontweight='bold')
        plt.ylabel(ylabel, fontweight='bold')
        plt.tight_layout()
        return plt.gcf()
    
    def heatmap(self, data, title="Beautiful Heatmap", cmap='viridis'):
        """Create a beautiful heatmap."""
        plt.figure(figsize=plt.rcParams['figure.figsize'])
        
        if isinstance(data, pd.DataFrame):
            sns.heatmap(data, annot=True, cmap=cmap, cbar=True, 
                       square=True, linewidths=0.5)
        else:
            sns.heatmap(data, annot=True, cmap=cmap, cbar=True, 
                       square=True, linewidths=0.5)
        
        plt.title(title, fontweight='bold', pad=20)
        plt.tight_layout()
        return plt.gcf()
    
    def reset_colors(self):
        """Reset color index to start from the beginning."""
        self.current_color_idx = 0

# ODE Solver Example Functions
def solve_ode_system(func, y0, t_span, args=()):
    """
    Solve an ordinary differential equation system.
    
    Args:
        func: Function defining the ODE system dy/dt = func(y, t, *args)
        y0: Initial conditions
        t_span: Time points to solve for
        args: Additional arguments for the function
    
    Returns:
        t_span, solution
    """
    solution = odeint(func, y0, t_span, args=args)
    return t_span, solution

# Example ODE Systems
def harmonic_oscillator(y, t, omega=1.0, damping=0.1):
    """Damped harmonic oscillator: d²x/dt² + 2γ(dx/dt) + ω²x = 0"""
    x, v = y
    dxdt = v
    dvdt = -2 * damping * v - omega**2 * x
    return [dxdt, dvdt]

def predator_prey(y, t, alpha=1.0, beta=0.1, gamma=1.5, delta=0.075):
    """Lotka-Volterra predator-prey model"""
    prey, predator = y
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]

def lorenz_system(y, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Lorenz chaotic system"""
    x, y_coord, z = y
    dxdt = sigma * (y_coord - x)
    dydt = x * (rho - z) - y_coord
    dzdt = x * y_coord - beta * z
    return [dxdt, dydt, dzdt]

# Demo function
def demo_beautiful_graphs():
    """Demonstrate the BeautifulGraphs class with various examples."""
    bg = BeautifulGraphs()
    
    print("🎨 Beautiful Graphs Demo")
    print("=" * 50)
    
    # 1. Simple line plot
    print("1. Creating a simple sine wave...")
    x = np.linspace(0, 4*np.pi, 100)
    y = np.sin(x) * np.exp(-x/10)
    bg.line_plot(x, y, "Damped Sine Wave", "Time", "Amplitude")
    plt.show()
    
    # 2. Multi-line plot with trigonometric functions
    print("2. Creating multi-line trigonometric plot...")
    bg.reset_colors()
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(2*x)
    bg.multi_line_plot(x, [y1, y2, y3], 
                      labels=['sin(x)', 'cos(x)', 'sin(2x)'],
                      title="Trigonometric Functions", 
                      xlabel="Angle (radians)", ylabel="Value")
    plt.show()
    
    # 3. ODE Example: Harmonic Oscillator
    print("3. Solving damped harmonic oscillator ODE...")
    bg.reset_colors()
    t = np.linspace(0, 10, 1000)
    y0 = [1.0, 0.0]  # Initial position and velocity
    t_sol, sol = solve_ode_system(harmonic_oscillator, y0, t, args=(2.0, 0.2))
    
    bg.multi_line_plot(t_sol, [sol[:, 0], sol[:, 1]], 
                      labels=['Position', 'Velocity'],
                      title="Damped Harmonic Oscillator", 
                      xlabel="Time", ylabel="Value")
    plt.show()
    
    # 4. ODE Example: Predator-Prey System
    print("4. Solving predator-prey system...")
    bg.reset_colors()
    t = np.linspace(0, 15, 1000)
    y0 = [10, 5]  # Initial prey and predator populations
    t_sol, sol = solve_ode_system(predator_prey, y0, t)
    
    bg.multi_line_plot(t_sol, [sol[:, 0], sol[:, 1]], 
                      labels=['Prey', 'Predator'],
                      title="Predator-Prey Dynamics", 
                      xlabel="Time", ylabel="Population")
    plt.show()
    
    # 5. Phase portrait of predator-prey
    print("5. Creating phase portrait...")
    bg.reset_colors()
    bg.scatter_plot(sol[:, 0], sol[:, 1], 
                   title="Predator-Prey Phase Portrait",
                   xlabel="Prey Population", ylabel="Predator Population",
                   size=20, alpha=0.6)
    plt.show()
    
    # 6. Bar plot example
    print("6. Creating sample bar plot...")
    bg.reset_colors()
    categories = ['Method A', 'Method B', 'Method C', 'Method D']
    values = [23, 45, 56, 78]
    bg.bar_plot(categories, values, "Performance Comparison", 
               "Methods", "Score")
    plt.show()
    
    # 7. Histogram example
    print("7. Creating histogram of random data...")
    bg.reset_colors()
    data = np.random.normal(100, 15, 1000)
    bg.histogram(data, bins=30, title="Normal Distribution Sample",
                xlabel="Value", ylabel="Frequency")
    plt.show()
    
    print("\n✨ Demo completed! All graphs should display beautifully.")
    print("You can now use BeautifulGraphs for your own data visualization needs!")

if __name__ == "__main__":
    demo_beautiful_graphs()
