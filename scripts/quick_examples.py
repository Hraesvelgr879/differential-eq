"""
Quick examples showing how to use BeautifulGraphs for common tasks
"""

from beautiful_graphs import BeautifulGraphs, solve_ode_system, lorenz_system
import numpy as np
import matplotlib.pyplot as plt

def quick_line_plot_example():
    """Quick example: Plot any mathematical function"""
    bg = BeautifulGraphs()
    
    # Example: Plot a custom function
    x = np.linspace(-5, 5, 200)
    y = x**3 - 3*x**2 + 2*x + 1
    
    bg.line_plot(x, y, "Cubic Function", "x", "f(x) = x³ - 3x² + 2x + 1")
    plt.show()

def quick_data_visualization():
    """Quick example: Visualize your own data"""
    bg = BeautifulGraphs()
    
    # Example: Sales data over months
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    sales = [1200, 1350, 1100, 1600, 1800, 1950]
    
    bg.bar_plot(months, sales, "Monthly Sales Report", "Month", "Sales ($)")
    plt.show()

def quick_ode_example():
    """Quick example: Solve and plot any ODE"""
    bg = BeautifulGraphs()
    
    # Example: Simple exponential decay
    def exponential_decay(y, t, k=0.5):
        return -k * y
    
    t = np.linspace(0, 10, 100)
    y0 = [100]  # Initial value
    t_sol, sol = solve_ode_system(exponential_decay, y0, t, args=(0.3,))
    
    bg.line_plot(t_sol, sol[:, 0], "Exponential Decay", "Time", "Value")
    plt.show()

def lorenz_attractor_3d():
    """Beautiful 3D visualization of Lorenz attractor"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Solve Lorenz system
    t = np.linspace(0, 30, 10000)
    y0 = [1.0, 1.0, 1.0]
    t_sol, sol = solve_ode_system(lorenz_system, y0, t)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory with color gradient
    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], 
           color='#FF6B6B', alpha=0.8, linewidth=0.8)
    
    ax.set_xlabel('X', fontweight='bold')
    ax.set_ylabel('Y', fontweight='bold')
    ax.set_zlabel('Z', fontweight='bold')
    ax.set_title('Lorenz Attractor - Chaotic System', fontweight='bold', pad=20)
    
    # Remove grid for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🚀 Quick Examples")
    print("=" * 30)
    
    print("1. Line plot example...")
    quick_line_plot_example()
    
    print("2. Data visualization example...")
    quick_data_visualization()
    
    print("3. ODE solving example...")
    quick_ode_example()
    
    print("4. 3D Lorenz attractor...")
    lorenz_attractor_3d()
    
    print("\n✅ All examples completed!")
