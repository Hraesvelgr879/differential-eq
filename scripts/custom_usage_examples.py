"""
Examples showing how to input your own data and create beautiful graphs
"""

from beautiful_graphs import BeautifulGraphs
import numpy as np
import matplotlib.pyplot as plt

def example_with_your_data():
    """Show how to use the library with your own data"""
    bg = BeautifulGraphs()
    
    print("📊 Custom Data Examples")
    print("=" * 40)
    
    # Example 1: Temperature data over a week
    print("1. Weekly temperature data...")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    temperatures = [22, 25, 23, 27, 29, 31, 28]  # Your data here
    
    bg.bar_plot(days, temperatures, "Weekly Temperature", "Day", "Temperature (°C)")
    plt.show()
    
    # Example 2: Stock price simulation
    print("2. Stock price over time...")
    days = np.arange(1, 31)  # 30 days
    # Simulate some stock data (replace with your actual data)
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(30) * 2)
    
    bg.line_plot(days, price, "Stock Price Movement", "Day", "Price ($)")
    plt.show()
    
    # Example 3: Multiple datasets comparison
    print("3. Comparing multiple datasets...")
    x = np.linspace(0, 10, 50)
    # Replace these with your actual datasets
    dataset1 = x**2 + np.random.normal(0, 5, 50)
    dataset2 = 2*x**2 + np.random.normal(0, 8, 50)
    dataset3 = 0.5*x**2 + np.random.normal(0, 3, 50)
    
    bg.multi_line_plot(x, [dataset1, dataset2, dataset3], 
                      labels=['Dataset A', 'Dataset B', 'Dataset C'],
                      title="Comparison of Three Datasets", 
                      xlabel="X Values", ylabel="Y Values")
    plt.show()
    
    # Example 4: Scatter plot for correlation
    print("4. Correlation analysis...")
    # Generate correlated data (replace with your data)
    np.random.seed(123)
    x_data = np.random.randn(100)
    y_data = 2 * x_data + np.random.randn(100) * 0.5  # Correlated with noise
    
    bg.scatter_plot(x_data, y_data, "Correlation Analysis", 
                   "Variable X", "Variable Y")
    plt.show()
    
    # Example 5: Distribution analysis
    print("5. Data distribution...")
    # Sample data - replace with your measurements
    measurements = np.random.normal(75, 12, 500)  # Mean=75, std=12
    
    bg.histogram(measurements, bins=25, title="Measurement Distribution",
                xlabel="Measurement Value", ylabel="Frequency")
    plt.show()

def simple_usage_template():
    """Template showing the simplest way to use the library"""
    print("\n📝 Simple Usage Template")
    print("=" * 40)
    print("""
# Step 1: Import and create the graph maker
from beautiful_graphs import BeautifulGraphs
bg = BeautifulGraphs()

# Step 2: Prepare your data
x_data = [1, 2, 3, 4, 5]  # Your X values
y_data = [2, 4, 6, 8, 10]  # Your Y values

# Step 3: Create beautiful graph with one line!
bg.line_plot(x_data, y_data, "My Beautiful Graph", "X Label", "Y Label")
plt.show()

# That's it! 🎉
    """)

if __name__ == "__main__":
    example_with_your_data()
    simple_usage_template()
