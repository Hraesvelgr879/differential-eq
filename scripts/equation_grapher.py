import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import re
from typing import Optional

class EquationGrapher:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 Beautiful Equation Grapher")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables
        self.x_min = tk.DoubleVar(value=-10)
        self.x_max = tk.DoubleVar(value=10)
        self.equation_var = tk.StringVar(value="sin(x)")
        
        # Colors for beautiful graphs
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        self.current_color = 0
        
        self.setup_ui()
        self.setup_matplotlib()
        self.plot_equation()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎨 Beautiful Equation Grapher", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Equation Input", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Equation input
        ttk.Label(input_frame, text="y = ").grid(row=0, column=0, sticky=tk.W)
        self.equation_entry = ttk.Entry(input_frame, textvariable=self.equation_var, 
                                       font=('Consolas', 12), width=40)
        self.equation_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 10))
        self.equation_entry.bind('<Return>', lambda e: self.plot_equation())
        self.equation_entry.bind('<KeyRelease>', self.on_equation_change)
        
        plot_button = ttk.Button(input_frame, text="📊 Plot", command=self.plot_equation)
        plot_button.grid(row=0, column=2, padx=(5, 0))
        
        # Range controls
        range_frame = ttk.Frame(input_frame)
        range_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(range_frame, text="X Range:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(range_frame, text="from").grid(row=0, column=1, padx=(10, 5))
        
        x_min_entry = ttk.Entry(range_frame, textvariable=self.x_min, width=8)
        x_min_entry.grid(row=0, column=2, padx=(0, 5))
        x_min_entry.bind('<Return>', lambda e: self.plot_equation())
        
        ttk.Label(range_frame, text="to").grid(row=0, column=3, padx=(5, 5))
        
        x_max_entry = ttk.Entry(range_frame, textvariable=self.x_max, width=8)
        x_max_entry.grid(row=0, column=4, padx=(0, 10))
        x_max_entry.bind('<Return>', lambda e: self.plot_equation())
        
        update_range_button = ttk.Button(range_frame, text="Update Range", 
                                        command=self.plot_equation)
        update_range_button.grid(row=0, column=5, padx=(5, 0))
        
        # Examples frame
        examples_frame = ttk.LabelFrame(main_frame, text="Quick Examples", padding="5")
        examples_frame.grid(row=1, column=3, sticky=(tk.W, tk.E, tk.N), padx=(10, 0))
        
        examples = [
            ("sin(x)", "sin(x)"),
            ("x²", "x**2"),
            ("e^x", "exp(x)"),
            ("ln(x)", "log(x)"),
            ("√x", "sqrt(x)"),
            ("1/x", "1/x"),
            ("sin(x)/x", "sin(x)/x"),
            ("x³-3x²+2x", "x**3-3*x**2+2*x"),
        ]
        
        for i, (display, equation) in enumerate(examples):
            btn = ttk.Button(examples_frame, text=display, width=12,
                           command=lambda eq=equation: self.set_equation(eq))
            btn.grid(row=i//2, column=i%2, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # Graph frame
        self.graph_frame = ttk.Frame(main_frame)
        self.graph_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), 
                             pady=(10, 0))
        
    def setup_matplotlib(self):
        """Setup matplotlib figure and canvas"""
        # Create figure with beautiful styling
        self.fig = Figure(figsize=(12, 6), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        
        # Style the plot
        self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_linewidth(0.8)
        self.ax.spines['bottom'].set_linewidth(0.8)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(self.graph_frame)
        toolbar_frame.pack(fill=tk.X)
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
    def safe_eval(self, expression: str, x_values: np.ndarray) -> Optional[np.ndarray]:
        """Safely evaluate mathematical expression"""
        # Replace common mathematical notation
        expression = expression.replace('^', '**')
        expression = expression.replace('ln', 'log')
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)  # 2x -> 2*x
        expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)  # x2 -> x*2
        expression = re.sub(r'\)(\w)', r')*\1', expression)  # )x -> )*x
        expression = re.sub(r'(\w)\(', r'\1*(', expression)  # x( -> x*(
        
        # Safe namespace for evaluation
        safe_dict = {
            'x': x_values,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'log10': np.log10,
            'sqrt': np.sqrt, 'abs': np.abs,
            'pi': np.pi, 'e': np.e,
            'floor': np.floor, 'ceil': np.ceil,
            'deg': np.degrees, 'rad': np.radians,
            '__builtins__': {}
        }
        
        try:
            # Handle division by zero and other issues
            with np.errstate(divide='ignore', invalid='ignore'):
                result = eval(expression, safe_dict)
                return np.array(result, dtype=float)
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None
    
    def plot_equation(self):
        """Plot the current equation"""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Get equation and range
            equation = self.equation_var.get().strip()
            if not equation:
                equation = "0"
            
            x_min = self.x_min.get()
            x_max = self.x_max.get()
            
            if x_min >= x_max:
                messagebox.showerror("Error", "X minimum must be less than X maximum")
                return
            
            # Generate x values
            x = np.linspace(x_min, x_max, 1000)
            
            # Evaluate equation
            y = self.safe_eval(equation, x)
            
            if y is None:
                self.ax.text(0.5, 0.5, f'Error: Cannot evaluate equation\ny = {equation}', 
                           transform=self.ax.transAxes, ha='center', va='center',
                           fontsize=14, color='red', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(-1, 1)
            else:
                # Handle infinite values
                y = np.where(np.isfinite(y), y, np.nan)
                
                # Plot with beautiful styling
                color = self.colors[self.current_color % len(self.colors)]
                self.current_color += 1
                
                self.ax.plot(x, y, color=color, linewidth=2.5, alpha=0.8, 
                           label=f'y = {equation}')
                
                # Add zero lines
                self.ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
                self.ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)
                
                # Set labels and title
                self.ax.set_xlabel('x', fontsize=12, fontweight='bold')
                self.ax.set_ylabel('y', fontsize=12, fontweight='bold')
                self.ax.set_title(f'y = {equation}', fontsize=14, fontweight='bold', pad=20)
                
                # Set reasonable y limits
                finite_y = y[np.isfinite(y)]
                if len(finite_y) > 0:
                    y_range = np.max(finite_y) - np.min(finite_y)
                    if y_range > 0:
                        margin = y_range * 0.1
                        self.ax.set_ylim(np.min(finite_y) - margin, np.max(finite_y) + margin)
                
                # Add legend
                self.ax.legend(frameon=True, fancybox=True, shadow=True)
            
            # Style the plot
            self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_linewidth(0.8)
            self.ax.spines['bottom'].set_linewidth(0.8)
            
            # Update canvas
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while plotting:\n{str(e)}")
    
    def set_equation(self, equation: str):
        """Set equation from example button"""
        self.equation_var.set(equation)
        self.plot_equation()
    
    def on_equation_change(self, event):
        """Handle equation change with delay for real-time plotting"""
        # Cancel previous scheduled plot
        if hasattr(self, '_plot_job'):
            self.root.after_cancel(self._plot_job)
        
        # Schedule new plot after 500ms delay
        self._plot_job = self.root.after(500, self.plot_equation)

def main():
    """Main function to run the equation grapher"""
    print("🎨 Starting Beautiful Equation Grapher...")
    print("=" * 50)
    print("Features:")
    print("• Type any mathematical equation in the input field")
    print("• Press Enter or click Plot to graph it")
    print("• Use quick example buttons for common functions")
    print("• Adjust X range as needed")
    print("• Real-time plotting as you type!")
    print("=" * 50)
    
    root = tk.Tk()
    app = EquationGrapher(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    print("✨ Equation Grapher is ready!")
    print("Try typing equations like: sin(x), x**2, exp(x), etc.")
    
    root.mainloop()

if __name__ == "__main__":
    main()
