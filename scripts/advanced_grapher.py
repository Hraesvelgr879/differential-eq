import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import re
from typing import Optional, List, Tuple

class AdvancedEquationGrapher:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Advanced Equation Grapher")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.equations = []  # List of (equation, color, visible) tuples
        self.x_min = tk.DoubleVar(value=-10)
        self.x_max = tk.DoubleVar(value=10)
        self.current_equation = tk.StringVar(value="sin(x)")
        
        # Colors
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#FF8C94', '#A8E6CF']
        self.color_index = 0
        
        self.setup_ui()
        self.setup_matplotlib()
        
    def setup_ui(self):
        """Setup advanced user interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main graphing tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="📊 Grapher")
        
        # Multiple equations tab
        self.multi_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.multi_frame, text="📈 Multiple Equations")
        
        self.setup_main_tab()
        self.setup_multi_tab()
        
    def setup_main_tab(self):
        """Setup main graphing tab"""
        # Configure grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Input frame
        input_frame = ttk.LabelFrame(self.main_frame, text="Equation Input", padding="10")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Equation input with better styling
        ttk.Label(input_frame, text="y = ", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W)
        
        self.equation_entry = ttk.Entry(input_frame, textvariable=self.current_equation, 
                                       font=('Consolas', 14), width=50)
        self.equation_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 10))
        self.equation_entry.bind('<Return>', lambda e: self.plot_single_equation())
        self.equation_entry.bind('<KeyRelease>', self.on_equation_change)
        
        # Buttons frame
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=0, column=2, padx=(5, 0))
        
        ttk.Button(buttons_frame, text="📊 Plot", 
                  command=self.plot_single_equation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="➕ Add to Multi", 
                  command=self.add_to_multi).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="🗑️ Clear", 
                  command=self.clear_plot).pack(side=tk.LEFT)
        
        # Range and options
        options_frame = ttk.Frame(input_frame)
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # X Range
        ttk.Label(options_frame, text="X Range:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(options_frame, textvariable=self.x_min, width=8).grid(row=0, column=1, padx=(10, 5))
        ttk.Label(options_frame, text="to").grid(row=0, column=2, padx=(0, 5))
        ttk.Entry(options_frame, textvariable=self.x_max, width=8).grid(row=0, column=3, padx=(0, 10))
        
        # Quick examples with categories
        self.setup_examples(input_frame)
        
        # Graph frame
        self.graph_frame = ttk.Frame(self.main_frame)
        self.graph_frame.pack(fill=tk.BOTH, expand=True)
        
    def setup_multi_tab(self):
        """Setup multiple equations tab"""
        # Left panel for equation list
        left_panel = ttk.Frame(self.multi_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Equation list
        list_frame = ttk.LabelFrame(left_panel, text="Equations", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.equation_listbox = tk.Listbox(list_container, font=('Consolas', 10), height=15)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.equation_listbox.yview)
        self.equation_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.equation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Buttons for list management
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Remove Selected", 
                  command=self.remove_equation).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_all_equations).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="📊 Plot All", 
                  command=self.plot_all_equations).pack(fill=tk.X)
        
        # Right panel for graph
        self.multi_graph_frame = ttk.Frame(self.multi_frame)
        self.multi_graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def setup_examples(self, parent):
        """Setup example equations with categories"""
        examples_frame = ttk.LabelFrame(parent, text="Quick Examples", padding="5")
        examples_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        categories = {
            "Basic": [
                ("Linear", "2*x + 1"),
                ("Quadratic", "x**2"),
                ("Cubic", "x**3 - 3*x"),
                ("Square Root", "sqrt(x)"),
            ],
            "Trigonometric": [
                ("Sine", "sin(x)"),
                ("Cosine", "cos(x)"),
                ("Tangent", "tan(x)"),
                ("Sine Wave", "2*sin(3*x)"),
            ],
            "Exponential": [
                ("e^x", "exp(x)"),
                ("Natural Log", "log(x)"),
                ("Decay", "exp(-x)"),
                ("Growth", "2**x"),
            ],
            "Advanced": [
                ("Sinc", "sin(x)/x"),
                ("Gaussian", "exp(-x**2)"),
                ("Damped Sine", "exp(-x/5)*sin(x)"),
                ("Rational", "1/(x**2 + 1)"),
            ]
        }
        
        for i, (category, equations) in enumerate(categories.items()):
            category_frame = ttk.LabelFrame(examples_frame, text=category, padding="3")
            category_frame.grid(row=i//2, column=i%2, sticky=(tk.W, tk.E), padx=5, pady=2)
            
            for j, (name, eq) in enumerate(equations):
                btn = ttk.Button(category_frame, text=name, width=12,
                               command=lambda equation=eq: self.set_equation(equation))
                btn.grid(row=j//2, column=j%2, padx=2, pady=1, sticky=(tk.W, tk.E))
    
    def setup_matplotlib(self):
        """Setup matplotlib figures for both tabs"""
        # Main tab figure
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.setup_plot_style(self.ax)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Multi tab figure
        self.multi_fig = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self.multi_ax = self.multi_fig.add_subplot(111)
        self.setup_plot_style(self.multi_ax)
        
        self.multi_canvas = FigureCanvasTkAgg(self.multi_fig, master=self.multi_graph_frame)
        self.multi_canvas.draw()
        self.multi_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbars
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        
        toolbar1 = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        toolbar1.update()
        
        toolbar2 = NavigationToolbar2Tk(self.multi_canvas, self.multi_graph_frame)
        toolbar2.update()
    
    def setup_plot_style(self, ax):
        """Setup beautiful plot styling"""
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
    
    def safe_eval(self, expression: str, x_values: np.ndarray) -> Optional[np.ndarray]:
        """Safely evaluate mathematical expression"""
        # Enhanced expression preprocessing
        expression = expression.replace('^', '**')
        expression = expression.replace('ln', 'log')
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        expression = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', expression)
        expression = re.sub(r'\)(\w)', r')*\1', expression)
        expression = re.sub(r'(\w)\(', r'\1*(', expression)
        
        # Extended safe namespace
        safe_dict = {
            'x': x_values,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'exp': np.exp, 'log': np.log, 'log10': np.log10, 'log2': np.log2,
            'sqrt': np.sqrt, 'abs': np.abs, 'sign': np.sign,
            'pi': np.pi, 'e': np.e,
            'floor': np.floor, 'ceil': np.ceil, 'round': np.round,
            'deg': np.degrees, 'rad': np.radians,
            'min': np.minimum, 'max': np.maximum,
            '__builtins__': {}
        }
        
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = eval(expression, safe_dict)
                return np.array(result, dtype=float)
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            return None
    
    def plot_single_equation(self):
        """Plot single equation on main tab"""
        equation = self.current_equation.get().strip()
        if not equation:
            return
            
        self.ax.clear()
        self.setup_plot_style(self.ax)
        
        x_min, x_max = self.x_min.get(), self.x_max.get()
        x = np.linspace(x_min, x_max, 1000)
        y = self.safe_eval(equation, x)
        
        if y is not None:
            y = np.where(np.isfinite(y), y, np.nan)
            color = self.colors[self.color_index % len(self.colors)]
            
            self.ax.plot(x, y, color=color, linewidth=2.5, alpha=0.8, label=f'y = {equation}')
            self.ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
            self.ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)
            
            self.ax.set_xlabel('x', fontsize=12, fontweight='bold')
            self.ax.set_ylabel('y', fontsize=12, fontweight='bold')
            self.ax.set_title(f'y = {equation}', fontsize=14, fontweight='bold', pad=20)
            self.ax.legend(frameon=True, fancybox=True, shadow=True)
            
            self.color_index += 1
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def add_to_multi(self):
        """Add current equation to multi-equation list"""
        equation = self.current_equation.get().strip()
        if equation and equation not in [eq[0] for eq in self.equations]:
            color = self.colors[len(self.equations) % len(self.colors)]
            self.equations.append((equation, color, True))
            self.update_equation_list()
    
    def update_equation_list(self):
        """Update the equation listbox"""
        self.equation_listbox.delete(0, tk.END)
        for i, (eq, color, visible) in enumerate(self.equations):
            status = "✓" if visible else "✗"
            self.equation_listbox.insert(tk.END, f"{status} y = {eq}")
            # Color coding would be nice but tkinter listbox is limited
    
    def remove_equation(self):
        """Remove selected equation from list"""
        selection = self.equation_listbox.curselection()
        if selection:
            index = selection[0]
            del self.equations[index]
            self.update_equation_list()
            self.plot_all_equations()
    
    def clear_all_equations(self):
        """Clear all equations"""
        self.equations.clear()
        self.update_equation_list()
        self.multi_ax.clear()
        self.setup_plot_style(self.multi_ax)
        self.multi_canvas.draw()
    
    def plot_all_equations(self):
        """Plot all equations on multi tab"""
        self.multi_ax.clear()
        self.setup_plot_style(self.multi_ax)
        
        if not self.equations:
            self.multi_canvas.draw()
            return
        
        x_min, x_max = self.x_min.get(), self.x_max.get()
        x = np.linspace(x_min, x_max, 1000)
        
        plotted_any = False
        for equation, color, visible in self.equations:
            if visible:
                y = self.safe_eval(equation, x)
                if y is not None:
                    y = np.where(np.isfinite(y), y, np.nan)
                    self.multi_ax.plot(x, y, color=color, linewidth=2.5, alpha=0.8, 
                                     label=f'y = {equation}')
                    plotted_any = True
        
        if plotted_any:
            self.multi_ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.3)
            self.multi_ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.3)
            self.multi_ax.set_xlabel('x', fontsize=12, fontweight='bold')
            self.multi_ax.set_ylabel('y', fontsize=12, fontweight='bold')
            self.multi_ax.set_title('Multiple Equations', fontsize=14, fontweight='bold', pad=20)
            self.multi_ax.legend(frameon=True, fancybox=True, shadow=True, 
                               bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.multi_fig.tight_layout()
        self.multi_canvas.draw()
    
    def set_equation(self, equation: str):
        """Set equation from example button"""
        self.current_equation.set(equation)
        self.plot_single_equation()
    
    def clear_plot(self):
        """Clear the current plot"""
        self.ax.clear()
        self.setup_plot_style(self.ax)
        self.canvas.draw()
    
    def on_equation_change(self, event):
        """Handle real-time equation changes"""
        if hasattr(self, '_plot_job'):
            self.root.after_cancel(self._plot_job)
        self._plot_job = self.root.after(800, self.plot_single_equation)

def main():
    """Run the advanced equation grapher"""
    print("🚀 Starting Advanced Equation Grapher...")
    print("=" * 60)
    print("Features:")
    print("• Real-time equation plotting")
    print("• Multiple equations on one graph")
    print("• Extensive example library")
    print("• Advanced mathematical functions")
    print("• Beautiful, customizable plots")
    print("=" * 60)
    
    root = tk.Tk()
    app = AdvancedEquationGrapher(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    print("✨ Advanced Equation Grapher is ready!")
    root.mainloop()

if __name__ == "__main__":
    main()
