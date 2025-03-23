import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, log, pi
from matplotlib import cm
import time

def limit_derivative(f, x, h, method='forward'):
    """
    Calculate numerical derivative using various finite difference methods.
    
    Parameters:
        f (function): Function to differentiate
        x (float): Point at which to calculate the derivative
        h (float): Step size
        method (str): Differentiation method - 'forward', 'backward', 'central', or 'complex'
    
    Returns:
        float: Approximated derivative value
    """
    if method == 'forward':
        return (f(x + h) - f(x)) / h
    elif method == 'backward':
        return (f(x) - f(x - h)) / h
    elif method == 'central':
        return (f(x + h) - f(x - h)) / (2 * h)
    elif method == 'complex':
        # Complex step differentiation - highly accurate for analytic functions
        # Requires function to support complex numbers
        try:
            return np.imag(f(x + 1j * h)) / h
        except (TypeError, ValueError):
            print(f"Complex step method not supported for this function. Falling back to central difference.")
            return (f(x + h) - f(x - h)) / (2 * h)
    else:
        raise ValueError(f"Unknown method: {method}")

# Test functions with their exact derivatives
class TestFunction:
    def __init__(self, func, derivative, name, domain=(1, 10)):
        self.func = func
        self.derivative = derivative
        self.name = name
        self.domain = domain

# Define test functions
test_functions = [
    TestFunction(
        func=lambda x: sin(x),
        derivative=lambda x: cos(x),
        name="f1(x) = sin(x)",
        domain=(0, 2*pi)
    ),
    TestFunction(
        func=lambda x: x**4,
        derivative=lambda x: 4*x**3,
        name="f2(x) = x^4"
    ),
    TestFunction(
        func=lambda x: x**2 * log(x),
        derivative=lambda x: 2*x*log(x) + x,
        name="f3(x) = x^2 * log(x)"
    ),
    TestFunction(
        func=lambda x: np.exp(-x**2),
        derivative=lambda x: -2*x*np.exp(-x**2),
        name="f4(x) = e^(-x^2)",
        domain=(-3, 3)
    )
]

def calculate_derivatives_at_point(f, x_point, function_name, methods=None):
    """
    Calculate derivatives of a function at a specific point using different methods and step sizes.
    
    Parameters:
        f (function): The function to differentiate
        x_point (float): Point at which to calculate derivatives
        function_name (str): Name of the function for display
        methods (list): List of methods to use (default: forward, central, complex)
    
    Returns:
        dict: Dictionary of results for each method and step size
    """
    if methods is None:
        methods = ['forward', 'central', 'complex']
    
    print(f"Derivatives of {function_name} at x={x_point}:")
    print(f"{'Method':<10} {'Step Size':<10} {'Approximation':<15} {'Error':<10}")
    print("-" * 50)
    
    # Try to evaluate the exact derivative for comparison
    try:
        exact_derivative = None
        for tf in test_functions:
            if tf.name == function_name:
                exact_derivative = tf.derivative(x_point)
                break
    except:
        exact_derivative = None
    
    results = {}
    
    # Test with different step sizes
    h_values = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    for method in methods:
        results[method] = []
        for h in h_values:
            try:
                start_time = time.time()
                d = limit_derivative(f, x_point, h, method)
                end_time = time.time()
                computation_time = (end_time - start_time) * 1000  # in milliseconds
                
                error = abs(d - exact_derivative) if exact_derivative is not None else float('nan')
                
                results[method].append({
                    'h': h, 
                    'value': d, 
                    'error': error,
                    'time': computation_time
                })
                
                error_str = f"{error:.6e}" if exact_derivative is not None else "N/A"
                print(f"{method:<10} {h:<10.6f} {d:<15.8f} {error_str}")
            except Exception as e:
                print(f"{method:<10} {h:<10.6f} Failed: {str(e)}")
    
    print()
    return results

def plot_error_analysis(results, function_name):
    """
    Plot error analysis for different derivative methods.
    
    Parameters:
        results (dict): Dictionary of results from calculate_derivatives_at_point
        function_name (str): Name of the function
    """
    plt.figure(figsize=(10, 6))
    
    for method, method_results in results.items():
        h_values = [r['h'] for r in method_results if not np.isnan(r['error'])]
        errors = [r['error'] for r in method_results if not np.isnan(r['error'])]
        
        if h_values and errors:
            plt.loglog(h_values, errors, 'o-', label=f"{method} method")
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Step Size (h)')
    plt.ylabel('Absolute Error')
    plt.title(f'Error Analysis for Derivatives of {function_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_approx_deriv(f, f_name, true_derivative_func=None, methods=None, domain=None):
    """
    Plot approximated derivatives using different methods and step sizes.
    
    Parameters:
        f (function): The function to differentiate
        f_name (str): Name of the function for display
        true_derivative_func (function): The exact derivative function
        methods (list): List of methods to use (default: forward, central)
        domain (tuple): Domain of the function (min_x, max_x)
    """
    if methods is None:
        methods = ['forward', 'central']
    
    if domain is None:
        domain = (1, 10)
    
    x_vals = np.linspace(domain[0], domain[1], 500)
    h_vals = [0.1, 0.01]
    
    plt.figure(figsize=(12, 8))
    
    # Plot true derivative if available
    if true_derivative_func:
        y_true = [true_derivative_func(x) for x in x_vals]
        plt.plot(x_vals, y_true, 'k-', label="True Derivative", linewidth=3)
    
    # Plot approximated derivatives
    colors = cm.rainbow(np.linspace(0, 1, len(methods) * len(h_vals)))
    color_idx = 0
    
    for method in methods:
        for h in h_vals:
            approx_derivs = [limit_derivative(f, x, h, method) for x in x_vals]
            plt.plot(x_vals, approx_derivs, linestyle='--', 
                     label=f"{method}, h={h}", color=colors[color_idx])
            color_idx += 1
    
    plt.legend()
    plt.title(f"Approximated Derivatives of {f_name}")
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_function_and_derivative(func_obj):
    """
    Visualize a function and its derivative together.
    
    Parameters:
        func_obj (TestFunction): Function object containing the function and derivative
    """
    domain = func_obj.domain
    x_vals = np.linspace(domain[0], domain[1], 500)
    
    # Calculate function and derivative values
    y_func = [func_obj.func(x) for x in x_vals]
    y_deriv = [func_obj.derivative(x) for x in x_vals]
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot function on first axis
    ax1.plot(x_vals, y_func, 'b-', linewidth=2, label=f"{func_obj.name}")
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis and plot derivative
    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_deriv, 'r-', linewidth=2, label=f"f'(x)")
    ax2.set_ylabel("f'(x)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add a title
    plt.title(f"Function and its Derivative: {func_obj.name}")
    
    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage and demonstration
if __name__ == "__main__":
    print("Advanced Numerical Derivatives Analysis")
    print("======================================\n")
    
    # Demonstrate error analysis
    func_idx = 1  # Use f2(x) = x^4
    test_func = test_functions[func_idx]
    
    print(f"Analyzing: {test_func.name}\n")
    
    # Calculate derivatives at x=2 using different methods
    results = calculate_derivatives_at_point(
        test_func.func, 
        x_point=2, 
        function_name=test_func.name,
        methods=['forward', 'backward', 'central', 'complex']
    )
    
    # Visualize the function and its derivative
    visualize_function_and_derivative(test_func)
    
    # Plot approximation error analysis
    plot_error_analysis(results, test_func.name)
    
    # Plot approximated derivatives compared to the true derivative
    plot_approx_deriv(
        test_func.func, 
        test_func.name, 
        true_derivative_func=test_func.derivative,
        methods=['forward', 'central', 'complex'],
        domain=test_func.domain
    )
    
    # Uncomment to test with other functions
    # for func in test_functions:
    #     visualize_function_and_derivative(func)
    #     plot_approx_deriv(
    #         func.func, 
    #         func.name, 
    #         true_derivative_func=func.derivative,
    #         methods=['forward', 'central', 'complex'],
    #         domain=func.domain
    #     )