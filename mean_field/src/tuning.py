# @title Helper Functions
# Imports
import numpy as np
from scipy.optimize import fsolve

def F_sigmoid(x, a, b, d=None):
    """
    Population activation function, F-I curve

    Args:
      x     : the population input
      d     : the slope of the function
      a     : the gain of the function
      b     : the threshold of the function

    Returns:
      f     : the population activation response f(x) for input x
    """

    # add the expression of f = F(x)
    f = (1 + np.exp(-a * (x - b))) ** -1 - (1 + np.exp(a * b)) ** -1

    return f


def dF_sigmoid(x, a, b, d=None):
    """
    Derivative of the population activation function.

    Args:
      x     : the population input
      a     : the gain of the function
      b : the threshold of the function

    Returns:
      dFdx  :  Derivative of the population activation function.
    """

    dFdx = a * np.exp(-a * (x - b)) * (1 + np.exp(-a * (x - b))) ** -2

    return dFdx


def F_sigmoid_inv(y, a, b, d=None):
    """
    Args:
      x         : the population input
      a         : the gain of the function
      b         : the threshold of the function
      d         : the slope of the function

    Returns:
      F_inverse : value of the inverse function
    """

    # x = b - (1 / a) * np.log((y + (1 + np.exp(a * b)) ** -1) ** -1 - 1)
    # x = b - (1 / a) * np.log(1/(  1/(1 + np.exp(a * b)) + y  ) - 1)

    # Compute the term inside the logarithm
    term = (1 / (1 + np.exp(a*b)) + y)
    safe_values = 1/term - 1
    x = b - (1 / a) * np.log(safe_values)
    # # Prevent division by zero by ensuring the denominator is never zero
    # denominator = np.where(term != 0, term, np.nan)  # Replace zero with NaN to avoid division by zero
    # numerator = 1 - denominator
    #
    # # Clip values for logarithm to avoid log(0) or negative values
    # safe_values = np.clip(numerator / denominator, 1e-10, np.inf)
    #
    # # Calculate the inverse
    # x = b - (1 / a) * np.log(safe_values)
    return x

def F_linear(x, a, b, d=None):
    """
    Population activation function, F-I curve

    Args:
     x     : the population input
     d     : the slope of the function
     a     : the gain of the function
     b     : the threshold of the function

    Returns:
     f     : the population activation response f(x) for input x
    """
    f = (a * x - b)
    return f

def dF_linear(y, a, b, d=None):
    """
    derivative of linear transfer function.
    """
    return np.full_like(y, a)

def F_leaky_relu(x, a, b, d):
    """
    Population activation function, F-I curve

    Args:
     x     : the population input
     d     : the slope of the function
     a     : the gain of the function
     b     : the threshold of the function

    Returns:
     f     : the population activation response f(x) for input x
    """
    numerator = (a * x - b)
    denominator = (1 - np.exp(-d * (a * x - b)))
    f = np.divide(numerator, denominator, out=np.full_like(numerator, np.inf), where=denominator != 0)

    return f

def F_elu(x, a, b, d):
    return np.where(a * x - b >= 0, a * x - b, d * (np.exp(a * x - b) - 1))
def F_elu_inv(y, a, b, d):
    """
    Inverse of the ELU function F_elu:
    F_elu(x, a, b, d) = a * x - b if a * x - b >= 0
                        d * (exp(a * x - b) - 1) if a * x - b < 0
    This function computes the inverse of F_elu.
    """
    x = np.where(y >= 0, (y + b) / a, (np.log(y / d + 1) + b) / a)

    return x

def dF_elu(x, a, b, d):
    """
     Derivative of the ELU function F_elu:
     F_elu(x, a, b, d) = a * x - b if a * x - b >= 0
                         d * (exp(a * x - b) - 1) if a * x - b < 0
     This function computes the derivative of F_elu.
     """
    derivative = np.where(a * x - b >= 0, a, d * a * np.exp(a * x - b))
    return derivative

def F_leaky_relu_inv(y, a, b, d):
    """
    Inverse of the function y = F(x) = (a * x - b) / (1 - np.exp(-d * (a * x - b))).
    Uses numerical solving for x.
    """

    # Define the equation to solve: y - (a * x - b) / (1 - np.exp(-d * (a * x - b))) = 0
    def equation(x):
        y_prime = F_leaky_relu(x, a, b, d)
        return y_prime - y

    # Use fsolve to find the root of the equation, starting from an initial guess
    x_solution = fsolve(equation, x0=np.zeros_like(y))  # Initial guess for x is 0
    return x_solution[0]

def dF_leaky_relu(x, a, b, d):
    """
    Derivative of the function F = (a * x - b) / (1 - np.exp(-d * (a * x - b))).
    """
    u = a * x - b
    v = 1 - np.exp(-d * u)

    # Derivatives of u and v
    du_dx = a
    dv_dx = d * np.exp(-d * u) * du_dx  # Chain rule applied

    # Quotient rule: (v * du/dx - u * dv/dx) / v^2
    return (v * du_dx - u * dv_dx) / v**2


def F_linear_inv(y, a, b, d=None):
    """
    Inverse of linear transfer function.
    """
    return (y + b) / a




