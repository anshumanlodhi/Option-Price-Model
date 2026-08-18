"""
Monte Carlo Simulation Engine for Indian Options Markets
Educational implementation focusing on European options with variance reduction
Integrates with existing Black-Scholes engine for validation and enhancement
"""

import pandas as pd
import numpy as np
from math import log, sqrt, exp, pi
from typing import Union, Dict, List, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Import your existing modules
from config.settings import get_config
from models.black_scholes import IndianBlackScholesEngine, black_scholes_call, black_scholes_put
from models.greeks import calculate_all_greeks
from data.collectors import IndianMarketDataCollector


class MonteCarloError(Exception):
    """Custom exception for Monte Carlo simulation errors"""
    pass


# =============================================================================
# CORE MONTE CARLO SIMULATION ENGINE
# =============================================================================

def generate_stock_paths(S0: float, T: float, r: float, sigma: float, q: float = 0.0,
                        n_simulations: int = 10000, n_steps: int = 252,
                        random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate stock price paths using geometric Brownian motion
    
    Mathematical Foundation:
    dS = S(μ dt + σ dW) where μ = r - q (risk-neutral drift)
    
    Discrete form: S_{t+1} = S_t * exp((r - q - σ²/2)dt + σ√dt * Z)
    where Z ~ N(0,1) are independent standard normal random variables
    
    Financial Interpretation:
    Simulates possible future stock price paths under risk-neutral measure.
    Each path represents one possible realization of stock price evolution.
    Used for pricing options where analytical solutions don't exist.
    
    Args:
        S0: Initial stock price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield (0 for NIFTY)
        n_simulations: Number of Monte Carlo paths
        n_steps: Number of time steps (252 = daily steps)
        random_seed: Seed for reproducible results
        
    Returns:
        Array of shape (n_simulations, n_steps+1) with stock price paths
    """
    if n_simulations <= 0:
        raise MonteCarloError("Number of simulations must be positive")
    if n_steps <= 0:
        raise MonteCarloError("Number of time steps must be positive")
    if T <= 0:
        raise MonteCarloError("Time to expiry must be positive")
    if sigma <= 0:
        raise MonteCarloError("Volatility must be positive")
    if S0 <= 0:
        raise MonteCarloError("Initial stock price must be positive")
    
    # Set random seed for reproducible results
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Time step size
    dt = T / n_steps
    
    # Pre-calculate constants for efficiency
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_term = sigma * sqrt(dt)
    
    # Generate random numbers for all paths and steps at once
    random_numbers = np.random.standard_normal((n_simulations, n_steps))
    
    # Initialize paths array
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0  # Initial price for all paths
    
    # Generate paths using vectorized operations
    for t in range(n_steps):
        # Geometric Brownian motion step
        paths[:, t + 1] = paths[:, t] * np.exp(drift + vol_term * random_numbers[:, t])
    
    return paths


def generate_antithetic_paths(S0: float, T: float, r: float, sigma: float, q: float = 0.0,
                             n_simulations: int = 10000, n_steps: int = 252,
                             random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate stock paths using antithetic variance reduction technique
    
    Mathematical Foundation:
    For each random variable Z, also use -Z. This creates negative correlation
    between paired paths, reducing variance by approximately 50%.
    
    Financial Interpretation:
    Antithetic variates ensure that for every "up" path, there's a corresponding
    "down" path. This balances the simulation and reduces Monte Carlo error
    without increasing computational cost significantly.
    
    Variance Reduction:
    Var(Average) = [Var(X) + Var(Y) + 2Cov(X,Y)] / 4
    Since Cov(X,Y) < 0 for antithetic variates, total variance is reduced.
    
    Args:
        Same as generate_stock_paths
        
    Returns:
        Array with antithetic paths (n_simulations must be even)
    """
    if n_simulations % 2 != 0:
        raise MonteCarloError("Number of simulations must be even for antithetic variates")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate half the required simulations
    half_sims = n_simulations // 2
    dt = T / n_steps
    
    # Pre-calculate constants
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_term = sigma * sqrt(dt)
    
    # Generate random numbers for half the paths
    random_numbers = np.random.standard_normal((half_sims, n_steps))
    
    # Initialize paths array for all simulations
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = S0
    
    # Generate first half of paths (normal)
    for t in range(n_steps):
        paths[:half_sims, t + 1] = paths[:half_sims, t] * np.exp(
            drift + vol_term * random_numbers[:, t]
        )
    
    # Generate second half using antithetic variates (-Z)
    for t in range(n_steps):
        paths[half_sims:, t + 1] = paths[half_sims:, t] * np.exp(
            drift + vol_term * (-random_numbers[:, t])
        )
    
    return paths


def monte_carlo_european_option(S0: float, K: float, T: float, r: float, sigma: float,
                               option_type: str, q: float = 0.0,
                               n_simulations: int = 10000, n_steps: int = 252,
                               use_antithetic: bool = True,
                               random_seed: Optional[int] = None) -> Dict[str, float]:
    """
    Price European options using Monte Carlo simulation
    
    Mathematical Foundation:
    Option Value = e^(-rT) * E[max(payoff, 0)]
    where payoff depends on option type and final stock price S_T
    
    For Call: payoff = S_T - K
    For Put: payoff = K - S_T
    
    Financial Interpretation:
    Monte Carlo provides unbiased estimate of option value by:
    1. Simulating many possible stock price paths
    2. Calculating payoff for each path
    3. Taking average and discounting to present value
    
    Args:
        S0, K, T, r, sigma, q: Standard option parameters
        option_type: 'CE' for call, 'PE' for put
        n_simulations: Number of Monte Carlo paths
        n_steps: Time discretization steps
        use_antithetic: Whether to use variance reduction
        random_seed: For reproducible results
        
    Returns:
        Dictionary with MC price, standard error, and simulation details
    """
    # Parameter validation
    if option_type.upper() not in ['CE', 'PE']:
        raise MonteCarloError(f"Invalid option type: {option_type}")
    
    # Generate stock price paths
    if use_antithetic and n_simulations % 2 == 0:
        paths = generate_antithetic_paths(S0, T, r, sigma, q, n_simulations, n_steps, random_seed)
        variance_reduction = "Antithetic Variates"
    else:
        paths = generate_stock_paths(S0, T, r, sigma, q, n_simulations, n_steps, random_seed)
        variance_reduction = "None"
    
    # Extract final stock prices
    final_prices = paths[:, -1]
    
    # Calculate option payoffs
    if option_type.upper() == 'CE':
        payoffs = np.maximum(final_prices - K, 0)  # Call payoff
    else:
        payoffs = np.maximum(K - final_prices, 0)  # Put payoff
    
    # Calculate present value of average payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    # Calculate standard error for confidence intervals
    payoff_std = np.std(payoffs)
    standard_error = (np.exp(-r * T) * payoff_std) / sqrt(n_simulations)
    
    # Calculate 95% confidence interval
    confidence_interval = 1.96 * standard_error
    
    return {
        'option_price': option_price,
        'standard_error': standard_error,
        'confidence_interval_95': confidence_interval,
        'lower_bound_95': option_price - confidence_interval,
        'upper_bound_95': option_price + confidence_interval,
        'payoff_std': payoff_std,
        'variance_reduction': variance_reduction,
        'n_simulations': n_simulations,
        'n_steps': n_steps,
        'final_prices_sample': final_prices[:10]  # First 10 for inspection
    }


def monte_carlo_control_variates(S0: float, K: float, T: float, r: float, sigma: float,
                                option_type: str, q: float = 0.0,
                                n_simulations: int = 10000, n_steps: int = 252,
                                random_seed: Optional[int] = None) -> Dict[str, float]:
    """
    Monte Carlo pricing with control variates variance reduction
    
    Mathematical Foundation:
    Control Variates: Use correlated variable with known expected value
    Adjusted Estimator: Y_adj = Y + c(X - E[X])
    where Y is option payoff, X is control variate, c is correlation coefficient
    
    Financial Interpretation:
    Uses geometric average Asian option as control variate since:
    1. It's correlated with European option payoff
    2. Has analytical solution for expected value
    3. Reduces variance through correlation adjustment
    
    Args:
        Same as monte_carlo_european_option
        
    Returns:
        Dictionary with control variate adjusted price and variance reduction
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate stock price paths
    paths = generate_stock_paths(S0, T, r, sigma, q, n_simulations, n_steps, random_seed)
    
    # Calculate European option payoffs
    final_prices = paths[:, -1]
    if option_type.upper() == 'CE':
        european_payoffs = np.maximum(final_prices - K, 0)
    else:
        european_payoffs = np.maximum(K - final_prices, 0)
    
    # Calculate geometric average along each path (control variate)
    geometric_averages = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    
    # Control variate payoffs (geometric Asian option)
    if option_type.upper() == 'CE':
        control_payoffs = np.maximum(geometric_averages - K, 0)
    else:
        control_payoffs = np.maximum(K - geometric_averages, 0)
    
    # Analytical value of geometric Asian option (control variate expected value)
    # Simplified calculation for educational purposes
    adj_sigma = sigma * sqrt((2 * n_steps + 1) / (6 * n_steps))
    adj_drift = 0.5 * (r - q - sigma**2/6)
    
    if option_type.upper() == 'CE':
        control_analytical = black_scholes_call(S0, K, T, adj_drift, adj_sigma, q)['call_price']
    else:
        control_analytical = black_scholes_put(S0, K, T, adj_drift, adj_sigma, q)['put_price']
    
    # Calculate optimal control variate coefficient
    covariance = np.cov(european_payoffs, control_payoffs)[0, 1]
    control_variance = np.var(control_payoffs)
    
    if control_variance > 0:
        c_optimal = -covariance / control_variance
    else:
        c_optimal = 0
    
    # Apply control variate adjustment
    control_mean = np.mean(control_payoffs)
    adjusted_payoffs = european_payoffs + c_optimal * (control_payoffs - control_mean)
    
    # Calculate control variate price
    cv_option_price = np.exp(-r * T) * np.mean(adjusted_payoffs)
    
    # Calculate variance reduction
    original_variance = np.var(european_payoffs)
    adjusted_variance = np.var(adjusted_payoffs)
    variance_reduction_ratio = adjusted_variance / original_variance if original_variance > 0 else 1
    
    # Standard error for adjusted estimator
    adjusted_std = np.std(adjusted_payoffs)
    cv_standard_error = (np.exp(-r * T) * adjusted_std) / sqrt(n_simulations)
    
    return {
        'option_price': cv_option_price,
        'standard_error': cv_standard_error,
        'confidence_interval_95': 1.96 * cv_standard_error,
        'control_variate_coeff': c_optimal,
        'variance_reduction_ratio': variance_reduction_ratio,
        'variance_reduction_pct': (1 - variance_reduction_ratio) * 100,
        'control_analytical_value': control_analytical,
        'variance_reduction': "Control Variates",
        'n_simulations': n_simulations
    }


# =============================================================================
# MONTE CARLO GREEKS CALCULATION
# =============================================================================

def monte_carlo_delta(S0: float, K: float, T: float, r: float, sigma: float,
                     option_type: str, q: float = 0.0,
                     n_simulations: int = 10000, bump_size: float = 1.0) -> Dict[str, float]:
    """
    Calculate Delta using Monte Carlo finite difference method
    
    Mathematical Foundation:
    Delta = ∂V/∂S ≈ [V(S + h) - V(S - h)] / (2h)
    where h is small bump in stock price
    
    Financial Interpretation:
    MC Delta provides alternative calculation method to analytical Greeks.
    Useful for validation and complex derivatives where analytical formulas
    don't exist. Shows hedge ratio from simulation perspective.
    
    Args:
        Standard option parameters plus:
        bump_size: Size of stock price bump for finite difference
        
    Returns:
        Dictionary with MC delta and comparison metrics
    """
    # Calculate option prices at S+h and S-h
    price_up = monte_carlo_european_option(
        S0 + bump_size, K, T, r, sigma, option_type, q, n_simulations, random_seed=42
    )['option_price']
    
    price_down = monte_carlo_european_option(
        S0 - bump_size, K, T, r, sigma, option_type, q, n_simulations, random_seed=42
    )['option_price']
    
    # Calculate finite difference delta
    mc_delta = (price_up - price_down) / (2 * bump_size)
    
    # Calculate analytical delta for comparison
    analytical_greeks = calculate_all_greeks(S0, K, T, r, sigma, option_type, q)
    analytical_delta = analytical_greeks['delta']
    
    # Calculate error metrics
    absolute_error = abs(mc_delta - analytical_delta)
    relative_error = absolute_error / abs(analytical_delta) if analytical_delta != 0 else np.inf
    
    return {
        'mc_delta': mc_delta,
        'analytical_delta': analytical_delta,
        'absolute_error': absolute_error,
        'relative_error_pct': relative_error * 100,
        'bump_size': bump_size,
        'price_up': price_up,
        'price_down': price_down
    }


def monte_carlo_gamma(S0: float, K: float, T: float, r: float, sigma: float,
                     option_type: str, q: float = 0.0,
                     n_simulations: int = 10000, bump_size: float = 1.0) -> Dict[str, float]:
    """
    Calculate Gamma using Monte Carlo second-order finite difference
    
    Mathematical Foundation:
    Gamma = ∂²V/∂S² ≈ [V(S + h) - 2V(S) + V(S - h)] / h²
    
    Financial Interpretation:
    MC Gamma shows convexity from simulation perspective.
    Important for dynamic hedging and understanding option behavior
    near the strike price.
    
    Args:
        Same as monte_carlo_delta
        
    Returns:
        Dictionary with MC gamma and validation metrics
    """
    # Calculate option prices at S+h, S, and S-h
    price_up = monte_carlo_european_option(
        S0 + bump_size, K, T, r, sigma, option_type, q, n_simulations, random_seed=42
    )['option_price']
    
    price_center = monte_carlo_european_option(
        S0, K, T, r, sigma, option_type, q, n_simulations, random_seed=42
    )['option_price']
    
    price_down = monte_carlo_european_option(
        S0 - bump_size, K, T, r, sigma, option_type, q, n_simulations, random_seed=42
    )['option_price']
    
    # Calculate finite difference gamma
    mc_gamma = (price_up - 2 * price_center + price_down) / (bump_size ** 2)
    
    # Analytical gamma for comparison
    analytical_greeks = calculate_all_greeks(S0, K, T, r, sigma, option_type, q)
    analytical_gamma = analytical_greeks['gamma']
    
    # Error metrics
    absolute_error = abs(mc_gamma - analytical_gamma)
    relative_error = absolute_error / analytical_gamma if analytical_gamma != 0 else np.inf
    
    return {
        'mc_gamma': mc_gamma,
        'analytical_gamma': analytical_gamma,
        'absolute_error': absolute_error,
        'relative_error_pct': relative_error * 100,
        'bump_size': bump_size,
        'price_up': price_up,
        'price_center': price_center,
        'price_down': price_down
    }


# =============================================================================
# CONVERGENCE ANALYSIS AND OPTIMIZATION
# =============================================================================

def analyze_monte_carlo_convergence(S0: float, K: float, T: float, r: float, sigma: float,
                                   option_type: str, q: float = 0.0,
                                   simulation_counts: List[int] = None) -> pd.DataFrame:
    """
    Analyze Monte Carlo convergence as number of simulations increases
    
    Mathematical Foundation:
    Central Limit Theorem: MC error decreases as 1/√N where N = simulations
    Standard Error = σ/√N where σ is payoff standard deviation
    
    Financial Interpretation:
    Convergence analysis helps determine optimal number of simulations
    for desired accuracy. Shows trade-off between computational cost
    and pricing precision. Essential for production MC implementations.
    
    Args:
        Standard option parameters plus:
        simulation_counts: List of simulation counts to test
        
    Returns:
        DataFrame with convergence analysis results
    """
    if simulation_counts is None:
        simulation_counts = [100, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    
    # Calculate analytical benchmark
    if option_type.upper() == 'CE':
        analytical_price = black_scholes_call(S0, K, T, r, sigma, q)['call_price']
    else:
        analytical_price = black_scholes_put(S0, K, T, r, sigma, q)['put_price']
    
    convergence_results = []
    
    for n_sims in simulation_counts:
        # Run multiple trials for each simulation count
        trials = 10
        mc_prices = []
        
        for trial in range(trials):
            mc_result = monte_carlo_european_option(
                S0, K, T, r, sigma, option_type, q, n_sims, 
                use_antithetic=True, random_seed=42 + trial
            )
            mc_prices.append(mc_result['option_price'])
        
        # Calculate statistics across trials
        mean_mc_price = np.mean(mc_prices)
        std_mc_price = np.std(mc_prices)
        
        # Calculate metrics
        bias = mean_mc_price - analytical_price
        rmse = sqrt(np.mean([(price - analytical_price)**2 for price in mc_prices]))
        
        convergence_results.append({
            'n_simulations': n_sims,
            'mean_mc_price': mean_mc_price,
            'analytical_price': analytical_price,
            'bias': bias,
            'std_across_trials': std_mc_price,
            'rmse': rmse,
            'theoretical_std_error': std_mc_price / sqrt(n_sims),
            'relative_error_pct': abs(bias / analytical_price) * 100,
            'trials_count': trials
        })
    
    convergence_df = pd.DataFrame(convergence_results)
    
    print("Monte Carlo Convergence Analysis:")
    print(f"Analytical Price: ₹{analytical_price:.4f}")
    print(f"Convergence from {min(simulation_counts)} to {max(simulation_counts)} simulations")
    
    return convergence_df


def compare_variance_reduction_methods(S0: float, K: float, T: float, r: float, sigma: float,
                                     option_type: str, q: float = 0.0,
                                     n_simulations: int = 10000) -> pd.DataFrame:
    """
    Compare different variance reduction techniques
    
    Educational Purpose:
    Demonstrates effectiveness of different variance reduction methods:
    1. Standard Monte Carlo (baseline)
    2. Antithetic Variates
    3. Control Variates
    
    Shows variance reduction, computational efficiency, and accuracy improvements.
    
    Args:
        Standard option parameters and simulation count
        
    Returns:
        DataFrame comparing all variance reduction methods
    """
    # Calculate analytical benchmark
    if option_type.upper() == 'CE':
        analytical_price = black_scholes_call(S0, K, T, r, sigma, q)['call_price']
    else:
        analytical_price = black_scholes_put(S0, K, T, r, sigma, q)['put_price']
    
    comparison_results = []
    
    # Method 1: Standard Monte Carlo
    start_time = time.time()
    standard_mc = monte_carlo_european_option(
        S0, K, T, r, sigma, option_type, q, n_simulations, 
        use_antithetic=False, random_seed=42
    )
    standard_time = time.time() - start_time
    
    comparison_results.append({
        'method': 'Standard MC',
        'option_price': standard_mc['option_price'],
        'standard_error': standard_mc['standard_error'],
        'confidence_interval': standard_mc['confidence_interval_95'],
        'computation_time': standard_time,
        'bias': standard_mc['option_price'] - analytical_price,
        'variance_reduction_vs_standard': 1.0,  # Baseline
        'efficiency_ratio': 1.0  # Baseline
    })
    
    # Method 2: Antithetic Variates
    start_time = time.time()
    antithetic_mc = monte_carlo_european_option(
        S0, K, T, r, sigma, option_type, q, n_simulations,
        use_antithetic=True, random_seed=42
    )
    antithetic_time = time.time() - start_time
    
    # Calculate variance reduction
    variance_reduction_antithetic = (standard_mc['standard_error']**2) / (antithetic_mc['standard_error']**2)
    efficiency_antithetic = variance_reduction_antithetic * (standard_time / antithetic_time)
    
    comparison_results.append({
        'method': 'Antithetic Variates',
        'option_price': antithetic_mc['option_price'],
        'standard_error': antithetic_mc['standard_error'],
        'confidence_interval': antithetic_mc['confidence_interval_95'],
        'computation_time': antithetic_time,
        'bias': antithetic_mc['option_price'] - analytical_price,
        'variance_reduction_vs_standard': variance_reduction_antithetic,
        'efficiency_ratio': efficiency_antithetic
    })
    
    # Method 3: Control Variates
    start_time = time.time()
    control_mc = monte_carlo_control_variates(
        S0, K, T, r, sigma, option_type, q, n_simulations, random_seed=42
    )
    control_time = time.time() - start_time
    
    # Calculate variance reduction for control variates
    variance_reduction_control = (standard_mc['standard_error']**2) / (control_mc['standard_error']**2)
    efficiency_control = variance_reduction_control * (standard_time / control_time)
    
    comparison_results.append({
        'method': 'Control Variates',
        'option_price': control_mc['option_price'],
        'standard_error': control_mc['standard_error'],
        'confidence_interval': control_mc['confidence_interval_95'],
        'computation_time': control_time,
        'bias': control_mc['option_price'] - analytical_price,
        'variance_reduction_vs_standard': variance_reduction_control,
        'efficiency_ratio': efficiency_control
    })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    print("Variance Reduction Methods Comparison:")
    print(f"Analytical Benchmark: ₹{analytical_price:.4f}")
    print(f"Simulation Count: {n_simulations:,}")
    
    return comparison_df


# =============================================================================
# COMPREHENSIVE MONTE CARLO VALIDATOR
# =============================================================================

class IndianMonteCarloValidator:
    """
    Comprehensive Monte Carlo validation system
    Integrates with your existing Black-Scholes and Greeks engines
    """
    
    def __init__(self):
        """Initialize MC validator with existing engines"""
        self.config = get_config()
        self.bs_engine = IndianBlackScholesEngine()
        
        # Default MC parameters from config
        self.default_simulations = self.config.calculations.monte_carlo['simulations']
        self.default_time_steps = self.config.calculations.monte_carlo['time_steps']
        self.random_seed = self.config.calculations.monte_carlo['random_seed']
        
        print("IndianMonteCarloValidator initialized with comprehensive validation suite")
    
    def validate_against_black_scholes(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate Monte Carlo prices against Black-Scholes analytical solutions
        
        Perfect for learning: Shows how MC converges to analytical solutions
        and where differences might occur due to discretization or random error.
        """
        validation_results = []
        
        for idx, row in options_df.iterrows():
            try:
                # Get Black-Scholes analytical price
                bs_result = self.bs_engine.price_single_option(
                    spot_price=row['Spot_Price'],
                    strike_price=row['Strike'],
                    time_to_expiry=row['Time_to_Expiry'],
                    volatility=row.get('Volatility', 0.2),
                    option_type=row['Option_Type'],
                    symbol=row.get('Symbol', 'NIFTY'),
                    show_details=False
                )
                
                analytical_price = bs_result['option_price']
                
                # Get Monte Carlo price with variance reduction
                mc_result = monte_carlo_european_option(
                    S0=row['Spot_Price'],
                    K=row['Strike'],
                    T=row['Time_to_Expiry'],
                    r=bs_result['risk_free_rate'],
                    sigma=row.get('Volatility', 0.2),
                    option_type=row['Option_Type'],
                    q=bs_result['dividend_yield'],
                    n_simulations=self.default_simulations,
                    use_antithetic=True,
                    random_seed=self.random_seed
                )
                
                mc_price = mc_result['option_price']
                
                # Calculate validation metrics
                absolute_error = abs(mc_price - analytical_price)
                relative_error = (absolute_error / analytical_price) * 100
                
                # Check if MC price is within confidence interval
                within_ci = (analytical_price >= mc_result['lower_bound_95'] and 
                           analytical_price <= mc_result['upper_bound_95'])
                
                validation_results.append({
                    'Strike': row['Strike'],
                    'Option_Type': row['Option_Type'],
                    'Time_to_Expiry': row['Time_to_Expiry'],
                    'BS_Price': analytical_price,
                    'MC_Price': mc_price,
                    'Absolute_Error': absolute_error,
                    'Relative_Error_Pct': relative_error,
                    'MC_Standard_Error': mc_result['standard_error'],
                    'Within_95_CI': within_ci,
                    'CI_Lower': mc_result['lower_bound_95'],
                    'CI_Upper': mc_result['upper_bound_95'],
                    'Moneyness': row['Spot_Price'] / row['Strike']
                })
                
            except Exception as e:
                print(f"Validation failed for row {idx}: {e}")
        
        validation_df = pd.DataFrame(validation_results)
        
        if len(validation_df) > 0:
            print(f"Monte Carlo vs Black-Scholes Validation:")
            print(f"Mean Relative Error: {validation_df['Relative_Error_Pct'].mean():.3f}%")
            print(f"Max Relative Error: {validation_df['Relative_Error_Pct'].max():.3f}%")
            print(f"Prices within 95% CI: {validation_df['Within_95_CI'].mean():.1%}")
        
        return validation_df
    
    def validate_greeks_accuracy(self, S0: float = 24000, K: float = 24000, 
                               T: float = 30/365, sigma: float = 0.16) -> Dict[str, any]:
        """
        Validate Monte Carlo Greeks against analytical Greeks
        """
        print("Validating MC Greeks against analytical calculations...")
        
        greeks_validation = {}
        
        for option_type in ['CE', 'PE']:
            # Calculate MC Delta and Gamma
            mc_delta_result = monte_carlo_delta(S0, K, T, 0.065, sigma, option_type, 
                                              n_simulations=50000, bump_size=1.0)
            
            mc_gamma_result = monte_carlo_gamma(S0, K, T, 0.065, sigma, option_type,
                                              n_simulations=50000, bump_size=1.0)
            
            greeks_validation[f'{option_type}_delta'] = mc_delta_result
            greeks_validation[f'{option_type}_gamma'] = mc_gamma_result
            
            print(f"\n{option_type} Option Greeks Validation:")
            print(f"Delta - MC: {mc_delta_result['mc_delta']:.4f}, "
                  f"Analytical: {mc_delta_result['analytical_delta']:.4f}, "
                  f"Error: {mc_delta_result['relative_error_pct']:.2f}%")
            print(f"Gamma - MC: {mc_gamma_result['mc_gamma']:.6f}, "
                  f"Analytical: {mc_gamma_result['analytical_gamma']:.6f}, "
                  f"Error: {mc_gamma_result['relative_error_pct']:.2f}%")
        
        return greeks_validation
    
    def run_comprehensive_validation(self, options_data: pd.DataFrame) -> Dict[str, any]:
        """
        Run complete Monte Carlo validation suite
        """
        print("=== Running Comprehensive Monte Carlo Validation ===")
        
        validation_results = {}
        
        # 1. Price validation against Black-Scholes
        price_validation = self.validate_against_black_scholes(options_data)
        validation_results['price_validation'] = price_validation
        
        # 2. Convergence analysis
        sample_option = options_data.iloc[0]
        convergence_analysis = analyze_monte_carlo_convergence(
            sample_option['Spot_Price'],
            sample_option['Strike'],
            sample_option['Time_to_Expiry'],
            0.065,
            sample_option.get('Volatility', 0.2),
            sample_option['Option_Type']
        )
        validation_results['convergence_analysis'] = convergence_analysis
        
        # 3. Variance reduction comparison
        variance_comparison = compare_variance_reduction_methods(
            sample_option['Spot_Price'],
            sample_option['Strike'],
            sample_option['Time_to_Expiry'],
            0.065,
            sample_option.get('Volatility', 0.2),
            sample_option['Option_Type']
        )
        validation_results['variance_reduction'] = variance_comparison
        
        # 4. Greeks validation
        greeks_validation = self.validate_greeks_accuracy()
        validation_results['greeks_validation'] = greeks_validation
        
        print("=== Comprehensive Monte Carlo Validation Complete ===")
        
        return validation_results


# =============================================================================
# VISUALIZATION AND ANALYSIS FUNCTIONS
# =============================================================================

def plot_monte_carlo_convergence(convergence_df: pd.DataFrame):
    """
    Visualize Monte Carlo convergence analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Monte Carlo Convergence Analysis', fontsize=16)
    
    # 1. Price convergence
    axes[0, 0].semilogx(convergence_df['n_simulations'], convergence_df['mean_mc_price'], 'b-o')
    axes[0, 0].axhline(y=convergence_df['analytical_price'].iloc[0], color='r', linestyle='--', 
                       label='Analytical Price')
    axes[0, 0].set_xlabel('Number of Simulations')
    axes[0, 0].set_ylabel('Option Price (₹)')
    axes[0, 0].set_title('Price Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE convergence
    axes[0, 1].loglog(convergence_df['n_simulations'], convergence_df['rmse'], 'g-o')
    axes[0, 1].set_xlabel('Number of Simulations')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE vs Simulations')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Relative error
    axes[1, 0].semilogx(convergence_df['n_simulations'], convergence_df['relative_error_pct'], 'm-o')
    axes[1, 0].set_xlabel('Number of Simulations')
    axes[1, 0].set_ylabel('Relative Error (%)')
    axes[1, 0].set_title('Relative Error Convergence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Standard error
    axes[1, 1].loglog(convergence_df['n_simulations'], convergence_df['std_across_trials'], 'c-o',
                      label='Empirical Std Error')
    theoretical_se = convergence_df['std_across_trials'].iloc[0] / np.sqrt(convergence_df['n_simulations'] / convergence_df['n_simulations'].iloc[0])
    axes[1, 1].loglog(convergence_df['n_simulations'], theoretical_se, 'r--',
                      label='Theoretical 1/√N')
    axes[1, 1].set_xlabel('Number of Simulations')
    axes[1, 1].set_ylabel('Standard Error')
    axes[1, 1].set_title('Standard Error Convergence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_monte_carlo_demo_data() -> pd.DataFrame:
    """
    Create comprehensive options data for Monte Carlo demonstration
    """
    # NIFTY options chain for demo
    spot_price = 24000
    strikes = np.array([22000, 23000, 24000, 25000, 26000])
    expiries = np.array([7, 30, 60]) / 365
    
    demo_data = []
    
    for expiry in expiries:
        for strike in strikes:
            for option_type in ['CE', 'PE']:
                # Realistic volatility with smile
                moneyness = spot_price / strike
                base_vol = 0.16 + 0.03 * abs(1 - moneyness)  # Volatility smile
                
                demo_data.append({
                    'Spot_Price': spot_price,
                    'Strike': strike,
                    'Time_to_Expiry': expiry,
                    'Option_Type': option_type,
                    'Volatility': base_vol,
                    'Symbol': 'NIFTY'
                })
    
    return pd.DataFrame(demo_data)


# =============================================================================
# COMPREHENSIVE DEMONSTRATION
# =============================================================================

def run_monte_carlo_demonstration():
    """
    Complete Monte Carlo demonstration showcasing all capabilities
    """
    print("=== Comprehensive Monte Carlo Simulation Demonstration ===")
    
    # Initialize validator
    validator = IndianMonteCarloValidator()
    
    # Create demo data
    demo_options = create_monte_carlo_demo_data()
    print(f"Created {len(demo_options)} demo options for testing")
    
    # Run comprehensive validation
    validation_results = validator.run_comprehensive_validation(demo_options)
    
    # Display key results
    print("\n=== Key Results Summary ===")
    
    # Price validation summary
    price_val = validation_results['price_validation']
    if len(price_val) > 0:
        print(f"Price Validation (MC vs Black-Scholes):")
        print(f"  Mean Relative Error: {price_val['Relative_Error_Pct'].mean():.3f}%")
        print(f"  Max Relative Error: {price_val['Relative_Error_Pct'].max():.3f}%")
        print(f"  Confidence Interval Coverage: {price_val['Within_95_CI'].mean():.1%}")
    
    # Variance reduction effectiveness
    var_reduction = validation_results['variance_reduction']
    print(f"\nVariance Reduction Effectiveness:")
    for idx, row in var_reduction.iterrows():
        print(f"  {row['method']}: {row['variance_reduction_vs_standard']:.2f}x variance reduction")
    
    # Convergence insights
    convergence = validation_results['convergence_analysis']
    final_accuracy = convergence.iloc[-1]['relative_error_pct']
    print(f"\nConvergence Analysis:")
    print(f"  Final accuracy ({convergence.iloc[-1]['n_simulations']:,} sims): {final_accuracy:.3f}%")
    print(f"  RMSE improvement: {convergence.iloc[0]['rmse']:.4f} → {convergence.iloc[-1]['rmse']:.4f}")
    
    print("\n=== Monte Carlo System Ready for Production Use ===")
    
    return validation_results


# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=== Indian Options Monte Carlo Simulation Engine ===")
    
    # Run comprehensive demonstration
    demo_results = run_monte_carlo_demonstration()
    
    print("\n=== Available Monte Carlo Components ===")
    print("✓ Core path generation (Geometric Brownian Motion)")
    print("✓ Variance reduction (Antithetic Variates, Control Variates)")
    print("✓ European options pricing with MC")
    print("✓ Monte Carlo Greeks calculation")
    print("✓ Convergence analysis and optimization")
    print("✓ Comprehensive validation against Black-Scholes")
    print("✓ Integration with existing pricing and Greeks engines")
    
    print(f"\nYour Indian Options Pricing System is now COMPLETE!")
    print("All planned components implemented:")
    print("• Data Collection & Market Structure")
    print("• Black-Scholes Pricing Engine")  
    print("• Analytical Greeks Calculator")
    print("• Volatility Surface & Model Validation")
    print("• Monte Carlo Simulation Engine ← NOW COMPLETE")
    
    print("\nReady for resume showcase, academic presentation, and practical options analysis!")
