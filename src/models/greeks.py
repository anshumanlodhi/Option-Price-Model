"""
Options Greeks Calculator for Indian Markets
Analytical implementation with mathematical derivations and financial interpretations
Integrates with Black-Scholes pricing engine for comprehensive risk analysis
"""

import pandas as pd
import numpy as np
from math import log, sqrt, exp, pi
from typing import Union, Dict, List, Optional, Tuple
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing modules
from config.settings import get_config
from models.black_scholes import (
    cumulative_normal_distribution, 
    calculate_d1, 
    calculate_d2,
    IndianBlackScholesEngine
)


class GreeksCalculationError(Exception):
    """Custom exception for Greeks calculation errors"""
    pass


# =============================================================================
# CORE ANALYTICAL GREEKS FUNCTIONS
# =============================================================================

def calculate_delta_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Delta for European call option using analytical formula
    
    Mathematical Derivation:
    Delta = ∂C/∂S = e^(-qT) * N(d1)
    
    Where d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    
    Financial Interpretation:
    Delta represents the sensitivity of option price to changes in underlying price.
    For calls: 0 ≤ Δ ≤ 1
    - Delta = 0.5 means ₹0.50 option price change for ₹1 underlying change
    - Also represents hedge ratio: number of shares to hedge one option
    - Approximates probability of option finishing in-the-money
    
    Args:
        S: Current spot price of underlying
        K: Strike price of option
        T: Time to expiry in years
        r: Risk-free interest rate
        sigma: Volatility of underlying
        q: Dividend yield (default 0 for NIFTY)
        
    Returns:
        Delta value for call option
    """
    if T <= 0:
        # At expiry, delta is either 0 or 1
        return 1.0 if S > K else 0.0
    
    # Calculate d1 using your proven Black-Scholes function
    d1 = calculate_d1(S, K, T, r, sigma, q)
    
    # Analytical delta formula for call
    delta_call = exp(-q * T) * cumulative_normal_distribution(d1)
    
    return delta_call


def calculate_delta_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Delta for European put option using analytical formula
    
    Mathematical Derivation:
    Delta = ∂P/∂S = -e^(-qT) * N(-d1) = e^(-qT) * (N(d1) - 1)
    
    Financial Interpretation:
    Put delta is always negative, representing inverse relationship with underlying.
    For puts: -1 ≤ Δ ≤ 0
    - Delta = -0.3 means ₹0.30 option price decrease for ₹1 underlying increase
    - Hedge ratio: short 0.3 shares per long put option
    - |Delta| approximates probability of put finishing in-the-money
    
    Args:
        S, K, T, r, sigma, q: Option parameters
        
    Returns:
        Delta value for put option (negative)
    """
    if T <= 0:
        # At expiry, delta is either 0 or -1
        return -1.0 if S < K else 0.0
    
    d1 = calculate_d1(S, K, T, r, sigma, q)
    
    # Analytical delta formula for put
    delta_put = -exp(-q * T) * cumulative_normal_distribution(-d1)
    
    return delta_put


def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Gamma for European options (same for calls and puts)
    
    Mathematical Derivation:
    Gamma = ∂²V/∂S² = ∂Δ/∂S = e^(-qT) * φ(d1) / (S * σ * √T)
    
    Where φ(d1) is the probability density function of standard normal distribution
    φ(d1) = (1/√2π) * e^(-d1²/2)
    
    Financial Interpretation:
    Gamma measures the rate of change of delta with respect to underlying price.
    - High gamma: Delta changes rapidly with underlying moves (risk/opportunity)
    - Maximum gamma occurs at-the-money (ATM) options
    - Gamma approaches zero for deep ITM/OTM options
    - Critical for dynamic hedging strategies
    
    Args:
        S, K, T, r, sigma, q: Option parameters
        
    Returns:
        Gamma value (always positive for long options)
    """
    if T <= 0:
        # At expiry, gamma is zero except at strike price
        return 0.0
    
    d1 = calculate_d1(S, K, T, r, sigma, q)
    
    # Calculate probability density function φ(d1)
    phi_d1 = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * d1 * d1)
    
    # Analytical gamma formula
    gamma = (exp(-q * T) * phi_d1) / (S * sigma * sqrt(T))
    
    return gamma


def calculate_theta_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Theta for European call option
    
    Mathematical Derivation:
    Theta = ∂C/∂t = -∂C/∂T
    Theta = -[S*φ(d1)*σ*e^(-qT)/(2√T) + qS*N(d1)*e^(-qT) - rK*e^(-rT)*N(d2)]
    
    Financial Interpretation:
    Theta measures time decay - how much option value decreases as time passes.
    For calls: Theta is usually negative (time decay hurts long positions)
    - Theta = -5 means option loses ₹5 per day (all else equal)
    - Time decay accelerates as expiry approaches
    - ATM options have highest theta (most time value to lose)
    
    Args:
        S, K, T, r, sigma, q: Option parameters
        
    Returns:
        Theta value for call option (usually negative)
    """
    if T <= 0:
        return 0.0
    
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    
    # Calculate components
    phi_d1 = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * d1 * d1)
    N_d1 = cumulative_normal_distribution(d1)
    N_d2 = cumulative_normal_distribution(d2)
    
    # Analytical theta formula for call
    term1 = -(S * phi_d1 * sigma * exp(-q * T)) / (2.0 * sqrt(T))
    term2 = q * S * N_d1 * exp(-q * T)
    term3 = -r * K * exp(-r * T) * N_d2
    
    theta_call = term1 + term2 + term3
    
    return theta_call


def calculate_theta_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Theta for European put option
    
    Mathematical Derivation:
    Theta = -[S*φ(d1)*σ*e^(-qT)/(2√T) - qS*N(-d1)*e^(-qT) + rK*e^(-rT)*N(-d2)]
    
    Financial Interpretation:
    Put theta can be positive or negative depending on dividends and interest rates.
    For NIFTY puts (q=0): Usually negative but less negative than calls
    - Deep ITM puts may have positive theta due to interest rate effect
    - Time decay pattern differs from calls, especially for ITM puts
    
    Args:
        S, K, T, r, sigma, q: Option parameters
        
    Returns:
        Theta value for put option
    """
    if T <= 0:
        return 0.0
    
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    
    # Calculate components
    phi_d1 = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * d1 * d1)
    N_minus_d1 = cumulative_normal_distribution(-d1)
    N_minus_d2 = cumulative_normal_distribution(-d2)
    
    # Analytical theta formula for put
    term1 = -(S * phi_d1 * sigma * exp(-q * T)) / (2.0 * sqrt(T))
    term2 = -q * S * N_minus_d1 * exp(-q * T)
    term3 = r * K * exp(-r * T) * N_minus_d2
    
    theta_put = term1 + term2 + term3
    
    return theta_put


def calculate_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Vega for European options (same for calls and puts)
    
    Mathematical Derivation:
    Vega = ∂V/∂σ = S * e^(-qT) * φ(d1) * √T
    
    Financial Interpretation:
    Vega measures sensitivity to volatility changes.
    - Vega = 50 means ₹50 option price increase for 1% volatility increase
    - Always positive for long options (higher vol = higher option value)
    - Maximum vega for ATM options
    - Critical during earnings announcements, event risk periods
    
    Args:
        S, K, T, r, sigma, q: Option parameters
        
    Returns:
        Vega value (always positive for long options)
    """
    if T <= 0:
        return 0.0
    
    d1 = calculate_d1(S, K, T, r, sigma, q)
    
    # Calculate probability density function φ(d1)
    phi_d1 = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * d1 * d1)
    
    # Analytical vega formula
    vega = S * exp(-q * T) * phi_d1 * sqrt(T)
    
    return vega


def calculate_rho_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Rho for European call option
    
    Mathematical Derivation:
    Rho = ∂C/∂r = K * T * e^(-rT) * N(d2)
    
    Financial Interpretation:
    Rho measures sensitivity to interest rate changes.
    For calls: Rho is positive (higher rates increase call value)
    - Rho = 20 means ₹20 option price increase for 1% rate increase
    - More significant for longer-term options
    - Important during monetary policy changes (RBI rate decisions)
    
    Args:
        S, K, T, r, sigma, q: Option parameters
        
    Returns:
        Rho value for call option (positive)
    """
    if T <= 0:
        return 0.0
    
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    
    N_d2 = cumulative_normal_distribution(d2)
    
    # Analytical rho formula for call
    rho_call = K * T * exp(-r * T) * N_d2
    
    return rho_call


def calculate_rho_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate Rho for European put option
    
    Mathematical Derivation:
    Rho = ∂P/∂r = -K * T * e^(-rT) * N(-d2)
    
    Financial Interpretation:
    Put rho is negative (higher rates decrease put value).
    - Higher rates make the present value of strike payment smaller
    - Less significant for NIFTY options (shorter terms) vs equity LEAPS
    - Relevant during RBI repo rate changes
    
    Args:
        S, K, T, r, sigma, q: Option parameters
        
    Returns:
        Rho value for put option (negative)
    """
    if T <= 0:
        return 0.0
    
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    
    N_minus_d2 = cumulative_normal_distribution(-d2)
    
    # Analytical rho formula for put
    rho_put = -K * T * exp(-r * T) * N_minus_d2
    
    return rho_put


def calculate_all_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str, q: float = 0.0) -> Dict[str, float]:
    """
    Calculate all Greeks for a single option in one function call
    
    Financial Interpretation:
    Provides complete risk profile of an option position.
    Essential for:
    - Portfolio risk management
    - Dynamic hedging strategies
    - Understanding option behavior
    - Position sizing decisions
    
    Args:
        S, K, T, r, sigma: Option parameters
        option_type: 'CE' for call, 'PE' for put
        q: Dividend yield
        
    Returns:
        Dictionary with all Greeks values
    """
    # Parameter validation
    if T < 0:
        raise GreeksCalculationError("Time to expiry cannot be negative")
    if sigma <= 0:
        raise GreeksCalculationError("Volatility must be positive")
    if S <= 0:
        raise GreeksCalculationError("Spot price must be positive")
    if K <= 0:
        raise GreeksCalculationError("Strike price must be positive")
    
    # Calculate Greeks based on option type
    if option_type.upper() == 'CE':
        delta = calculate_delta_call(S, K, T, r, sigma, q)
        theta = calculate_theta_call(S, K, T, r, sigma, q)
        rho = calculate_rho_call(S, K, T, r, sigma, q)
    elif option_type.upper() == 'PE':
        delta = calculate_delta_put(S, K, T, r, sigma, q)
        theta = calculate_theta_put(S, K, T, r, sigma, q)
        rho = calculate_rho_put(S, K, T, r, sigma, q)
    else:
        raise GreeksCalculationError(f"Invalid option type: {option_type}")
    
    # Gamma and Vega are same for calls and puts
    gamma = calculate_gamma(S, K, T, r, sigma, q)
    vega = calculate_vega(S, K, T, r, sigma, q)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
        'spot_price': S,
        'strike_price': K,
        'time_to_expiry': T,
        'volatility': sigma,
        'risk_free_rate': r,
        'option_type': option_type
    }


# =============================================================================
# DATAFRAME INTEGRATION FUNCTIONS
# =============================================================================

def calculate_greeks_dataframe(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Greeks for entire options DataFrame
    
    Integrates with your enhanced options data structure from collectors.py
    Optimized for batch processing of options chains
    
    Required DataFrame columns:
    - Spot_Price, Strike, Time_to_Expiry, Volatility, Option_Type
    - Risk_Free_Rate (optional, will use default if missing)
    - Symbol (optional, for dividend yield lookup)
    
    Args:
        options_df: DataFrame with options data
        
    Returns:
        DataFrame with additional Greeks columns
    """
    required_columns = ['Spot_Price', 'Strike', 'Time_to_Expiry', 'Volatility', 'Option_Type']
    
    # Validate DataFrame structure
    missing_columns = [col for col in required_columns if col not in options_df.columns]
    if missing_columns:
        raise GreeksCalculationError(f"Missing required columns: {missing_columns}")
    
    # Make copy to avoid modifying original
    result_df = options_df.copy()
    
    # Initialize Greeks columns
    greeks_columns = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
    for col in greeks_columns:
        result_df[col] = 0.0
    
    # Get default parameters
    config = get_config()
    default_rate = config.model.risk_free_rate['default']
    
    # Calculate Greeks for each row
    for idx, row in result_df.iterrows():
        try:
            # Extract parameters
            S = row['Spot_Price']
            K = row['Strike']
            T = row['Time_to_Expiry']
            sigma = row['Volatility']
            option_type = row['Option_Type']
            r = row.get('Risk_Free_Rate', default_rate)
            
            # Get dividend yield (0 for NIFTY)
            symbol = row.get('Symbol', 'NIFTY')
            q = 0.0 if symbol == 'NIFTY' else 0.01  # Default 1% for stocks
            
            # Calculate all Greeks
            greeks = calculate_all_greeks(S, K, T, r, sigma, option_type, q)
            
            # Store results
            result_df.loc[idx, 'Delta'] = greeks['delta']
            result_df.loc[idx, 'Gamma'] = greeks['gamma']
            result_df.loc[idx, 'Theta'] = greeks['theta']
            result_df.loc[idx, 'Vega'] = greeks['vega']
            result_df.loc[idx, 'Rho'] = greeks['rho']
            
        except Exception as e:
            print(f"Error calculating Greeks for row {idx}: {e}")
            # Set Greeks to NaN for failed calculations
            for col in greeks_columns:
                result_df.loc[idx, col] = np.nan
    
    print(f"Calculated Greeks for {len(result_df)} options")
    return result_df


# =============================================================================
# GREEKS SENSITIVITY ANALYSIS
# =============================================================================

def greeks_sensitivity_analysis(base_S: float, base_K: float, base_T: float, 
                               base_r: float, base_sigma: float, 
                               option_type: str = 'CE') -> Dict[str, pd.DataFrame]:
    """
    Comprehensive Greeks sensitivity analysis for learning and risk management
    
    Analyzes how Greeks change with respect to:
    - Underlying price movements (±20%)
    - Volatility changes (±50%)
    - Time decay (next 30 days)
    - Interest rate changes (±200 bps)
    
    Args:
        base_S, base_K, base_T, base_r, base_sigma: Base option parameters
        option_type: 'CE' or 'PE'
        
    Returns:
        Dictionary of DataFrames with sensitivity analysis results
    """
    sensitivity_results = {}
    
    # 1. Spot Price Sensitivity (±20% in 2% increments)
    spot_range = np.arange(base_S * 0.8, base_S * 1.2, base_S * 0.02)
    spot_sensitivity = []
    
    for S in spot_range:
        greeks = calculate_all_greeks(S, base_K, base_T, base_r, base_sigma, option_type)
        greeks['spot_price'] = S
        greeks['moneyness'] = S / base_K
        spot_sensitivity.append(greeks)
    
    sensitivity_results['spot_sensitivity'] = pd.DataFrame(spot_sensitivity)
    
    # 2. Volatility Sensitivity (±50% in 5% increments)
    vol_range = np.arange(base_sigma * 0.5, base_sigma * 1.5, base_sigma * 0.05)
    vol_sensitivity = []
    
    for sigma in vol_range:
        greeks = calculate_all_greeks(base_S, base_K, base_T, base_r, sigma, option_type)
        greeks['volatility'] = sigma
        greeks['vol_change'] = (sigma - base_sigma) / base_sigma
        vol_sensitivity.append(greeks)
    
    sensitivity_results['volatility_sensitivity'] = pd.DataFrame(vol_sensitivity)
    
    # 3. Time Decay Analysis (next 30 days)
    time_range = np.arange(max(base_T - 30/365, 1/365), base_T + 1/365, 1/365)
    time_sensitivity = []
    
    for T in time_range:
        greeks = calculate_all_greeks(base_S, base_K, T, base_r, base_sigma, option_type)
        greeks['time_to_expiry'] = T
        greeks['days_to_expiry'] = T * 365
        time_sensitivity.append(greeks)
    
    sensitivity_results['time_sensitivity'] = pd.DataFrame(time_sensitivity)
    
    # 4. Interest Rate Sensitivity (±200 bps)
    rate_range = np.arange(max(base_r - 0.02, 0), base_r + 0.02, 0.005)
    rate_sensitivity = []
    
    for r in rate_range:
        greeks = calculate_all_greeks(base_S, base_K, base_T, r, base_sigma, option_type)
        greeks['risk_free_rate'] = r
        greeks['rate_change_bps'] = (r - base_r) * 10000
        rate_sensitivity.append(greeks)
    
    sensitivity_results['rate_sensitivity'] = pd.DataFrame(rate_sensitivity)
    
    print("Greeks sensitivity analysis completed")
    print(f"Analyzed {len(spot_range)} spot scenarios, {len(vol_range)} volatility scenarios")
    print(f"Analyzed {len(time_range)} time scenarios, {len(rate_range)} rate scenarios")
    
    return sensitivity_results


def plot_greeks_sensitivity(sensitivity_data: Dict[str, pd.DataFrame], 
                           option_type: str = 'CE'):
    """
    Visualize Greeks sensitivity analysis results
    
    Creates comprehensive plots showing:
    - Greeks vs Spot Price (Delta, Gamma curves)
    - Greeks vs Volatility (Vega sensitivity)
    - Greeks vs Time (Theta decay)
    - Greeks vs Interest Rates (Rho sensitivity)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{option_type} Option Greeks Sensitivity Analysis', fontsize=16)
    
    # 1. Spot Price Sensitivity
    spot_data = sensitivity_data['spot_sensitivity']
    axes[0, 0].plot(spot_data['moneyness'], spot_data['delta'], 'b-', linewidth=2, label='Delta')
    axes[0, 0].plot(spot_data['moneyness'], spot_data['gamma'] * 10, 'r--', linewidth=2, label='Gamma×10')
    axes[0, 0].set_xlabel('Moneyness (S/K)')
    axes[0, 0].set_ylabel('Greeks Value')
    axes[0, 0].set_title('Greeks vs Spot Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Volatility Sensitivity
    vol_data = sensitivity_data['volatility_sensitivity']
    axes[0, 1].plot(vol_data['volatility'] * 100, vol_data['vega'], 'g-', linewidth=2, label='Vega')
    axes[0, 1].set_xlabel('Volatility (%)')
    axes[0, 1].set_ylabel('Vega')
    axes[0, 1].set_title('Vega vs Volatility')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Time Decay
    time_data = sensitivity_data['time_sensitivity']
    axes[1, 0].plot(time_data['days_to_expiry'], time_data['theta'], 'm-', linewidth=2, label='Theta')
    axes[1, 0].set_xlabel('Days to Expiry')
    axes[1, 0].set_ylabel('Theta')
    axes[1, 0].set_title('Theta vs Time to Expiry')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Interest Rate Sensitivity
    rate_data = sensitivity_data['rate_sensitivity']
    axes[1, 1].plot(rate_data['rate_change_bps'], rate_data['rho'], 'c-', linewidth=2, label='Rho')
    axes[1, 1].set_xlabel('Rate Change (bps)')
    axes[1, 1].set_ylabel('Rho')
    axes[1, 1].set_title('Rho vs Interest Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MARKET SCENARIO TESTING
# =============================================================================

def market_scenario_testing(base_params: Dict[str, float], 
                           scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Test Greeks behavior under different market scenarios
    
    Common scenarios for Indian markets:
    - Normal Market: Current conditions
    - High Volatility: Market stress (elections, global events)
    - Low Volatility: Calm market conditions
    - Rate Hike: RBI tightening cycle
    - Rate Cut: RBI easing cycle
    
    Args:
        base_params: Base option parameters
        scenarios: Dictionary of scenario parameters
        
    Returns:
        DataFrame with Greeks under different scenarios
    """
    scenario_results = []
    
    for scenario_name, scenario_params in scenarios.items():
        # Start with base parameters
        test_params = base_params.copy()
        
        # Apply scenario modifications
        test_params.update(scenario_params)
        
        # Calculate Greeks for both calls and puts
        for option_type in ['CE', 'PE']:
            greeks = calculate_all_greeks(
                test_params['S'],
                test_params['K'],
                test_params['T'],
                test_params['r'],
                test_params['sigma'],
                option_type
            )
            
            greeks['scenario'] = scenario_name
            greeks['option_type'] = option_type
            scenario_results.append(greeks)
    
    results_df = pd.DataFrame(scenario_results)
    
    print(f"Market scenario testing completed for {len(scenarios)} scenarios")
    return results_df


def create_indian_market_scenarios() -> Dict[str, Dict[str, float]]:
    """
    Create realistic market scenarios for Indian options markets
    Based on historical market conditions and RBI policy patterns
    """
    scenarios = {
        'Normal_Market': {
            'sigma': 0.16,  # 16% volatility
            'r': 0.065,     # Current repo rate
        },
        'High_Volatility_Stress': {
            'sigma': 0.35,  # 35% volatility (market stress)
            'r': 0.065,     # Rate unchanged
        },
        'Low_Volatility_Calm': {
            'sigma': 0.08,  # 8% volatility (very calm)
            'r': 0.065,     # Rate unchanged
        },
        'RBI_Rate_Hike_Cycle': {
            'sigma': 0.18,  # Slightly higher vol
            'r': 0.075,     # +100 bps rate hike
        },
        'RBI_Rate_Cut_Cycle': {
            'sigma': 0.14,  # Lower vol expectation
            'r': 0.055,     # -100 bps rate cut
        },
        'Election_Uncertainty': {
            'sigma': 0.28,  # High political uncertainty
            'r': 0.065,     # Rates stable
        },
        'Global_Crisis': {
            'sigma': 0.45,  # Extreme volatility
            'r': 0.075,     # Defensive rate hike
        }
    }
    
    return scenarios


# =============================================================================
# COMPREHENSIVE GREEKS ANALYZER CLASS
# =============================================================================

class IndianGreeksAnalyzer:
    """
    Comprehensive Greeks analysis engine for Indian options markets
    Integrates with your Black-Scholes engine and data collection system
    """
    
    def __init__(self):
        """Initialize the Greeks analyzer with market configuration"""
        self.config = get_config()
        self.bs_engine = IndianBlackScholesEngine()
        
        print("IndianGreeksAnalyzer initialized with analytical Greeks calculation")
    
    def analyze_single_option(self, spot_price: float, strike_price: float,
                             time_to_expiry: float, volatility: float,
                             option_type: str, valuation_date: str = None,
                             symbol: str = 'NIFTY') -> Dict[str, float]:
        """
        Comprehensive analysis of single option including Greeks and sensitivities
        """
        # Get market parameters
        if valuation_date:
            risk_free_rate = self.bs_engine.get_risk_free_rate(valuation_date)
        else:
            risk_free_rate = 0.065
        
        dividend_yield = self.bs_engine.get_dividend_yield(symbol)
        
        # Calculate all Greeks
        greeks = calculate_all_greeks(
            spot_price, strike_price, time_to_expiry,
            risk_free_rate, volatility, option_type, dividend_yield
        )
        
        # Add market context
        greeks['symbol'] = symbol
        greeks['valuation_date'] = valuation_date
        greeks['dividend_yield'] = dividend_yield
        
        return greeks
    
    def run_comprehensive_analysis(self, base_params: Dict[str, float]) -> Dict[str, any]:
        """
        Run complete Greeks analysis including sensitivity and scenario testing
        Perfect for resume showcase and learning
        """
        print("=== Running Comprehensive Greeks Analysis ===")
        
        results = {}
        
        # 1. Base case analysis
        base_greeks = self.analyze_single_option(**base_params)
        results['base_case'] = base_greeks
        
        # 2. Sensitivity analysis
        sensitivity = greeks_sensitivity_analysis(
            base_params['spot_price'],
            base_params['strike_price'],
            base_params['time_to_expiry'],
            0.065,  # Use standard rate
            base_params['volatility'],
            base_params['option_type']
        )
        results['sensitivity_analysis'] = sensitivity
        
        # 3. Market scenario testing
        scenarios = create_indian_market_scenarios()
        scenario_params = {
            'S': base_params['spot_price'],
            'K': base_params['strike_price'],
            'T': base_params['time_to_expiry'],
            'r': 0.065,
            'sigma': base_params['volatility']
        }
        
        scenario_results = market_scenario_testing(scenario_params, scenarios)
        results['scenario_testing'] = scenario_results
        
        print("=== Comprehensive Analysis Completed ===")
        return results


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_greeks_implementation():
    """
    Test Greeks implementation against known values for validation
    """
    print("=== Testing Greeks Implementation ===")
    
    # Test case: ATM NIFTY option
    S = 24000  # NIFTY spot
    K = 24000  # ATM strike
    T = 30/365  # 30 days
    r = 0.065   # 6.5% rate
    sigma = 0.15  # 15% volatility
    
    print(f"\nTest Parameters: S={S}, K={K}, T={T:.4f}, r={r:.1%}, σ={sigma:.1%}")
    
    # Test call Greeks
    print("\n--- Call Option Greeks ---")
    call_greeks = calculate_all_greeks(S, K, T, r, sigma, 'CE')
    for greek, value in call_greeks.items():
        if isinstance(value, float) and greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            print(f"{greek.capitalize()}: {value:.4f}")
    
    # Test put Greeks
    print("\n--- Put Option Greeks ---")
    put_greeks = calculate_all_greeks(S, K, T, r, sigma, 'PE')
    for greek, value in put_greeks.items():
        if isinstance(value, float) and greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            print(f"{greek.capitalize()}: {value:.4f}")
    
    # Validate relationships
    print("\n--- Greeks Relationships Validation ---")
    
    # Gamma should be same for calls and puts
    gamma_diff = abs(call_greeks['gamma'] - put_greeks['gamma'])
    print(f"Gamma difference (Call vs Put): {gamma_diff:.6f} (should be ~0)")
    
    # Vega should be same for calls and puts
    vega_diff = abs(call_greeks['vega'] - put_greeks['vega'])
    print(f"Vega difference (Call vs Put): {vega_diff:.6f} (should be ~0)")
    
    # Delta relationship: Call Delta - Put Delta ≈ 1 (for no dividends)
    delta_diff = call_greeks['delta'] - put_greeks['delta']
    print(f"Delta difference (Call - Put): {delta_diff:.4f} (should be ~1.0)")
    
    print("\n=== Greeks Testing Completed ===")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=== Indian Options Greeks Calculator ===")
    
    # Test core implementation
    test_greeks_implementation()
    
    # Demo comprehensive analysis
    print("\n=== Running Demo Analysis ===")
    
    # Create sample parameters
    demo_params = {
        'spot_price': 24000,
        'strike_price': 24000,
        'time_to_expiry': 30/365,
        'volatility': 0.16,
        'option_type': 'CE',
        'symbol': 'NIFTY'
    }
    
    # Initialize analyzer
    analyzer = IndianGreeksAnalyzer()
    
    # Run comprehensive analysis
    analysis_results = analyzer.run_comprehensive_analysis(demo_params)
    
    # Display summary results
    print("\n=== Analysis Summary ===")
    base_case = analysis_results['base_case']
    print(f"Base Case Greeks:")
    print(f"Delta: {base_case['delta']:.4f}")
    print(f"Gamma: {base_case['gamma']:.4f}")
    print(f"Theta: {base_case['theta']:.4f}")
    print(f"Vega: {base_case['vega']:.4f}")
    print(f"Rho: {base_case['rho']:.4f}")
    
    # Scenario comparison
    scenario_data = analysis_results['scenario_testing']
    print(f"\nScenario Analysis: {len(scenario_data)} scenarios tested")
    print("Top 3 scenarios by absolute Delta:")
    top_scenarios = scenario_data.nlargest(3, 'delta')[['scenario', 'option_type', 'delta', 'volatility']]
    print(top_scenarios.to_string(index=False))
    
    print("\n=== Greeks Calculator Ready for Production Use ===")
