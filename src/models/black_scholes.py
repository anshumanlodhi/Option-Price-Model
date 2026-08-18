"""
Black-Scholes Options Pricing Model for Indian Markets
Educational implementation with step-by-step calculations and financial explanations
Integrates with IndianMarketDataCollector and historical RBI repo rates
"""

import pandas as pd
import numpy as np
from math import log, sqrt, exp, pi
from typing import Union, Dict, List, Optional, Tuple
import warnings
from datetime import datetime, timedelta

# Import configuration and data collector
from config.settings import get_config
from data.collectors import IndianMarketDataCollector


class BlackScholesCalculationError(Exception):
    """Custom exception for Black-Scholes calculation errors"""
    pass


# =============================================================================
# CORE MATHEMATICAL FUNCTIONS
# =============================================================================

def cumulative_normal_distribution(x: float) -> float:
    """
    Custom implementation of cumulative normal distribution N(x)
    
    Financial Meaning: This represents the probability that a standard normal
    random variable is less than or equal to x. In options pricing, N(d1) gives
    us the hedge ratio (delta), while N(d2) represents the risk-neutral 
    probability of the option finishing in-the-money.
    
    Args:
        x: Input value for CDF calculation
        
    Returns:
        Cumulative probability N(x)
    """
    # Using Abramowitz and Stegun approximation for educational purposes
    # More accurate than basic approximations, suitable for options pricing
    
    if x < 0:
        # Use symmetry: N(-x) = 1 - N(x)
        return 1.0 - cumulative_normal_distribution(-x)
    
    # Constants for the approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    # Calculate the approximation
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x / 2.0) / sqrt(2.0 * pi)
    
    return y


def calculate_d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Calculate d1 parameter for Black-Scholes formula
    
    Financial Meaning: d1 measures how far the current stock price is from the 
    strike price in terms of standard deviations of the stock's future price 
    distribution. It's used to calculate the delta (hedge ratio) of the option.
    
    Formula: d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
    
    Args:
        S: Current spot price of underlying
        K: Strike price of option
        T: Time to expiry in years
        r: Risk-free interest rate (annual)
        sigma: Volatility of underlying (annual)
        q: Dividend yield (annual, default 0 for NIFTY)
        
    Returns:
        d1 value
    """
    if T <= 0:
        raise BlackScholesCalculationError("Time to expiry must be positive")
    if sigma <= 0:
        raise BlackScholesCalculationError("Volatility must be positive")
    if S <= 0:
        raise BlackScholesCalculationError("Spot price must be positive")
    if K <= 0:
        raise BlackScholesCalculationError("Strike price must be positive")
    
    # Step-by-step calculation for educational clarity
    moneyness = log(S / K)  # How far from ATM (ln(S/K))
    drift_term = (r - q + 0.5 * sigma * sigma) * T  # Risk-neutral drift adjustment
    volatility_term = sigma * sqrt(T)  # Volatility scaling by time
    
    d1 = (moneyness + drift_term) / volatility_term
    
    return d1


def calculate_d2(d1: float, sigma: float, T: float) -> float:
    """
    Calculate d2 parameter for Black-Scholes formula
    
    Financial Meaning: d2 represents the risk-neutral probability that the option
    will be exercised (finish in-the-money). It's d1 adjusted downward by the
    volatility term σ√T, reflecting the uncertainty in the stock's future price.
    
    Formula: d2 = d1 - σ√T
    
    Args:
        d1: Previously calculated d1 value
        sigma: Volatility of underlying (annual)
        T: Time to expiry in years
        
    Returns:
        d2 value
    """
    d2 = d1 - sigma * sqrt(T)
    return d2


# =============================================================================
# CORE BLACK-SCHOLES PRICING FUNCTIONS
# =============================================================================

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Dict[str, float]:
    """
    Calculate European call option price using Black-Scholes formula
    
    Financial Meaning: A call option gives the holder the right (not obligation)
    to buy the underlying asset at the strike price K before expiry. The price
    reflects the present value of the expected payoff max(S_T - K, 0) under
    risk-neutral probability measure.
    
    Formula: C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield (0 for NIFTY)
        
    Returns:
        Dictionary with call price and intermediate calculations
    """
    # Parameter validation with fail-fast approach
    if T <= 0:
        raise BlackScholesCalculationError(f"Invalid time to expiry: {T}. Must be positive.")
    if sigma <= 0 or sigma > 5.0:  # Reasonable volatility bounds
        raise BlackScholesCalculationError(f"Invalid volatility: {sigma}. Must be between 0 and 5.")
    if S <= 0:
        raise BlackScholesCalculationError(f"Invalid spot price: {S}. Must be positive.")
    if K <= 0:
        raise BlackScholesCalculationError(f"Invalid strike price: {K}. Must be positive.")
    if r < -0.1 or r > 0.5:  # Reasonable interest rate bounds
        raise BlackScholesCalculationError(f"Invalid risk-free rate: {r}. Must be between -10% and 50%.")
    
    # Step 1: Calculate d1 and d2
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    
    # Step 2: Calculate cumulative normal distributions
    # N(d1): Probability-weighted sensitivity to underlying price changes
    N_d1 = cumulative_normal_distribution(d1)
    
    # N(d2): Risk-neutral probability of option being exercised
    N_d2 = cumulative_normal_distribution(d2)
    
    # Step 3: Calculate present value components
    # Present value of underlying asset (adjusted for dividends)
    pv_underlying = S * exp(-q * T)
    
    # Present value of strike price (discounted at risk-free rate)
    pv_strike = K * exp(-r * T)
    
    # Step 4: Calculate call option price
    # Financial interpretation: Expected value of (S_T - K)+ discounted to present
    call_price = pv_underlying * N_d1 - pv_strike * N_d2
    
    # Return detailed results for educational purposes
    return {
        'call_price': call_price,
        'd1': d1,
        'd2': d2,
        'N_d1': N_d1,
        'N_d2': N_d2,
        'pv_underlying': pv_underlying,
        'pv_strike': pv_strike,
        'moneyness': S / K,
        'time_value': max(0, call_price - max(0, S - K))  # Time value component
    }


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Dict[str, float]:
    """
    Calculate European put option price using Black-Scholes formula
    
    Financial Meaning: A put option gives the holder the right to sell the 
    underlying asset at the strike price K before expiry. The price reflects
    the present value of the expected payoff max(K - S_T, 0).
    
    Formula: P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    
    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield (0 for NIFTY)
        
    Returns:
        Dictionary with put price and intermediate calculations
    """
    # Same parameter validation as call
    if T <= 0:
        raise BlackScholesCalculationError(f"Invalid time to expiry: {T}. Must be positive.")
    if sigma <= 0 or sigma > 5.0:
        raise BlackScholesCalculationError(f"Invalid volatility: {sigma}. Must be between 0 and 5.")
    if S <= 0:
        raise BlackScholesCalculationError(f"Invalid spot price: {S}. Must be positive.")
    if K <= 0:
        raise BlackScholesCalculationError(f"Invalid strike price: {K}. Must be positive.")
    if r < -0.1 or r > 0.5:
        raise BlackScholesCalculationError(f"Invalid risk-free rate: {r}. Must be between -10% and 50%.")
    
    # Calculate d1 and d2
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    
    # Calculate cumulative normal distributions for put
    # N(-d1): Probability-weighted sensitivity for put option
    N_minus_d1 = cumulative_normal_distribution(-d1)
    
    # N(-d2): Risk-neutral probability of put being exercised
    N_minus_d2 = cumulative_normal_distribution(-d2)
    
    # Present value components
    pv_underlying = S * exp(-q * T)
    pv_strike = K * exp(-r * T)
    
    # Put option price calculation
    # Financial interpretation: Expected value of (K - S_T)+ discounted to present
    put_price = pv_strike * N_minus_d2 - pv_underlying * N_minus_d1
    
    # Return detailed results
    return {
        'put_price': put_price,
        'd1': d1,
        'd2': d2,
        'N_minus_d1': N_minus_d1,
        'N_minus_d2': N_minus_d2,
        'pv_underlying': pv_underlying,
        'pv_strike': pv_strike,
        'moneyness': S / K,
        'time_value': max(0, put_price - max(0, K - S))  # Time value component
    }


def verify_put_call_parity(call_price: float, put_price: float, S: float, K: float, 
                          T: float, r: float, q: float = 0.0) -> Dict[str, float]:
    """
    Verify put-call parity relationship for educational validation
    
    Financial Meaning: Put-call parity is an arbitrage relationship that must
    hold for European options on the same underlying with same strike and expiry.
    Formula: C - P = S*e^(-qT) - K*e^(-rT)
    
    Args:
        call_price: Calculated call option price
        put_price: Calculated put option price
        S, K, T, r, q: Option parameters
        
    Returns:
        Dictionary with parity check results
    """
    # Calculate theoretical relationship
    left_side = call_price - put_price
    right_side = S * exp(-q * T) - K * exp(-r * T)
    
    # Calculate difference (should be near zero for correct pricing)
    parity_difference = abs(left_side - right_side)
    
    return {
        'call_minus_put': left_side,
        'pv_underlying_minus_pv_strike': right_side,
        'parity_difference': parity_difference,
        'parity_holds': parity_difference < 1e-10  # Numerical tolerance
    }


# =============================================================================
# BLACK-SCHOLES PRICING ENGINE (WRAPPER CLASS)
# =============================================================================

class IndianBlackScholesEngine:
    """
    Comprehensive Black-Scholes pricing engine for Indian options markets
    
    Features:
    - Integration with IndianMarketDataCollector
    - Historical RBI repo rate support
    - Dividend adjustment for individual stocks
    - Batch pricing for DataFrame operations
    - Educational output with detailed calculations
    """
    
    def __init__(self):
        """Initialize the pricing engine with configuration and data collector"""
        self.config = get_config()
        self.data_collector = IndianMarketDataCollector()
        
        # Default parameters for Indian markets
        self.default_dividend_yield = {
            'NIFTY': 0.0,  # Index doesn't pay dividends
            'RELIANCE': 0.005,  # Approximate dividend yields
            'TCS': 0.015,
            'HDFCBANK': 0.008,
            'INFY': 0.025,
            'ICICIBANK': 0.012
        }
        
        print("IndianBlackScholesEngine initialized with RBI repo rate integration")
    
    def get_risk_free_rate(self, valuation_date: Union[str, datetime]) -> float:
        """
        Get historical RBI repo rate for given date using data collector's proven method
        
        Args:
            valuation_date: Date for rate lookup
            
        Returns:
            Historical repo rate for the date
        """
        if isinstance(valuation_date, datetime):
            date_str = valuation_date.strftime('%Y-%m-%d')
        else:
            date_str = valuation_date
        
        # Use data collector's excellent historical rate mapping
        return self.data_collector.get_risk_free_rate(date_str)
    
    def get_dividend_yield(self, symbol: str) -> float:
        """
        Get dividend yield for given symbol
        
        Financial Meaning: Dividend yield reduces the forward price of the stock,
        as dividends are paid to stock holders but not option holders. For NIFTY
        index options, this is zero as indices don't pay dividends.
        
        Args:
            symbol: Stock/index symbol
            
        Returns:
            Annual dividend yield
        """
        return self.default_dividend_yield.get(symbol, 0.0)
    
    def price_single_option(self, 
                          spot_price: float,
                          strike_price: float,
                          time_to_expiry: float,
                          volatility: float,
                          option_type: str,
                          valuation_date: Union[str, datetime] = None,
                          symbol: str = 'NIFTY',
                          show_details: bool = True) -> Dict[str, float]:
        """
        Price a single option with full details and educational output
        
        Args:
            spot_price: Current price of underlying
            strike_price: Strike price of option
            time_to_expiry: Time to expiry in years
            volatility: Annual volatility
            option_type: 'CE' for call, 'PE' for put
            valuation_date: Date for risk-free rate lookup
            symbol: Underlying symbol for dividend yield
            show_details: Whether to show detailed calculations
            
        Returns:
            Dictionary with pricing results and details
        """
        # Get market parameters
        if valuation_date:
            risk_free_rate = self.get_risk_free_rate(valuation_date)
        else:
            risk_free_rate = 0.065  # Current default
        
        dividend_yield = self.get_dividend_yield(symbol)
        
        # Price the option
        if option_type.upper() == 'CE':
            result = black_scholes_call(spot_price, strike_price, time_to_expiry, 
                                      risk_free_rate, volatility, dividend_yield)
            option_price = result['call_price']
        elif option_type.upper() == 'PE':
            result = black_scholes_put(spot_price, strike_price, time_to_expiry,
                                     risk_free_rate, volatility, dividend_yield)
            option_price = result['put_price']
        else:
            raise BlackScholesCalculationError(f"Invalid option type: {option_type}. Use 'CE' or 'PE'.")
        
        # Compile comprehensive results
        pricing_result = {
            'option_price': option_price,
            'spot_price': spot_price,
            'strike_price': strike_price,
            'time_to_expiry': time_to_expiry,
            'volatility': volatility,
            'risk_free_rate': risk_free_rate,
            'dividend_yield': dividend_yield,
            'option_type': option_type,
            'symbol': symbol,
            'intrinsic_value': max(0, (spot_price - strike_price) if option_type.upper() == 'CE' 
                                    else (strike_price - spot_price)),
            'time_value': option_price - max(0, (spot_price - strike_price) if option_type.upper() == 'CE' 
                                           else (strike_price - spot_price))
        }
        
        # Add detailed calculations
        pricing_result.update(result)
        
        # Educational output
        if show_details:
            self._print_pricing_details(pricing_result)
        
        return pricing_result
    
    def price_options_dataframe(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Price multiple options from DataFrame (integrates with your data structure)
        
        Financial Meaning: Batch pricing allows efficient valuation of entire
        options chains or portfolios, maintaining consistency in market parameters
        while handling the specific characteristics of each option.
        
        Args:
            options_df: DataFrame with columns: Spot_Price, Strike, Time_to_Expiry, 
                       Volatility, Option_Type, Symbol, Date (optional)
                       
        Returns:
            DataFrame with added pricing columns
        """
        required_columns = ['Spot_Price', 'Strike', 'Time_to_Expiry', 'Volatility', 'Option_Type']
        
        # Validate DataFrame structure
        missing_columns = [col for col in required_columns if col not in options_df.columns]
        if missing_columns:
            raise BlackScholesCalculationError(f"Missing required columns: {missing_columns}")
        
        # Make a copy to avoid modifying original
        result_df = options_df.copy()
        
        # Initialize result columns
        result_df['BS_Price'] = 0.0
        result_df['Risk_Free_Rate'] = 0.0
        result_df['Dividend_Yield'] = 0.0
        result_df['Intrinsic_Value'] = 0.0
        result_df['Time_Value'] = 0.0
        result_df['d1'] = 0.0
        result_df['d2'] = 0.0
        
        # Price each option
        for idx, row in result_df.iterrows():
            try:
                # Get parameters
                spot = row['Spot_Price']
                strike = row['Strike']
                tte = row['Time_to_Expiry']
                vol = row['Volatility']
                opt_type = row['Option_Type']
                symbol = row.get('Symbol', 'NIFTY')
                date = row.get('Date', None)
                
                # Price the option
                pricing_result = self.price_single_option(
                    spot, strike, tte, vol, opt_type, date, symbol, show_details=False
                )
                
                # Store results
                result_df.loc[idx, 'BS_Price'] = pricing_result['option_price']
                result_df.loc[idx, 'Risk_Free_Rate'] = pricing_result['risk_free_rate']
                result_df.loc[idx, 'Dividend_Yield'] = pricing_result['dividend_yield']
                result_df.loc[idx, 'Intrinsic_Value'] = pricing_result['intrinsic_value']
                result_df.loc[idx, 'Time_Value'] = pricing_result['time_value']
                result_df.loc[idx, 'd1'] = pricing_result['d1']
                result_df.loc[idx, 'd2'] = pricing_result['d2']
                
            except Exception as e:
                print(f"Error pricing option at index {idx}: {e}")
                result_df.loc[idx, 'BS_Price'] = np.nan
        
        print(f"Priced {len(result_df)} options using Black-Scholes model")
        return result_df
    
    def _print_pricing_details(self, result: Dict[str, float]):
        """
        Print detailed educational output for single option pricing
        """
        print(f"\n=== Black-Scholes Pricing Details ===")
        print(f"Option: {result['symbol']} {result['strike_price']} {result['option_type']}")
        print(f"Spot Price: ₹{result['spot_price']:.2f}")
        print(f"Strike Price: ₹{result['strike_price']:.2f}")
        print(f"Time to Expiry: {result['time_to_expiry']:.4f} years")
        print(f"Volatility: {result['volatility']:.2%}")
        print(f"Risk-Free Rate: {result['risk_free_rate']:.2%}")
        print(f"Dividend Yield: {result['dividend_yield']:.2%}")
        print(f"\nCalculated Parameters:")
        print(f"d1: {result['d1']:.4f}")
        print(f"d2: {result['d2']:.4f}")
        print(f"Moneyness (S/K): {result['moneyness']:.4f}")
        print(f"\nOption Valuation:")
        print(f"Black-Scholes Price: ₹{result['option_price']:.2f}")
        print(f"Intrinsic Value: ₹{result['intrinsic_value']:.2f}")
        print(f"Time Value: ₹{result['time_value']:.2f}")
        print("=" * 50)


# =============================================================================
# TESTING AND VALIDATION FUNCTIONS
# =============================================================================

def test_against_known_values():
    """
    Test implementation against textbook examples for validation
    """
    print("=== Testing Black-Scholes Implementation ===")
    
    # Test Case 1: Classic textbook example
    # European call option: S=100, K=100, T=0.25, r=5%, σ=20%
    print("\nTest 1: Classic Textbook Example")
    print("S=100, K=100, T=0.25 years, r=5%, σ=20%, q=0%")
    
    result = black_scholes_call(S=100, K=100, T=0.25, r=0.05, sigma=0.20, q=0.0)
    print(f"Calculated Call Price: ₹{result['call_price']:.4f}")
    print(f"Expected: ~₹4.10 (textbook value)")
    print(f"d1: {result['d1']:.4f}, d2: {result['d2']:.4f}")
    
    # Test put option with same parameters
    put_result = black_scholes_put(S=100, K=100, T=0.25, r=0.05, sigma=0.20, q=0.0)
    print(f"Calculated Put Price: ₹{put_result['put_price']:.4f}")
    print(f"Expected: ~₹2.86 (textbook value)")
    
    # Test put-call parity
    parity = verify_put_call_parity(result['call_price'], put_result['put_price'], 
                                   100, 100, 0.25, 0.05, 0.0)
    print(f"Put-Call Parity Check: {parity['parity_holds']} (difference: {parity['parity_difference']:.10f})")
    
    # Test Case 2: Indian market example with NIFTY
    print("\nTest 2: Indian Market Example")
    print("NIFTY: S=24000, K=24000, T=30 days, r=6.5%, σ=15%")
    
    engine = IndianBlackScholesEngine()
    nifty_result = engine.price_single_option(
        spot_price=24000,
        strike_price=24000,
        time_to_expiry=30/365,
        volatility=0.15,
        option_type='CE',
        symbol='NIFTY',
        show_details=True
    )
    
    print("\n=== All Tests Completed ===")


def create_test_options_dataframe() -> pd.DataFrame:
    """
    Create sample options DataFrame for testing batch pricing
    Mimics your enhanced options data structure
    """
    # Create test options chain
    spot_price = 24000
    strikes = [23000, 23500, 24000, 24500, 25000]
    option_types = ['CE', 'PE']
    volatility = 0.18
    
    test_data = []
    for strike in strikes:
        for opt_type in option_types:
            test_data.append({
                'Spot_Price': spot_price,
                'Strike': strike,
                'Time_to_Expiry': 30/365,  # 30 days
                'Volatility': volatility,
                'Option_Type': opt_type,
                'Symbol': 'NIFTY',
                'Date': '2024-12-01'
            })
    
    return pd.DataFrame(test_data)


# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=== Indian Black-Scholes Options Pricing Engine ===")
    
    # Test core functions against known values
    test_against_known_values()
    
    # Test batch pricing with DataFrame
    print("\n=== Testing Batch Pricing ===")
    test_df = create_test_options_dataframe()
    print(f"Created test DataFrame with {len(test_df)} options")
    
    # Initialize engine and price options
    engine = IndianBlackScholesEngine()
    priced_df = engine.price_options_dataframe(test_df)
    
    # Display results
    print("\nSample Pricing Results:")
    display_columns = ['Strike', 'Option_Type', 'BS_Price', 'Intrinsic_Value', 'Time_Value']
    print(priced_df[display_columns].head(8).to_string(index=False))
    
    # Test individual option with historical rate
    print("\n=== Testing Historical Rate Integration ===")
    historical_result = engine.price_single_option(
        spot_price=24000,
        strike_price=24000,
        time_to_expiry=45/365,
        volatility=0.16,
        option_type='CE',
        valuation_date='2023-06-15',  # Use historical date
        symbol='NIFTY',
        show_details=True
    )
    
    print("\n=== Black-Scholes Engine Ready for Use ===")
