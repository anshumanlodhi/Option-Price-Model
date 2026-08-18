
"""
Options Pricing Module for Indian Markets
Simple implementation using Black-Scholes model
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import math
from typing import Dict, List, Tuple, Optional

class BlackScholesCalculator:
    """
    A simple Black-Scholes calculator for educational purposes
    We'll implement the basic formulas before using external libraries
    """

    def __init__(self, spot_price: float, strike_price: float, 
                 time_to_expiry: float, risk_free_rate: float, 
                 volatility: float, dividend_yield: float = 0.0):
        """
        Initialize the Black-Scholes calculator

        Parameters:
        -----------
        spot_price : float
            Current price of the underlying asset
        strike_price : float
            Strike price of the option
        time_to_expiry : float
            Time to expiry in years (e.g., 30 days = 30/365)
        risk_free_rate : float
            Risk-free rate as decimal (e.g., 0.065 for 6.5%)
        volatility : float
            Volatility as decimal (e.g., 0.20 for 20%)
        dividend_yield : float, optional
            Dividend yield as decimal (default: 0.0)
        """
        self.S = spot_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        self.q = dividend_yield

        # Calculate d1 and d2 (key Black-Scholes parameters)
        self._calculate_d_parameters()

    def _calculate_d_parameters(self):
        """
        Calculate d1 and d2 parameters for Black-Scholes formula

        Theory:
        d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        d2 = d1 - σ√T
        """
        if self.T <= 0:
            raise ValueError("Time to expiry must be positive")
        
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
            
        if self.S <= 0 or self.K <= 0:
            raise ValueError("Spot price and strike price must be positive")

        sqrt_T = math.sqrt(self.T)

        self.d1 = (math.log(self.S / self.K) + 
                   (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt_T)

        self.d2 = self.d1 - self.sigma * sqrt_T

    def call_price(self) -> float:
        """
        Calculate European call option price

        Formula: C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
        """
        return (self.S * math.exp(-self.q * self.T) * norm.cdf(self.d1) - 
                self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2))

    def put_price(self) -> float:
        """
        Calculate European put option price

        Formula: P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
        """
        return (self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2) - 
                self.S * math.exp(-self.q * self.T) * norm.cdf(-self.d1))

    def delta(self, option_type: str = 'call') -> float:
        """
        Calculate Delta (price sensitivity to underlying price)

        Theory: Delta measures how much the option price changes 
        for a ₹1 change in the underlying price
        """
        if option_type.lower() == 'call':
            return math.exp(-self.q * self.T) * norm.cdf(self.d1)
        else:  # put
            return math.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)

    def gamma(self) -> float:
        """
        Calculate Gamma (rate of change of Delta)

        Theory: Gamma measures how much Delta changes
        for a ₹1 change in the underlying price
        """
        sqrt_T = math.sqrt(self.T)
        return (math.exp(-self.q * self.T) * norm.pdf(self.d1)) / (self.S * self.sigma * sqrt_T)

    def theta(self, option_type: str = 'call') -> float:
        """
        Calculate Theta (time decay)

        Theory: Theta measures how much the option price decreases
        as time passes (usually expressed per day)
        """
        if self.T <= 0:
            return 0.0
            
        sqrt_T = math.sqrt(self.T)

        # Common terms
        term1 = -(self.S * math.exp(-self.q * self.T) * norm.pdf(self.d1) * self.sigma) / (2 * sqrt_T)
        
        if option_type.lower() == 'call':
            term2 = -self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(self.d1)
            term3 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2)
            return term1 + term2 + term3
        else:  # put
            term2 = -self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(-self.d1)
            term3 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return term1 + term2 - term3

    def vega(self) -> float:
        """
        Calculate Vega (sensitivity to volatility)

        Theory: Vega measures how much the option price changes
        for a 1% change in volatility
        """
        sqrt_T = math.sqrt(self.T)
        return self.S * math.exp(-self.q * self.T) * norm.pdf(self.d1) * sqrt_T

    def rho(self, option_type: str = 'call') -> float:
        """
        Calculate Rho (sensitivity to interest rate)

        Theory: Rho measures how much the option price changes
        for a 1% change in interest rates
        """
        if option_type.lower() == 'call':
            return self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:  # put
            return -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-self.d2)

    def all_greeks(self, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for the option

        Returns:
        --------
        dict : Dictionary containing all Greeks
        """
        return {
            'price': self.call_price() if option_type.lower() == 'call' else self.put_price(),
            'delta': self.delta(option_type),
            'gamma': self.gamma(),
            'theta': self.theta(option_type),
            'vega': self.vega(),
            'rho': self.rho(option_type)
        }
    
    def validate_put_call_parity(self, tolerance: float = 1e-4) -> bool:
        """
        Validate put-call parity relationship
        
        Formula: C - P = S*e^(-qT) - K*e^(-rT)
        
        Args:
            tolerance: Tolerance for validation
            
        Returns:
            True if put-call parity holds within tolerance
        """
        call_price = self.call_price()
        put_price = self.put_price()
        
        left_side = call_price - put_price
        right_side = (self.S * math.exp(-self.q * self.T) - 
                     self.K * math.exp(-self.r * self.T))
        
        return abs(left_side - right_side) < tolerance

# Example usage and testing
def test_black_scholes():
    """
    Test the Black-Scholes calculator with sample Indian market data
    """
    # Example: NIFTY option
    # Spot: 24,000, Strike: 24,000, 30 days to expiry, 6.5% risk-free rate, 15% volatility

    calculator = BlackScholesCalculator(
        spot_price=24000,
        strike_price=24000,
        time_to_expiry=30/365,  # 30 days in years
        risk_free_rate=0.065,   # 6.5% RBI repo rate
        volatility=0.15         # 15% volatility
    )

    print("Black-Scholes Calculator Test")
    print("=" * 50)
    print(f"Spot Price: ₹{calculator.S:,.2f}")
    print(f"Strike Price: ₹{calculator.K:,.2f}")
    print(f"Time to Expiry: {calculator.T*365:.0f} days")
    print(f"Risk-Free Rate: {calculator.r:.1%}")
    print(f"Volatility: {calculator.sigma:.1%}")
    print()

    # Calculate call option metrics
    call_greeks = calculator.all_greeks('call')
    print("CALL OPTION:")
    print(f"  Price: ₹{call_greeks['price']:.2f}")
    print(f"  Delta: {call_greeks['delta']:.4f}")
    print(f"  Gamma: {call_greeks['gamma']:.6f}")
    print(f"  Theta: ₹{call_greeks['theta']:.2f}")
    print(f"  Vega: ₹{call_greeks['vega']:.2f}")
    print(f"  Rho: ₹{call_greeks['rho']:.2f}")
    print()

    # Calculate put option metrics
    put_greeks = calculator.all_greeks('put')
    print("PUT OPTION:")
    print(f"  Price: ₹{put_greeks['price']:.2f}")
    print(f"  Delta: {put_greeks['delta']:.4f}")
    print(f"  Gamma: {put_greeks['gamma']:.6f}")
    print(f"  Theta: ₹{put_greeks['theta']:.2f}")
    print(f"  Vega: ₹{put_greeks['vega']:.2f}")
    print(f"  Rho: ₹{put_greeks['rho']:.2f}")

if __name__ == "__main__":
    test_black_scholes()
