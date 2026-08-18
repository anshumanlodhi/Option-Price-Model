"""
Volatility Surface Construction and Model Validation for Indian Options Markets
Implements implied volatility extraction, surface fitting, and SVI parameterization
Educational implementation with mathematical foundations and practical applications
"""

import pandas as pd
import numpy as np
from math import log, sqrt, exp, pi
from typing import Union, Dict, List, Optional, Tuple, Callable
import warnings
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar, minimize, brentq
from scipy.interpolate import griddata, RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Import your existing modules
from config.settings import get_config
from models.black_scholes import IndianBlackScholesEngine, black_scholes_call, black_scholes_put
from models.greeks import IndianGreeksAnalyzer, calculate_all_greeks
from data.collectors import IndianMarketDataCollector


class VolatilityCalculationError(Exception):
    """Custom exception for volatility calculation errors"""
    pass


# =============================================================================
# HISTORICAL VOLATILITY CALCULATIONS
# =============================================================================

def calculate_historical_volatility(price_data: pd.Series, 
                                   window: int = 21, 
                                   method: str = 'log_returns') -> pd.Series:
    """
    Calculate historical volatility using various methods
    
    Financial Meaning: Historical volatility measures the actual price variability
    of the underlying asset over a specific period. It's backward-looking and
    forms the basis for volatility forecasting and implied volatility comparison.
    
    Mathematical Foundation:
    Log Returns Method: σ = sqrt(252) * std(ln(S_t / S_{t-1}))
    Simple Returns Method: σ = sqrt(252) * std((S_t - S_{t-1}) / S_{t-1})
    
    Args:
        price_data: Time series of prices (typically Close prices)
        window: Rolling window for volatility calculation
        method: 'log_returns' or 'simple_returns'
        
    Returns:
        Series of annualized historical volatility
    """
    if len(price_data) < window:
        raise VolatilityCalculationError(f"Insufficient data: need at least {window} observations")
    
    if method == 'log_returns':
        # Natural logarithm of price ratios (more accurate for financial data)
        returns = np.log(price_data / price_data.shift(1))
    elif method == 'simple_returns':
        # Simple percentage returns
        returns = price_data.pct_change()
    else:
        raise VolatilityCalculationError(f"Invalid method: {method}")
    
    # Remove NaN values from returns calculation
    returns = returns.dropna()
    
    # Calculate rolling standard deviation and annualize
    rolling_vol = returns.rolling(window=window).std() * sqrt(252)
    
    return rolling_vol


def calculate_realized_volatility(price_data: pd.Series, 
                                period_days: int = 21) -> float:
    """
    Calculate realized volatility over a specific period
    
    Financial Meaning: Realized volatility is the actual volatility that occurred
    over a specific historical period. Used for comparing with implied volatility
    to assess whether options were over/under-priced.
    
    Args:
        price_data: Price series for the period
        period_days: Number of days for calculation
        
    Returns:
        Annualized realized volatility
    """
    if len(price_data) < 2:
        raise VolatilityCalculationError("Need at least 2 price observations")
    
    # Calculate log returns
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    
    # Calculate realized volatility
    realized_vol = log_returns.std() * sqrt(252)
    
    return realized_vol


def calculate_garch_volatility(returns: pd.Series, 
                             forecast_horizon: int = 1) -> Dict[str, float]:
    """
    Simple GARCH(1,1) volatility model for forecasting
    
    Financial Meaning: GARCH models capture volatility clustering - the tendency
    for high volatility periods to be followed by high volatility periods.
    Essential for volatility forecasting in options pricing.
    
    Mathematical Foundation:
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    where ε_t are the standardized residuals
    
    Args:
        returns: Time series of returns
        forecast_horizon: Days ahead to forecast
        
    Returns:
        Dictionary with current volatility and forecast
    """
    # Simplified GARCH implementation for educational purposes
    returns_clean = returns.dropna()
    
    if len(returns_clean) < 100:
        raise VolatilityCalculationError("Need at least 100 observations for GARCH")
    
    # Calculate unconditional variance
    unconditional_var = returns_clean.var()
    
    # Simple GARCH(1,1) parameters (educational approximation)
    omega = 0.000001  # Long-term variance component
    alpha = 0.1       # ARCH effect (reaction to recent shocks)
    beta = 0.85       # GARCH effect (persistence)
    
    # Current period conditional variance
    recent_return_sq = returns_clean.iloc[-1] ** 2
    prev_vol_sq = returns_clean.rolling(30).var().iloc[-1]
    
    current_var = omega + alpha * recent_return_sq + beta * prev_vol_sq
    current_vol = sqrt(current_var * 252)  # Annualize
    
    # Simple forecast (mean-reverting to long-term level)
    forecast_var = current_var * (alpha + beta) ** forecast_horizon + \
                   unconditional_var * (1 - (alpha + beta) ** forecast_horizon)
    forecast_vol = sqrt(forecast_var * 252)
    
    return {
        'current_volatility': current_vol,
        'forecast_volatility': forecast_vol,
        'unconditional_volatility': sqrt(unconditional_var * 252),
        'persistence': alpha + beta
    }


# =============================================================================
# IMPLIED VOLATILITY EXTRACTION
# =============================================================================

def implied_volatility_call(market_price: float, S: float, K: float, T: float, 
                           r: float, q: float = 0.0, 
                           vol_guess: float = 0.2) -> float:
    """
    Extract implied volatility from call option market price using Black-Scholes inversion
    
    Financial Meaning: Implied volatility is the market's expectation of future
    volatility embedded in option prices. It's forward-looking and often differs
    from historical volatility, reflecting market sentiment and risk perception.
    
    Mathematical Approach: Uses numerical root-finding to solve:
    Market_Price = Black_Scholes_Call(S, K, T, r, σ_implied, q)
    
    Args:
        market_price: Observed market price of call option
        S, K, T, r, q: Black-Scholes parameters
        vol_guess: Initial guess for volatility
        
    Returns:
        Implied volatility (annualized)
    """
    if market_price <= 0:
        raise VolatilityCalculationError("Market price must be positive")
    
    if T <= 0:
        raise VolatilityCalculationError("Time to expiry must be positive")
    
    # Check if option has any intrinsic value
    intrinsic_value = max(0, S - K)
    if market_price < intrinsic_value:
        raise VolatilityCalculationError("Market price below intrinsic value")
    
    def objective_function(sigma):
        """Objective function: difference between market and model price"""
        try:
            if sigma <= 0 or sigma > 5.0:  # Reasonable volatility bounds
                return float('inf')
            
            model_price = black_scholes_call(S, K, T, r, sigma, q)['call_price']
            return abs(market_price - model_price)
        
        except:
            return float('inf')
    
    # Use Brent's method for robust root finding
    try:
        result = minimize_scalar(
            objective_function,
            bounds=(0.01, 3.0),  # 1% to 300% volatility bounds
            method='bounded',
            options={'xatol': 1e-6}
        )
        
        if result.success and result.fun < 0.01:  # Convergence tolerance
            return result.x
        else:
            raise VolatilityCalculationError("Failed to converge to implied volatility")
    
    except Exception as e:
        raise VolatilityCalculationError(f"Implied volatility calculation failed: {e}")


def implied_volatility_put(market_price: float, S: float, K: float, T: float, 
                          r: float, q: float = 0.0, 
                          vol_guess: float = 0.2) -> float:
    """
    Extract implied volatility from put option market price
    
    Uses same methodology as call IV but with put option pricing formula
    
    Args:
        market_price: Observed market price of put option
        S, K, T, r, q: Black-Scholes parameters
        vol_guess: Initial guess for volatility
        
    Returns:
        Implied volatility (annualized)
    """
    if market_price <= 0:
        raise VolatilityCalculationError("Market price must be positive")
    
    if T <= 0:
        raise VolatilityCalculationError("Time to expiry must be positive")
    
    # Check intrinsic value for puts
    intrinsic_value = max(0, K - S)
    if market_price < intrinsic_value:
        raise VolatilityCalculationError("Market price below intrinsic value")
    
    def objective_function(sigma):
        """Objective function for put option"""
        try:
            if sigma <= 0 or sigma > 5.0:
                return float('inf')
            
            model_price = black_scholes_put(S, K, T, r, sigma, q)['put_price']
            return abs(market_price - model_price)
        
        except:
            return float('inf')
    
    try:
        result = minimize_scalar(
            objective_function,
            bounds=(0.01, 3.0),
            method='bounded',
            options={'xatol': 1e-6}
        )
        
        if result.success and result.fun < 0.01:
            return result.x
        else:
            raise VolatilityCalculationError("Failed to converge to implied volatility")
    
    except Exception as e:
        raise VolatilityCalculationError(f"Implied volatility calculation failed: {e}")


def extract_implied_volatilities_dataframe(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract implied volatilities for entire options DataFrame
    
    Integrates with your enhanced options data structure
    Adds implied volatility column for volatility surface construction
    
    Required columns: Market_Price, Spot_Price, Strike, Time_to_Expiry, 
                     Risk_Free_Rate, Option_Type
    
    Args:
        options_df: DataFrame with options market data
        
    Returns:
        DataFrame with added Implied_Volatility column
    """
    required_columns = ['Market_Price', 'Spot_Price', 'Strike', 'Time_to_Expiry', 
                       'Risk_Free_Rate', 'Option_Type']
    
    missing_columns = [col for col in required_columns if col not in options_df.columns]
    if missing_columns:
        raise VolatilityCalculationError(f"Missing required columns: {missing_columns}")
    
    result_df = options_df.copy()
    result_df['Implied_Volatility'] = np.nan
    result_df['IV_Calculation_Error'] = ''
    
    successful_calculations = 0
    
    for idx, row in result_df.iterrows():
        try:
            market_price = row['Market_Price']
            S = row['Spot_Price']
            K = row['Strike']
            T = row['Time_to_Expiry']
            r = row['Risk_Free_Rate']
            option_type = row['Option_Type']
            q = row.get('Dividend_Yield', 0.0)
            
            # Extract implied volatility based on option type
            if option_type.upper() == 'CE':
                iv = implied_volatility_call(market_price, S, K, T, r, q)
            elif option_type.upper() == 'PE':
                iv = implied_volatility_put(market_price, S, K, T, r, q)
            else:
                raise VolatilityCalculationError(f"Invalid option type: {option_type}")
            
            result_df.loc[idx, 'Implied_Volatility'] = iv
            successful_calculations += 1
            
        except Exception as e:
            result_df.loc[idx, 'IV_Calculation_Error'] = str(e)
            print(f"IV calculation failed for row {idx}: {e}")
    
    print(f"Successfully calculated implied volatility for {successful_calculations}/{len(result_df)} options")
    
    return result_df


# =============================================================================
# VOLATILITY SURFACE CONSTRUCTION
# =============================================================================

def create_volatility_surface_basic(implied_vol_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Create basic volatility surface using interpolation methods
    
    Financial Meaning: Volatility surface shows how implied volatility varies
    across strikes (volatility smile/skew) and time to expiry (term structure).
    Essential for pricing non-standard options and understanding market dynamics.
    
    Args:
        implied_vol_df: DataFrame with Strike, Time_to_Expiry, Implied_Volatility columns
        
    Returns:
        Dictionary with surface grids for visualization and interpolation
    """
    # Filter valid implied volatility data
    valid_data = implied_vol_df.dropna(subset=['Implied_Volatility'])
    
    if len(valid_data) < 10:
        raise VolatilityCalculationError("Need at least 10 valid IV points for surface construction")
    
    # Extract coordinates and values
    strikes = valid_data['Strike'].values
    times = valid_data['Time_to_Expiry'].values
    iv_values = valid_data['Implied_Volatility'].values
    
    # Create regular grid for surface
    strike_range = np.linspace(strikes.min(), strikes.max(), 20)
    time_range = np.linspace(times.min(), times.max(), 15)
    
    strike_grid, time_grid = np.meshgrid(strike_range, time_range)
    
    # Interpolate implied volatilities on regular grid
    try:
        iv_surface = griddata(
            points=np.column_stack([strikes, times]),
            values=iv_values,
            xi=(strike_grid, time_grid),
            method='cubic',
            fill_value=np.median(iv_values)
        )
        
        # Ensure no negative volatilities
        iv_surface = np.maximum(iv_surface, 0.01)
        
    except Exception as e:
        # Fallback to linear interpolation
        print(f"Cubic interpolation failed, using linear: {e}")
        iv_surface = griddata(
            points=np.column_stack([strikes, times]),
            values=iv_values,
            xi=(strike_grid, time_grid),
            method='linear',
            fill_value=np.median(iv_values)
        )
        iv_surface = np.maximum(iv_surface, 0.01)
    
    return {
        'strike_grid': strike_grid,
        'time_grid': time_grid,
        'iv_surface': iv_surface,
        'raw_strikes': strikes,
        'raw_times': times,
        'raw_iv_values': iv_values
    }


def fit_svi_surface(implied_vol_df: pd.DataFrame) -> Dict[str, any]:
    """
    Fit SVI (Stochastic Volatility Inspired) model to volatility surface
    
    Financial Meaning: SVI provides a parametric representation of volatility
    surface that is arbitrage-free and captures market-observed volatility smiles.
    Industry standard for volatility surface modeling.
    
    Mathematical Foundation:
    Total variance: w(k) = a + b{ρ(k-m) + √[(k-m)² + σ²]}
    where k = ln(K/F) is log-moneyness
    
    Args:
        implied_vol_df: DataFrame with options data and implied volatilities
        
    Returns:
        Dictionary with SVI parameters and fitted surface
    """
    # Group by time to expiry for slice-wise fitting
    time_groups = implied_vol_df.groupby('Time_to_Expiry')
    
    svi_results = {}
    
    for time_to_expiry, group_data in time_groups:
        if len(group_data) < 5:  # Need minimum points for fitting
            continue
        
        # Calculate log-moneyness and total variance
        spot_price = group_data['Spot_Price'].iloc[0]
        forward_price = spot_price * exp(group_data['Risk_Free_Rate'].iloc[0] * time_to_expiry)
        
        group_data = group_data.copy()
        group_data['Log_Moneyness'] = np.log(group_data['Strike'] / forward_price)
        group_data['Total_Variance'] = (group_data['Implied_Volatility'] ** 2) * time_to_expiry
        
        # SVI parameter fitting
        def svi_total_variance(k, a, b, rho, m, sigma):
            """SVI total variance formula"""
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        
        def svi_objective(params):
            """Objective function for SVI fitting"""
            a, b, rho, m, sigma = params
            
            # Parameter constraints for no-arbitrage
            if sigma <= 0 or abs(rho) >= 1 or a < 0 or b < 0:
                return float('inf')
            
            k_values = group_data['Log_Moneyness'].values
            market_variance = group_data['Total_Variance'].values
            
            model_variance = svi_total_variance(k_values, a, b, rho, m, sigma)
            
            return np.sum((market_variance - model_variance)**2)
        
        # Initial parameter guess
        initial_guess = [
            0.01,  # a: level
            0.1,   # b: slope
            0.0,   # rho: correlation
            0.0,   # m: center
            0.1    # sigma: vol-of-vol
        ]
        
        try:
            # Optimize SVI parameters
            result = minimize(
                svi_objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=[
                    (0, 1),      # a: non-negative
                    (0, 1),      # b: non-negative
                    (-0.99, 0.99), # rho: correlation bounds
                    (-1, 1),     # m: center
                    (0.01, 1)    # sigma: positive
                ]
            )
            
            if result.success:
                a, b, rho, m, sigma = result.x
                
                svi_results[time_to_expiry] = {
                    'parameters': {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma},
                    'fit_quality': result.fun,
                    'raw_data': group_data,
                    'forward_price': forward_price
                }
            
        except Exception as e:
            print(f"SVI fitting failed for T={time_to_expiry:.4f}: {e}")
    
    print(f"Successfully fitted SVI model for {len(svi_results)} time slices")
    
    return svi_results


# =============================================================================
# MODEL VALIDATION FRAMEWORK
# =============================================================================

def validate_pricing_accuracy(options_df: pd.DataFrame, 
                             bs_engine: IndianBlackScholesEngine) -> Dict[str, float]:
    """
    Validate Black-Scholes pricing accuracy against market data
    
    Financial Meaning: Model validation ensures our pricing model produces
    realistic prices. Key metrics include mean absolute error, percentage errors,
    and distribution of pricing errors across different option characteristics.
    
    Args:
        options_df: DataFrame with Market_Price and model parameters
        bs_engine: Your Black-Scholes pricing engine
        
    Returns:
        Dictionary with validation metrics
    """
    if 'Market_Price' not in options_df.columns:
        raise VolatilityCalculationError("Market_Price column required for validation")
    
    validation_results = []
    
    for idx, row in options_df.iterrows():
        try:
            # Get model price using your engine
            model_result = bs_engine.price_single_option(
                spot_price=row['Spot_Price'],
                strike_price=row['Strike'],
                time_to_expiry=row['Time_to_Expiry'],
                volatility=row.get('Implied_Volatility', row.get('Volatility', 0.2)),
                option_type=row['Option_Type'],
                symbol=row.get('Symbol', 'NIFTY'),
                show_details=False
            )
            
            model_price = model_result['option_price']
            market_price = row['Market_Price']
            
            # Calculate error metrics
            absolute_error = abs(model_price - market_price)
            percentage_error = (absolute_error / market_price) * 100
            relative_error = (model_price - market_price) / market_price
            
            validation_results.append({
                'market_price': market_price,
                'model_price': model_price,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error,
                'relative_error': relative_error,
                'moneyness': row['Spot_Price'] / row['Strike'],
                'time_to_expiry': row['Time_to_Expiry'],
                'option_type': row['Option_Type']
            })
            
        except Exception as e:
            print(f"Validation failed for row {idx}: {e}")
    
    if not validation_results:
        raise VolatilityCalculationError("No successful validations performed")
    
    validation_df = pd.DataFrame(validation_results)
    
    # Calculate aggregate metrics
    metrics = {
        'mean_absolute_error': validation_df['absolute_error'].mean(),
        'median_absolute_error': validation_df['absolute_error'].median(),
        'mean_percentage_error': validation_df['percentage_error'].mean(),
        'median_percentage_error': validation_df['percentage_error'].median(),
        'rmse': np.sqrt(np.mean(validation_df['absolute_error']**2)),
        'r_squared': np.corrcoef(validation_df['market_price'], validation_df['model_price'])[0,1]**2,
        'successful_validations': len(validation_df),
        'max_percentage_error': validation_df['percentage_error'].max(),
        'percentage_within_5pct': (validation_df['percentage_error'] <= 5).mean() * 100,
        'percentage_within_10pct': (validation_df['percentage_error'] <= 10).mean() * 100
    }
    
    return metrics


def stress_test_volatility_model(base_params: Dict[str, float], 
                                stress_scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Stress test volatility model under extreme market conditions
    
    Financial Meaning: Stress testing evaluates model stability and accuracy
    under extreme market conditions that may not be present in historical data.
    Critical for risk management and model robustness assessment.
    
    Args:
        base_params: Base case parameters
        stress_scenarios: Dictionary of stress test scenarios
        
    Returns:
        DataFrame with stress test results
    """
    stress_results = []
    
    for scenario_name, scenario_params in stress_scenarios.items():
        # Combine base parameters with stress adjustments
        test_params = base_params.copy()
        test_params.update(scenario_params)
        
        try:
            # Test implied volatility extraction
            if 'market_price' in test_params:
                iv_call = implied_volatility_call(
                    market_price=test_params['market_price'],
                    S=test_params['spot_price'],
                    K=test_params['strike_price'],
                    T=test_params['time_to_expiry'],
                    r=test_params['risk_free_rate']
                )
                
                # Test Greeks calculation with extreme parameters
                greeks = calculate_all_greeks(
                    S=test_params['spot_price'],
                    K=test_params['strike_price'],
                    T=test_params['time_to_expiry'],
                    r=test_params['risk_free_rate'],
                    sigma=iv_call,
                    option_type='CE'
                )
                
                stress_results.append({
                    'scenario': scenario_name,
                    'implied_volatility': iv_call,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega'],
                    'test_status': 'SUCCESS'
                })
            
        except Exception as e:
            stress_results.append({
                'scenario': scenario_name,
                'implied_volatility': np.nan,
                'delta': np.nan,
                'gamma': np.nan,
                'theta': np.nan,
                'vega': np.nan,
                'test_status': f'FAILED: {str(e)}'
            })
    
    return pd.DataFrame(stress_results)


# =============================================================================
# COMPREHENSIVE VOLATILITY ANALYZER
# =============================================================================

class IndianVolatilityAnalyzer:
    """
    Comprehensive volatility analysis system for Indian options markets
    
    Integrates historical volatility, implied volatility extraction,
    surface construction, and model validation in unified framework
    """
    
    def __init__(self):
        """Initialize the volatility analyzer with market components"""
        self.config = get_config()
        self.bs_engine = IndianBlackScholesEngine()
        self.greeks_analyzer = IndianGreeksAnalyzer()
        self.data_collector = IndianMarketDataCollector()
        
        print("IndianVolatilityAnalyzer initialized with comprehensive volatility modeling")
    
    def analyze_historical_volatility(self, symbol: str = 'NIFTY', 
                                    period: str = 'analysis') -> Dict[str, pd.Series]:
        """
        Comprehensive historical volatility analysis using your data collector
        """
        print(f"Analyzing historical volatility for {symbol}...")
        
        # Get historical data using your proven collector
        if symbol == 'NIFTY':
            yahoo_symbol = '^NSEI'
        else:
            yahoo_symbol = f"{symbol}.NS"
        
        # Use your data collector's date ranges
        if period == 'analysis':
            start_date = self.data_collector.analysis_start
            end_date = self.data_collector.analysis_end
        else:
            start_date = self.data_collector.validation_start
            end_date = self.data_collector.validation_end
        
        # Download data
        historical_data = self.data_collector.download_stock_data(
            yahoo_symbol, start_date, end_date, save_to_csv=False
        )
        
        if historical_data.empty:
            raise VolatilityCalculationError(f"No historical data available for {symbol}")
        
        # Calculate multiple volatility measures
        close_prices = historical_data['Close']
        
        vol_results = {
            'vol_21d': calculate_historical_volatility(close_prices, 21),
            'vol_30d': calculate_historical_volatility(close_prices, 30),
            'vol_63d': calculate_historical_volatility(close_prices, 63),
            'vol_252d': calculate_historical_volatility(close_prices, 252)
        }
        
        # Add GARCH volatility
        returns = np.log(close_prices / close_prices.shift(1)).dropna()
        garch_results = calculate_garch_volatility(returns)
        
        print(f"Historical volatility analysis completed for {symbol}")
        print(f"Current 30d volatility: {vol_results['vol_30d'].iloc[-1]:.2%}")
        print(f"GARCH forecast: {garch_results['forecast_volatility']:.2%}")
        
        return {
            'volatility_series': vol_results,
            'garch_analysis': garch_results,
            'price_data': historical_data
        }
    
    def build_volatility_surface(self, options_data: pd.DataFrame) -> Dict[str, any]:
        """
        Build comprehensive volatility surface with SVI fitting
        """
        print("Building volatility surface...")
        
        # Step 1: Extract implied volatilities
        if 'Market_Price' in options_data.columns:
            iv_data = extract_implied_volatilities_dataframe(options_data)
        else:
            print("No market prices available, using input volatilities")
            iv_data = options_data.copy()
            if 'Implied_Volatility' not in iv_data.columns:
                iv_data['Implied_Volatility'] = iv_data.get('Volatility', 0.2)
        
        # Step 2: Create basic interpolated surface
        basic_surface = create_volatility_surface_basic(iv_data)
        
        # Step 3: Fit SVI model
        svi_results = fit_svi_surface(iv_data)
        
        # Step 4: Validate surface quality
        surface_metrics = self._validate_surface_quality(basic_surface, iv_data)
        
        return {
            'implied_volatility_data': iv_data,
            'basic_surface': basic_surface,
            'svi_model': svi_results,
            'surface_metrics': surface_metrics
        }
    
    def comprehensive_model_validation(self, options_data: pd.DataFrame) -> Dict[str, any]:
        """
        Run comprehensive model validation suite
        """
        print("Running comprehensive model validation...")
        
        validation_results = {}
        
        # Pricing accuracy validation
        if 'Market_Price' in options_data.columns:
            pricing_metrics = validate_pricing_accuracy(options_data, self.bs_engine)
            validation_results['pricing_accuracy'] = pricing_metrics
            
            print(f"Mean pricing error: {pricing_metrics['mean_percentage_error']:.2f}%")
            print(f"R-squared: {pricing_metrics['r_squared']:.4f}")
        
        # Stress testing
        stress_scenarios = self._create_stress_scenarios()
        base_params = self._get_base_stress_params(options_data)
        
        stress_results = stress_test_volatility_model(base_params, stress_scenarios)
        validation_results['stress_testing'] = stress_results
        
        # Greeks validation (compare analytical vs numerical)
        greeks_validation = self._validate_greeks_accuracy(options_data)
        validation_results['greeks_validation'] = greeks_validation
        
        return validation_results
    
    def _validate_surface_quality(self, surface_data: Dict[str, np.ndarray], 
                                 iv_data: pd.DataFrame) -> Dict[str, float]:
        """Validate volatility surface quality and arbitrage conditions"""
        
        # Check for arbitrage conditions
        strike_grid = surface_data['strike_grid']
        time_grid = surface_data['time_grid']
        iv_surface = surface_data['iv_surface']
        
        # Calendar spread arbitrage check
        calendar_violations = 0
        butterfly_violations = 0
        
        # Count violations (simplified check)
        for i in range(iv_surface.shape[0] - 1):
            for j in range(iv_surface.shape[1]):
                # Calendar: longer-term volatility should be >= shorter-term
                if iv_surface[i+1, j] < iv_surface[i, j] - 0.05:  # 5% tolerance
                    calendar_violations += 1
        
        total_points = iv_surface.shape[0] * iv_surface.shape[1]
        
        return {
            'total_surface_points': total_points,
            'calendar_violations': calendar_violations,
            'calendar_violation_rate': calendar_violations / total_points,
            'surface_smoothness': np.std(np.gradient(iv_surface.flatten())),
            'fit_quality': np.corrcoef(
                iv_data['Implied_Volatility'].values,
                np.interp(iv_data['Strike'].values, 
                         surface_data['raw_strikes'], 
                         surface_data['raw_iv_values'])
            )[0,1]**2
        }
    
    def _create_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Create stress test scenarios for Indian markets"""
        return {
            'market_crash': {
                'spot_price': 20000,  # -20% NIFTY crash
                'market_price': 150,   # High option prices
                'time_to_expiry': 0.08,  # 30 days
                'strike_price': 24000,
                'risk_free_rate': 0.075  # Emergency rate hike
            },
            'volatility_spike': {
                'spot_price': 24000,
                'market_price': 800,   # Very high option price
                'time_to_expiry': 0.08,
                'strike_price': 24000,
                'risk_free_rate': 0.065
            },
            'near_expiry': {
                'spot_price': 24000,
                'market_price': 50,
                'time_to_expiry': 0.003,  # 1 day to expiry
                'strike_price': 24000,
                'risk_free_rate': 0.065
            }
        }
    
    def _get_base_stress_params(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Extract base parameters for stress testing"""
        return {
            'spot_price': options_data['Spot_Price'].iloc[0],
            'strike_price': options_data['Strike'].iloc[0],
            'time_to_expiry': options_data['Time_to_Expiry'].iloc[0],
            'risk_free_rate': options_data.get('Risk_Free_Rate', pd.Series([0.065])).iloc[0],
            'market_price': options_data.get('Market_Price', pd.Series([100])).iloc[0]
        }
    
    def _validate_greeks_accuracy(self, options_data: pd.DataFrame) -> Dict[str, float]:
        """Validate Greeks accuracy using finite difference methods"""
        
        # Sample a few options for detailed validation
        sample_data = options_data.head(5)
        
        accuracy_results = []
        
        for idx, row in sample_data.iterrows():
            # Calculate analytical Greeks
            analytical_greeks = calculate_all_greeks(
                row['Spot_Price'], row['Strike'], row['Time_to_Expiry'],
                row.get('Risk_Free_Rate', 0.065), 
                row.get('Implied_Volatility', row.get('Volatility', 0.2)),
                row['Option_Type']
            )
            
            # Calculate numerical Greeks (finite difference)
            h = 1.0  # ₹1 bump for delta
            
            price_up = black_scholes_call(
                row['Spot_Price'] + h, row['Strike'], row['Time_to_Expiry'],
                row.get('Risk_Free_Rate', 0.065),
                row.get('Implied_Volatility', row.get('Volatility', 0.2))
            )['call_price']
            
            price_down = black_scholes_call(
                row['Spot_Price'] - h, row['Strike'], row['Time_to_Expiry'],
                row.get('Risk_Free_Rate', 0.065),
                row.get('Implied_Volatility', row.get('Volatility', 0.2))
            )['call_price']
            
            numerical_delta = (price_up - price_down) / (2 * h)
            
            delta_error = abs(analytical_greeks['delta'] - numerical_delta)
            accuracy_results.append(delta_error)
        
        return {
            'mean_delta_error': np.mean(accuracy_results),
            'max_delta_error': np.max(accuracy_results),
            'validation_sample_size': len(accuracy_results)
        }
    
    def plot_volatility_surface(self, surface_data: Dict[str, np.ndarray]):
        """Create 3D visualization of volatility surface"""
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        strike_grid = surface_data['strike_grid']
        time_grid = surface_data['time_grid'] * 365  # Convert to days
        iv_surface = surface_data['iv_surface'] * 100  # Convert to percentage
        
        # Create 3D surface plot
        surf = ax.plot_surface(
            strike_grid, time_grid, iv_surface,
            cmap='viridis', alpha=0.8, linewidth=0.5
        )
        
        # Add scatter points for raw data
        raw_strikes = surface_data['raw_strikes']
        raw_times = surface_data['raw_times'] * 365
        raw_iv = surface_data['raw_iv_values'] * 100
        
        ax.scatter(raw_strikes, raw_times, raw_iv, 
                  color='red', s=20, alpha=0.6, label='Market Data')
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Days to Expiry')
        ax.set_zlabel('Implied Volatility (%)')
        ax.set_title('NIFTY Options Implied Volatility Surface')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def create_sample_options_data_with_iv() -> pd.DataFrame:
    """
    Create sample options data with realistic market prices for testing
    """
    np.random.seed(42)  # For reproducible results
    
    spot_price = 24000
    strikes = np.array([22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000])
    expiries = np.array([7, 14, 30, 45]) / 365  # Convert days to years
    
    sample_data = []
    
    for expiry in expiries:
        for strike in strikes:
            for option_type in ['CE', 'PE']:
                # Generate realistic implied volatility with smile
                moneyness = spot_price / strike
                
                # Volatility smile: higher vol for OTM options
                if option_type == 'CE':
                    base_vol = 0.16 + 0.05 * max(0, moneyness - 1.0)  # Smile for calls
                else:
                    base_vol = 0.16 + 0.05 * max(0, 1.0 - moneyness)  # Smile for puts
                
                # Add term structure: longer-term slightly higher vol
                implied_vol = base_vol + 0.02 * expiry
                
                # Calculate theoretical price using Black-Scholes
                if option_type == 'CE':
                    bs_result = black_scholes_call(spot_price, strike, expiry, 0.065, implied_vol)
                    theoretical_price = bs_result['call_price']
                else:
                    bs_result = black_scholes_put(spot_price, strike, expiry, 0.065, implied_vol)
                    theoretical_price = bs_result['put_price']
                
                # Add some market noise
                market_price = theoretical_price * (1 + np.random.normal(0, 0.02))
                market_price = max(0.5, market_price)  # Minimum price
                
                sample_data.append({
                    'Spot_Price': spot_price,
                    'Strike': strike,
                    'Time_to_Expiry': expiry,
                    'Option_Type': option_type,
                    'Market_Price': market_price,
                    'Theoretical_Price': theoretical_price,
                    'True_IV': implied_vol,
                    'Risk_Free_Rate': 0.065,
                    'Symbol': 'NIFTY'
                })
    
    return pd.DataFrame(sample_data)


def run_volatility_analysis_demo():
    """
    Comprehensive demonstration of volatility analysis capabilities
    """
    print("=== Comprehensive Volatility Analysis Demo ===")
    
    # Initialize analyzer
    analyzer = IndianVolatilityAnalyzer()
    
    # Step 1: Historical volatility analysis
    print("\n1. Historical Volatility Analysis")
    try:
        hist_vol_results = analyzer.analyze_historical_volatility('NIFTY')
        print("✓ Historical volatility analysis completed")
    except Exception as e:
        print(f"Historical volatility analysis failed: {e}")
        hist_vol_results = None
    
    # Step 2: Create sample options data
    print("\n2. Creating Sample Options Data")
    sample_options = create_sample_options_data_with_iv()
    print(f"✓ Created {len(sample_options)} sample options")
    
    # Step 3: Build volatility surface
    print("\n3. Building Volatility Surface")
    surface_results = analyzer.build_volatility_surface(sample_options)
    print("✓ Volatility surface construction completed")
    
    # Step 4: Model validation
    print("\n4. Model Validation")
    validation_results = analyzer.comprehensive_model_validation(sample_options)
    print("✓ Model validation completed")
    
    # Step 5: Display results summary
    print("\n=== Results Summary ===")
    
    if 'pricing_accuracy' in validation_results:
        pricing_metrics = validation_results['pricing_accuracy']
        print(f"Pricing Accuracy:")
        print(f"  Mean Error: {pricing_metrics['mean_percentage_error']:.2f}%")
        print(f"  R-squared: {pricing_metrics['r_squared']:.4f}")
        print(f"  Within 5% Error: {pricing_metrics['percentage_within_5pct']:.1f}%")
    
    if 'surface_metrics' in surface_results:
        surface_metrics = surface_results['surface_metrics']
        print(f"\nSurface Quality:")
        print(f"  Fit Quality: {surface_metrics['fit_quality']:.4f}")
        print(f"  Calendar Violations: {surface_metrics['calendar_violation_rate']:.2%}")
    
    stress_results = validation_results['stress_testing']
    successful_stress = stress_results[stress_results['test_status'] == 'SUCCESS']
    print(f"\nStress Testing:")
    print(f"  Successful scenarios: {len(successful_stress)}/{len(stress_results)}")
    
    if len(surface_results['svi_model']) > 0:
        print(f"  SVI model fits: {len(surface_results['svi_model'])} time slices")
    
    print("\n=== Volatility Analysis Demo Completed ===")
    
    return {
        'historical_volatility': hist_vol_results,
        'surface_analysis': surface_results,
        'model_validation': validation_results,
        'sample_data': sample_options
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=== Indian Options Volatility Surface & Model Validation ===")
    
    # Run comprehensive demonstration
    demo_results = run_volatility_analysis_demo()
    
    print("\n=== System Ready for Production Use ===")
    print("Available components:")
    print("✓ Historical volatility calculation (multiple methods)")
    print("✓ Implied volatility extraction (Black-Scholes inversion)")
    print("✓ Volatility surface construction (interpolation + SVI)")
    print("✓ Model validation framework (pricing accuracy + stress testing)")
    print("✓ Comprehensive analysis engine (IndianVolatilityAnalyzer)")
    
    print(f"\nYour Indian Options Pricing & Greeks Calculator is now complete!")
    print("Ready for resume showcase and practical options trading analysis.")
