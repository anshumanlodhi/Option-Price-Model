
"""
Integration Example: Combining Data Collection with Options Pricing
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    from src.data.collectors import IndianMarketDataCollector
    from src.models.option_pricing import BlackScholesCalculator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class OptionsAnalyzer:
    """
    Combines data collection with options pricing analysis
    """

    def __init__(self, 
                 itm_threshold: float = 1.02, 
                 otm_threshold: float = 0.98,
                 default_volatility: float = 0.20):
        """
        Initialize the options analyzer
        
        Args:
            itm_threshold: Threshold for ITM classification (default: 1.02)
            otm_threshold: Threshold for OTM classification (default: 0.98)
            default_volatility: Default volatility if not available (default: 0.20)
        """
        self.data_collector = IndianMarketDataCollector()
        self.itm_threshold = itm_threshold
        self.otm_threshold = otm_threshold
        self.default_volatility = default_volatility

    def analyze_stock_options(self, symbol: str, strikes: List[float], expiry_days: int = 30):
        """
        Analyze options for a given stock with multiple strike prices

        Parameters:
        -----------
        symbol : str
            Stock symbol (e.g., 'RELIANCE.NS')
        strikes : List[float]
            List of strike prices to analyze
        expiry_days : int
            Days to expiry (default: 30)

        Returns:
        --------
        pd.DataFrame : Options analysis results
        """
        # Validate inputs
        if not strikes or len(strikes) == 0:
            print(f"❌ No strike prices provided for {symbol}")
            return pd.DataFrame()
            
        if expiry_days <= 0:
            print(f"❌ Invalid expiry days: {expiry_days}")
            return pd.DataFrame()

        try:
            # Get current stock data
            stock_data = self.data_collector.download_stock_data(
                symbol, 
                self.data_collector.validation_start,
                self.data_collector.validation_end,
                save_to_csv=False
            )

            if stock_data.empty:
                print(f"❌ No data available for {symbol}")
                return pd.DataFrame()

            # Get current price and volatility
            current_price = stock_data['Close'].iloc[-1]
            volatility = stock_data['Volatility_21d'].iloc[-1]
            
            # Handle missing volatility
            if pd.isna(volatility) or volatility <= 0:
                volatility = self.default_volatility
                print(f"⚠️ Using default volatility: {volatility:.1%}")

            # Use recent date for risk-free rate
            recent_date = stock_data.index[-1].strftime('%Y-%m-%d')
            risk_free_rate = self.data_collector.get_risk_free_rate(recent_date)
            
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {e}")
            return pd.DataFrame()

        print(f"📊 Analyzing {symbol} Options")
        print(f"Current Price: ₹{current_price:.2f}")
        print(f"Volatility (21d): {volatility:.1%}")
        print(f"Risk-Free Rate: {risk_free_rate:.1%}")
        print(f"Days to Expiry: {expiry_days}")
        print()

        # Analyze each strike price
        results = []

        for strike in strikes:
            try:
                calculator = BlackScholesCalculator(
                    spot_price=current_price,
                    strike_price=strike,
                    time_to_expiry=expiry_days/365,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility if not pd.isna(volatility) else 0.20
                )

                call_greeks = calculator.all_greeks('call')
                put_greeks = calculator.all_greeks('put')

                # Determine moneyness
                moneyness = current_price / strike
                if moneyness > self.itm_threshold:
                    money_status = "ITM"  # In-the-money for calls
                elif moneyness < self.otm_threshold:
                    money_status = "OTM"  # Out-of-the-money for calls
                else:
                    money_status = "ATM"  # At-the-money

                # Calculate additional metrics
                intrinsic_value_call = max(0, current_price - strike)
                intrinsic_value_put = max(0, strike - current_price)
                time_value_call = call_greeks['price'] - intrinsic_value_call
                time_value_put = put_greeks['price'] - intrinsic_value_put

                results.append({
                    'Strike': strike,
                    'Moneyness': money_status,
                    'Moneyness_Ratio': round(moneyness, 3),
                    'Call_Price': call_greeks['price'],
                    'Put_Price': put_greeks['price'],
                    'Call_Delta': call_greeks['delta'],
                    'Put_Delta': put_greeks['delta'],
                    'Gamma': call_greeks['gamma'],
                    'Call_Theta': call_greeks['theta'],
                    'Put_Theta': put_greeks['theta'],
                    'Vega': call_greeks['vega'],
                    'Call_Intrinsic': intrinsic_value_call,
                    'Put_Intrinsic': intrinsic_value_put,
                    'Call_Time_Value': time_value_call,
                    'Put_Time_Value': time_value_put
                })

            except Exception as e:
                print(f"⚠️ Error calculating for strike {strike}: {e}")

        return pd.DataFrame(results)
    
    def analyze_risk_metrics(self, options_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk metrics for the options chain
        
        Args:
            options_df: DataFrame with options analysis results
            
        Returns:
            Dictionary with risk metrics
        """
        if options_df.empty:
            return {}
            
        risk_metrics = {
            'total_options': len(options_df),
            'avg_call_delta': options_df['Call_Delta'].mean(),
            'avg_put_delta': options_df['Put_Delta'].mean(),
            'max_gamma': options_df['Gamma'].max(),
            'avg_vega': options_df['Vega'].mean(),
            'avg_call_theta': options_df['Call_Theta'].mean(),
            'avg_put_theta': options_df['Put_Theta'].mean(),
            'itm_count': len(options_df[options_df['Moneyness'] == 'ITM']),
            'atm_count': len(options_df[options_df['Moneyness'] == 'ATM']),
            'otm_count': len(options_df[options_df['Moneyness'] == 'OTM'])
        }
        
        return risk_metrics

def example_analysis():
    """
    Example: Analyze RELIANCE stock options
    """
    analyzer = OptionsAnalyzer()

    # Get RELIANCE current price range for realistic strikes
    reliance_data = analyzer.data_collector.download_stock_data(
        'RELIANCE.NS', '2024-01-01', '2024-05-30', save_to_csv=False
    )

    if not reliance_data.empty:
        current_price = reliance_data['Close'].iloc[-1]

        # Create strike prices around current price (±10%)
        strikes = []
        base_strike = int(current_price / 50) * 50  # Round to nearest 50
        for i in range(-4, 5):  # 9 strikes total
            strikes.append(base_strike + i * 50)

        print("🎯 RELIANCE Options Analysis")
        print("=" * 50)

        # Analyze options
        options_df = analyzer.analyze_stock_options('RELIANCE.NS', strikes, 30)

        if not options_df.empty:
            # Display results in a formatted table
            print("📋 Options Chain Analysis:")
            print()
            print(options_df.round(2).to_string(index=False))
            
            # Add summary statistics
            print("\n📊 Summary Statistics:")
            print(f"  Total Options Analyzed: {len(options_df)}")
            print(f"  ITM Options: {len(options_df[options_df['Moneyness'] == 'ITM'])}")
            print(f"  ATM Options: {len(options_df[options_df['Moneyness'] == 'ATM'])}")
            print(f"  OTM Options: {len(options_df[options_df['Moneyness'] == 'OTM'])}")
            print(f"  Average Call Price: ₹{options_df['Call_Price'].mean():.2f}")
            print(f"  Average Put Price: ₹{options_df['Put_Price'].mean():.2f}")
            print(f"  Max Gamma: {options_df['Gamma'].max():.6f}")

            # Save results with proper path
            output_dir = os.path.join(project_root, 'data', 'analysis')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'reliance_options_analysis.csv')
            options_df.to_csv(output_file, index=False)
            print(f"\n✅ Results saved to '{output_file}'")
        else:
            print("❌ No options data generated")
    else:
        print("❌ Could not get RELIANCE data for analysis")

if __name__ == "__main__":
    example_analysis()
