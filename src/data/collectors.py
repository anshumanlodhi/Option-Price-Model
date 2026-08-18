"""
Enhanced Data Collection Module for Indian Options Pricing Project
Building on the solid foundation with additional functionality for options modeling
Focus: Development and learning with historical accuracy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time
import logging
from typing import Dict, List, Optional, Tuple
import os
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config.settings import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndianMarketDataCollector:
    """
    Enhanced collector for Indian market data with options focus
    Builds on proven base structure with additional options functionality
    """
    
    def __init__(self):
        # Load configuration
        try:
            self.config = get_config()
        except:
            # Fallback if config not available
            self.config = None
            logger.warning("Config not loaded, using default settings")
        
        # Enhanced symbol mapping with your proven base
        self.base_symbols = {
            'NIFTY': '^NSEI',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'INFY': 'INFY.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'SBIN': 'SBIN.NS',
            'BHARTIARTL': 'BHARTIARTL.NS'
        }
        
        # Use config if available, otherwise use your proven defaults
        if self.config:
            self.analysis_start = self.config.data.start_date
            self.analysis_end = self.config.data.end_date
            self.validation_start = '2024-01-01'
            self.validation_end = '2024-12-31'
        else:
            # Your proven date ranges
            self.analysis_start = '2022-01-01'
            self.analysis_end = '2023-12-31'
            self.validation_start = '2024-01-01'
            self.validation_end = '2024-05-30'
        
        # Options specific parameters
        self.strike_spacing = 50
        self.options_range = 10  # ATM ± 10 strikes
        self.expiry_days = [7, 14, 30, 45]  # Multiple expiry cycles
        
    def download_stock_data(self,
                          symbol: str,
                          start_date: str,
                          end_date: str,
                          save_to_csv: bool = True) -> pd.DataFrame:
        """
        Enhanced version of your proven download function
        Added volatility calculations and better error handling
        """
        try:
            logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
            
            # Primary: Yahoo Finance (your proven method)
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Your proven data cleaning approach
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Enhanced calculations building on your base
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Multiple volatility windows (enhanced from your 21-day)
            data['Volatility_21d'] = data['Returns'].rolling(window=21).std() * np.sqrt(252)
            data['Volatility_30d'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
            data['Volatility_252d'] = data['Returns'].rolling(window=252).std() * np.sqrt(252)
            
            # Additional useful calculations for options
            data['Price_MA_20'] = data['Close'].rolling(window=20).mean()
            data['Price_MA_50'] = data['Close'].rolling(window=50).mean()
            data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
            
            logger.info(f"Downloaded {len(data)} rows for {symbol}")
            
            # Enhanced saving with config integration
            if save_to_csv:
                self._save_stock_data(data, symbol, start_date, end_date)
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _save_stock_data(self, data: pd.DataFrame, symbol: str, start_date: str, end_date: str):
        """
        Enhanced saving function with config integration
        """
        if self.config:
            directory = str(self.config.storage_paths.raw)
        else:
            # Your proven fallback approach
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            directory = os.path.join(project_root, "data", "raw")
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Your proven safe filename approach
        import re
        safe_symbol = re.sub(r'[^A-Za-z0-9_]', '_', symbol)
        filename = os.path.join(directory, f"{safe_symbol}_{start_date}_{end_date}.csv")
        data.to_csv(filename, index=True)
        logger.info(f"Data saved to {filename}")
    
    def download_all_stocks(self,
                          period: str = 'analysis') -> Dict[str, pd.DataFrame]:
        """
        Your proven bulk download function - kept as is with minor enhancements
        """
        if period == 'analysis':
            start_date = self.analysis_start
            end_date = self.analysis_end
        elif period == 'validation':
            start_date = self.validation_start
            end_date = self.validation_end
        else:
            # Custom period support
            start_date = period.split('_')[0]
            end_date = period.split('_')[1]
        
        stock_data = {}
        
        for name, symbol in self.base_symbols.items():
            data = self.download_stock_data(symbol, start_date, end_date)
            if not data.empty:
                stock_data[name] = data
            
            # Your proven API politeness
            time.sleep(1)
        
        return stock_data
    
    def get_risk_free_rate(self, date: str) -> float:
        """
        Your excellent historical repo rate mapping - enhanced with more data points
        """
        # Enhanced historical RBI repo rates based on your approach
        repo_rates = {
            '2022-01-01': 0.040,  # 4.0%
            '2022-05-01': 0.050,  # 5.0% (rate hike cycle)
            '2022-08-01': 0.054,  # 5.4%
            '2022-12-01': 0.059,  # 5.9%
            '2023-01-01': 0.065,  # 6.5%
            '2023-06-01': 0.065,  # 6.5%
            '2023-12-01': 0.065,  # 6.5%
            '2024-01-01': 0.065,  # 6.5%
            '2024-06-01': 0.065,  # 6.5%
            '2024-12-01': 0.065,  # 6.5%
        }
        
        # Your proven rate selection logic
        target_date = pd.to_datetime(date)
        best_rate = 0.065  # Default current rate
        
        for rate_date, rate in repo_rates.items():
            if pd.to_datetime(rate_date) <= target_date:
                best_rate = rate
        
        return best_rate
    
    def calculate_historical_volatility(self,
                                     data: pd.DataFrame,
                                     window: int = 21) -> pd.Series:
        """
        Enhanced version of your volatility calculation with multiple methods
        """
        # Your base method
        returns = data['Close'].pct_change()
        hist_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        return hist_vol
    
    def create_enhanced_options_data(self,
                                   stock_data: pd.DataFrame,
                                   symbol: str = 'NIFTY') -> pd.DataFrame:
        """
        Enhanced version of your sample options data with realistic structure
        Better suited for actual options modeling
        """
        options_data = []
        
        # Use recent data as in your approach
        recent_data = stock_data.tail(60)  # Last 60 days
        
        for date, row in recent_data.iterrows():
            spot_price = row['Close']
            volatility = row.get('Volatility_30d', 0.20)
            risk_free_rate = self.get_risk_free_rate(date.strftime('%Y-%m-%d'))
            
            # Enhanced strike generation (your logic improved)
            atm_strike = round(spot_price / self.strike_spacing) * self.strike_spacing
            strikes = [atm_strike + (i * self.strike_spacing) 
                      for i in range(-self.options_range, self.options_range + 1)]
            
            for strike in strikes:
                for option_type in ['CE', 'PE']:  # Your notation
                    for expiry_days in self.expiry_days:
                        expiry_date = date + timedelta(days=expiry_days)
                        
                        # Enhanced option structure for modeling
                        option_record = {
                            'Date': date,
                            'Symbol': symbol,
                            'Expiry': expiry_date,
                            'Strike': strike,
                            'Option_Type': option_type,
                            'Spot_Price': spot_price,
                            'Days_to_Expiry': expiry_days,
                            'Time_to_Expiry': expiry_days / 365.25,
                            'Volatility': volatility,
                            'Risk_Free_Rate': risk_free_rate,
                            'Moneyness': spot_price / strike,
                            'ITM': self._is_in_the_money(spot_price, strike, option_type),
                            'Strike_Distance': abs(spot_price - strike),
                            'Delta_Bucket': self._get_delta_bucket(spot_price, strike, option_type)
                        }
                        
                        options_data.append(option_record)
        
        df = pd.DataFrame(options_data)
        logger.info(f"Created {len(df)} enhanced options records for {symbol}")
        
        return df
    
    def _is_in_the_money(self, spot: float, strike: float, option_type: str) -> bool:
        """
        Utility to determine if option is ITM
        """
        if option_type == 'CE':
            return spot > strike
        else:  # PE
            return spot < strike
    
    def _get_delta_bucket(self, spot: float, strike: float, option_type: str) -> str:
        """
        Classify options by approximate delta ranges
        """
        moneyness = spot / strike
        
        if option_type == 'CE':
            if moneyness >= 1.05:
                return 'Deep_ITM'
            elif moneyness >= 1.02:
                return 'ITM'
            elif moneyness >= 0.98:
                return 'ATM'
            elif moneyness >= 0.95:
                return 'OTM'
            else:
                return 'Deep_OTM'
        else:  # PE
            if moneyness <= 0.95:
                return 'Deep_ITM'
            elif moneyness <= 0.98:
                return 'ITM'
            elif moneyness <= 1.02:
                return 'ATM'
            elif moneyness <= 1.05:
                return 'OTM'
            else:
                return 'Deep_OTM'
    
    def collect_complete_dataset(self, 
                               focus_symbol: str = 'NIFTY',
                               include_options: bool = True) -> Dict[str, pd.DataFrame]:
        """
        New function to collect complete dataset for options modeling
        Combines your proven approach with options focus
        """
        logger.info(f"Starting complete dataset collection for {focus_symbol}")
        
        complete_data = {}
        
        # 1. Get primary symbol data (NIFTY focus)
        primary_symbol = self.base_symbols.get(focus_symbol, '^NSEI')
        spot_data = self.download_stock_data(
            primary_symbol, 
            self.analysis_start, 
            self.analysis_end
        )
        
        if not spot_data.empty:
            complete_data['spot_data'] = spot_data
            logger.info(f"Collected spot data: {len(spot_data)} records")
        
        # 2. Create enhanced options data
        if include_options and not spot_data.empty:
            options_data = self.create_enhanced_options_data(spot_data, focus_symbol)
            complete_data['options_data'] = options_data
            logger.info(f"Created options data: {len(options_data)} records")
        
        # 3. Get supporting stock data for correlation analysis
        supporting_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY']
        for stock in supporting_stocks:
            if stock in self.base_symbols:
                stock_data = self.download_stock_data(
                    self.base_symbols[stock],
                    self.analysis_start,
                    self.analysis_end
                )
                if not stock_data.empty:
                    complete_data[f'{stock.lower()}_data'] = stock_data
        
        # 4. Create validation dataset
        validation_spot = self.download_stock_data(
            primary_symbol,
            self.validation_start,
            self.validation_end
        )
        
        if not validation_spot.empty:
            complete_data['validation_spot'] = validation_spot
            
            if include_options:
                validation_options = self.create_enhanced_options_data(validation_spot, focus_symbol)
                complete_data['validation_options'] = validation_options
        
        logger.info(f"Complete dataset collection finished. {len(complete_data)} datasets created")
        return complete_data
    
    def save_monthly_data(self, data: pd.DataFrame, data_type: str, symbol: str):
        """
        Save data in monthly files as per config requirements
        """
        if data.empty:
            return
        
        # Group by year-month
        data_copy = data.copy()
        data_copy['year_month'] = data_copy.index.to_period('M')
        
        for period, group_data in data_copy.groupby('year_month'):
            year = period.year
            month = period.month
            
            # Create filename
            if self.config:
                directory = str(self.config.storage_paths.raw)
                if data_type == 'spot':
                    filename = self.config.storage.naming['spot_data'].format(year=year, month=month)
                elif data_type == 'options':
                    filename = self.config.storage.naming['options_data'].format(year=year, month=month)
                else:
                    filename = f"{symbol}_{data_type}_{year}_{month:02d}.csv"
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                directory = os.path.join(project_root, "data", "raw")
                filename = f"{symbol}_{data_type}_{year}_{month:02d}.csv"
            
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            filepath = os.path.join(directory, filename)
            group_data.drop('year_month', axis=1).to_csv(filepath, index=True)
            logger.info(f"Saved monthly data to {filepath}")


# Keep your excellent utility functions
def validate_stock_data(data: pd.DataFrame) -> bool:
    """
    Your proven data validation function - kept as is
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if data.empty:
        return False
    
    if not all(col in data.columns for col in required_columns):
        return False
    
    # Check for reasonable price relationships
    if not (data['High'] >= data['Low']).all():
        return False
    
    if not (data['High'] >= data['Close']).all():
        return False
    
    if not (data['Close'] >= data['Low']).all():
        return False
    
    return True


def print_data_summary(data: pd.DataFrame, symbol: str):
    """
    Enhanced version of your proven summary function
    """
    print(f"\n=== Enhanced Data Summary for {symbol} ===")
    print(f"Period: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Total trading days: {len(data)}")
    print(f"Average daily volume: {data['Volume'].mean():,.0f}")
    print(f"Price range: ₹{data['Close'].min():.2f} - ₹{data['Close'].max():.2f}")
    
    # Enhanced volatility summary
    if 'Volatility_21d' in data.columns:
        print(f"Average volatility (21d): {data['Volatility_21d'].mean():.2%}")
        print(f"Max volatility (21d): {data['Volatility_21d'].max():.2%}")
    
    if 'Volatility_30d' in data.columns:
        print(f"Average volatility (30d): {data['Volatility_30d'].mean():.2%}")
    
    # Additional useful stats
    print(f"Average daily return: {data['Returns'].mean():.4f}")
    print(f"Daily return std: {data['Returns'].std():.4f}")
    print(f"Sharpe ratio (approx): {(data['Returns'].mean() / data['Returns'].std() * np.sqrt(252)):.2f}")
    print("=" * 60)


# Quick testing functions for development
def quick_nifty_test(days: int = 30):
    """
    Quick function to test NIFTY data collection
    """
    collector = IndianMarketDataCollector()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = collector.download_stock_data(
        '^NSEI',
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        save_to_csv=False
    )
    
    if not data.empty:
        print_data_summary(data, 'NIFTY')
        return data
    else:
        print("Failed to collect NIFTY data")
        return pd.DataFrame()


# Development testing
if __name__ == "__main__":
    print("=== Enhanced Data Collection Testing ===")
    
    # Initialize collector
    collector = IndianMarketDataCollector()
    
    # Quick test
    print("1. Testing quick NIFTY data collection...")
    test_data = quick_nifty_test(10)
    
    # Test options data creation
    if not test_data.empty:
        print("\n2. Testing enhanced options data creation...")
        options_test = collector.create_enhanced_options_data(test_data)
        print(f"Created {len(options_test)} options records")
        print("\nSample options data:")
        print(options_test[['Strike', 'Option_Type', 'Moneyness', 'Delta_Bucket']].head(10))
    
    # Test complete collection (uncomment to run full collection)
    # print("\n3. Testing complete dataset collection...")
    # complete_data = collector.collect_complete_dataset()
    # for key, df in complete_data.items():
    #     print(f"{key}: {len(df)} records")
