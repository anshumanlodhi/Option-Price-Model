# test_setup.py
# Quick test to verify all dependencies are working

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Test options pricing libraries
try:
    import py_vollib
    print("✓ py_vollib imported successfully")
except ImportError as e:
    print(f"✗ py_vollib import failed: {e}")

try:
    import mibian
    print("✓ mibian imported successfully")
except ImportError as e:
    print(f"✗ mibian import failed: {e}")

# Test nsepy for Indian data
try:
    import nsepy
    print("✓ nsepy imported successfully")
except ImportError as e:
    print(f"✗ nsepy import failed: {e}")

# Test basic functionality
print("\n--- Testing Basic Functionality ---")

# Test pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"✓ Pandas DataFrame created: {df.shape}")

# Test numpy
arr = np.array([1, 2, 3, 4, 5])
print(f"✓ NumPy array created: {arr.mean()}")

# Test scipy
normal_dist = stats.norm(0, 1)
print(f"✓ SciPy normal distribution: {normal_dist.cdf(0)}")

# Test matplotlib
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Test Plot")
plt.savefig("test_plot.png")
plt.close()
print("✓ Matplotlib plot saved as test_plot.png")

# Test yfinance with Indian stock
try:
    ticker = yf.Ticker("RELIANCE.NS")
    info = ticker.info
    print(f"✓ Yahoo Finance: Retrieved {info.get('longName', 'Reliance')} data")
except Exception as e:
    print(f"✗ Yahoo Finance test failed: {e}")

# Test py_vollib Black-Scholes
try:
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.greeks.analytical import delta
    
    # Test calculation
    price = black_scholes('c', 100, 105, 0.25, 0.05, 0.2)
    delta_val = delta('c', 100, 105, 0.25, 0.05, 0.2)
    
    print(f"✓ Black-Scholes call price: {price:.4f}")
    print(f"✓ Delta calculation: {delta_val:.4f}")
except Exception as e:
    print(f"✗ py_vollib test failed: {e}")

# Test mibian
try:
    import mibian
    
    # Create option pricing instance
    option = mibian.BS([100, 105, 0.05, 0.25], volatility=20)
    call_price = option.callPrice
    put_price = option.putPrice
    
    print(f"✓ Mibian call price: {call_price:.4f}")
    print(f"✓ Mibian put price: {put_price:.4f}")
except Exception as e:
    print(f"✗ Mibian test failed: {e}")

print("\n--- Setup Test Complete ---")
print("If all tests passed, you're ready to proceed!")