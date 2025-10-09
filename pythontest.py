#!/usr/bin/env python3
"""Test if environment is set up correctly"""

print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except ImportError as e:
    print("✗ NumPy not installed:", e)

try:
    import scipy
    import scipy.io
    print("✓ SciPy:", scipy.__version__)
except ImportError as e:
    print("✗ SciPy not installed:", e)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    print("✓ Matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("✗ Matplotlib not installed:", e)

try:
    from PIL import Image
    print("✓ Pillow (PIL):", Image.__version__)
except ImportError as e:
    print("✗ Pillow not installed:", e)

try:
    import sklearn
    print("✓ scikit-learn:", sklearn.__version__)
except ImportError as e:
    print("✗ scikit-learn not installed:", e)

try:
    import plotly
    print("✓ Plotly:", plotly.__version__)
except ImportError:
    print("○ Plotly not installed (optional)")

print("\n" + "="*50)

# Test loading a .mat file
print("\nTesting scipy.io.loadmat...")
try:
    # Create a dummy .mat file
    import scipy.io
    dummy_data = {'test': np.array([1, 2, 3])}
    scipy.io.savemat('/tmp/test.mat', dummy_data)
    loaded = scipy.io.loadmat('/tmp/test.mat')
    print("✓ Can load .mat files")
except Exception as e:
    print("✗ Error loading .mat files:", e)

print("\n✓✓✓ Environment setup complete! ✓✓✓")