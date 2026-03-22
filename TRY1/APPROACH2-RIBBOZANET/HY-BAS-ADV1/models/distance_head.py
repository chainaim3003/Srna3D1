"""distance_head.py — Re-imports from BASIC (identical code).
Note: pair_dim is a constructor argument, NOT hardcoded. BASIC uses 64, ADV1 uses 80.
The same DistanceMatrixHead class works for both — just pass pair_dim=80 in config.
"""
import sys, os
_basic_models = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'models'))
if _basic_models not in sys.path:
    sys.path.insert(0, _basic_models)
from distance_head import DistanceMatrixHead
