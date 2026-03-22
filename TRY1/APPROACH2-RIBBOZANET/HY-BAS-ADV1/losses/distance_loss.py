"""distance_loss.py — Re-imports from BASIC/losses/distance_loss.py."""
import sys, os
_basic_losses = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'losses'))
if _basic_losses not in sys.path:
    sys.path.insert(0, _basic_losses)
from distance_loss import DistanceMatrixLoss
