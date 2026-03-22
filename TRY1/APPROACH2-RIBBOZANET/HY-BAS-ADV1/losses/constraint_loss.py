"""constraint_loss.py — Re-imports from BASIC/losses/constraint_loss.py."""
import sys, os
_basic_losses = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'losses'))
if _basic_losses not in sys.path:
    sys.path.insert(0, _basic_losses)
from constraint_loss import BondConstraintLoss, ClashPenaltyLoss
