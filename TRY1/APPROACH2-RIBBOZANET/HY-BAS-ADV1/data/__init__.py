"""data package — Re-imports from BASIC (identical code)."""
import sys, os
_basic_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'data'))
if _basic_data not in sys.path:
    sys.path.insert(0, _basic_data)
from dataset import RNAStructureDataset, load_training_data
from collate import collate_rna_structures
