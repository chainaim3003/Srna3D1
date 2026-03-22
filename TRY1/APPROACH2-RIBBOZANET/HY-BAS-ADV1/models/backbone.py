"""backbone.py — Re-imports from BASIC (identical code, avoids duplication).
For Kaggle: upload BASIC/models/backbone.py directly as part of adv1-code dataset.
"""
import sys, os
_basic_models = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'models'))
if _basic_models not in sys.path:
    sys.path.insert(0, _basic_models)
from backbone import load_backbone, tokenize_sequence, OFFICIAL_BASE_TO_IDX, OFFICIAL_PAD_IDX
