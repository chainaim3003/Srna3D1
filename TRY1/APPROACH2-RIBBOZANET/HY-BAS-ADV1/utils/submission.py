"""submission.py — Re-imports from BASIC/utils/submission.py."""
import sys, os
_basic_utils = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'BASIC', 'utils'))
if _basic_utils not in sys.path:
    sys.path.insert(0, _basic_utils)
from submission import format_submission, load_test_sequences
