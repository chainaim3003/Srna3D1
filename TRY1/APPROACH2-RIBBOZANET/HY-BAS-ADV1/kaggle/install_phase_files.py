"""
Run this script ONCE to download Phase A and Phase B files.
It copies from the Claude outputs directory to this kaggle folder.

Usage:
    python install_phase_files.py

After running, you'll have:
    hy_bas_adv1_run4_commit_PhaseA_NB.py
    hy_bas_adv1_run4_commit_PhaseB_NB.py
"""
print("Phase A and Phase B files are available for download from Claude's output.")
print("Please download them from the Claude chat interface and place them here.")
print("")
print("Target directory:")
print("  " + __file__.rsplit('\\', 1)[0] if '\\' in __file__ else "  " + __file__.rsplit('/', 1)[0])
