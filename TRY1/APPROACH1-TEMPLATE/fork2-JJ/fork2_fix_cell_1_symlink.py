import os
src = "/kaggle/input/competitions/stanford-rna-3d-folding-2"
dst = "/kaggle/input/stanford-rna-3d-folding-2"
if os.path.exists(src) and not os.path.exists(dst):
    os.symlink(src, dst)
    print("Symlink created")
else:
    print("No symlink needed")
