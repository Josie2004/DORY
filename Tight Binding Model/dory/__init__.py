# dory/__init__.py
from .orbitals import OrbitalOrder, make_orbital_order
from .xmlio import DoryConfig, load_config
from .sk import SKParams, sk_block_sp3d5
from .model3d import Hamiltonian3D
#import spin 

# import logging

# # Configure a module-level logger
# logger = logging.getLogger("dory")
# logger.setLevel(logging.DEBUG)  # or INFO for less verbosity

# # File handler (logs go to dory.log)
# fh = logging.FileHandler("dory.log")
# fh.setLevel(logging.DEBUG)

# # Console handler (optional, prints to stdout)
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# # Format
# formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
# fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# # Add handlers
# logger.addHandler(fh)
# logger.addHandler(ch)