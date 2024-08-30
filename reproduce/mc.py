import sys
import os

# Add the mergecraft directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from mergecraft import *

print('Imported mergecraft')