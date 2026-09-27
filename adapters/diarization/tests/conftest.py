import sys
from pathlib import Path

# Allow "import main" without installing the adapter as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
