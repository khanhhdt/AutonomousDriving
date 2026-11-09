import sys
from pathlib import Path

_AGENTS_ROOT = Path(__file__).resolve().parent.parent / "third_party"
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))
