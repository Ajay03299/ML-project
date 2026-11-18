# Notebooks

The experiments are scripts in `experiments/` (easier to review and merge). Use notebooks only to
explore; one owner per notebook. Shared setup:

```python
import sys; sys.path.append("..")
from experiments.common import get_data, parser
args = parser("nb").parse_args([])          # settings from src/config.py; add ["--quick"] for a small sample
panel, splits = get_data(args)              # same data and splits as every script
```
