from pathlib import Path

about = {}
here = Path(__file__).parent.resolve()

with open(here / "__version__.py", "r") as f:
    exec(f.read(), about)
__version__ = about["__version__"]
