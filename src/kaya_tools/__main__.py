"""Entry point for `python -m kaya_tools` and the `bin/kaya` launcher."""

import sys

if sys.version_info < (3, 11):
    sys.exit(f"kaya needs Python 3.11 or newer (this is {sys.version.split()[0]}); set KAYA_PYTHON to a newer one")

from kaya_tools.cli import main  # noqa: E402

sys.exit(main())
