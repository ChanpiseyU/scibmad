# localized tetsing script to run the simple tests in test_scibmad_simple.py

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_scibmad_simple.py", "-v"],
    cwd=".",
)

sys.exit(result.returncode)