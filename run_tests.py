# localized testing script to run the tests in test_newpatch.py

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_scibmad_simple.py", "-v"],
    cwd=".",
)

sys.exit(result.returncode)