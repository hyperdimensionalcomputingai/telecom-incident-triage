"""Build the illustrated report from the completed experiment results."""

from settings import RUN_DIR
from study import report

if __name__ == "__main__":
    report(RUN_DIR)
