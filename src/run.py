"""Run the four experiments on the prepared dataset."""
from threadpoolctl import threadpool_limits
from settings import RUN_DIR
from study import run

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run(RUN_DIR)
