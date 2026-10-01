"""Generate the configured dataset and freeze its input records."""

from config import Config
from settings import CONFIG_FILE, RUN_DIR
from study import prepare

if __name__ == "__main__":
    prepare(RUN_DIR, Config.read(CONFIG_FILE))
