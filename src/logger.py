import logging
import os
from datetime import datetime

LOGGING = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(), "logs", LOGGING)
os.makedirs(log_path, exist_ok=True)

LOGGING_PATH = os.path.join(log_path, LOGGING)

logging.basicConfig(
    filename=LOGGING_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)