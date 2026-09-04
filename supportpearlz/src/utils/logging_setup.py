import logging
from src.config import settings

def setup_logging():
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("supportpearlz.log")
        ]
    )
    return logging.getLogger(__name__)