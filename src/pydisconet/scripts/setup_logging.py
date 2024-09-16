import logging

def setup_logging(level=logging.INFO):
    """
    Set up the global logging configuration.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Automatically set up logging when this module is imported
setup_logging()
