import logging
from rich.logging import RichHandler
from rich.console import Console

# Initialize Rich console for beautiful printing
console = Console()

def setup_logger(name: str = "ZS_Analytics"):
    """
    Sets up a logger with RichHandler for professional CLI output.
    """
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console, show_time=True)]
    )
    return logging.getLogger(name)

logger = setup_logger()
