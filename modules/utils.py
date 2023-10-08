"""General purpose utility functions.

This module do not import other PA modules.
"""
import logging

from InquirerPy import inquirer
from InquirerPy.validator import PathValidator

logger = logging.getLogger(__name__)

def confirm_action(message: str='') -> bool:
    """Confirm execution of an action."""

    if not message:
        message = 'Are you sure?'
    confirm = inquirer.confirm(message=message).execute()
    logger.debug(f'"{message}" = {confirm}')
    return confirm