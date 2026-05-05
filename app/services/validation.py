import re
from typing import Tuple


MAX_MESSAGE_LENGTH = 2000
MAX_SESSION_ID_LENGTH = 100
SESSION_ID_PATTERN = r'^[a-zA-Z0-9\-_]{1,100}$'


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_message(message: str) -> str:
    """Validate and sanitize user message."""
    if not message or not isinstance(message, str):
        raise ValidationError("Message must be a non-empty string")

    message = message.strip()
    if len(message) == 0:
        raise ValidationError("Message cannot be empty or whitespace-only")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValidationError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH}")

    # Remove null bytes which could cause issues
    message = message.replace('\x00', '')

    return message


def validate_session_id(session_id: str) -> str:
    """Validate session ID format."""
    if not session_id or not isinstance(session_id, str):
        raise ValidationError("Session ID must be a non-empty string")

    if len(session_id) > MAX_SESSION_ID_LENGTH:
        raise ValidationError(f"Session ID exceeds maximum length of {MAX_SESSION_ID_LENGTH}")

    if not re.match(SESSION_ID_PATTERN, session_id):
        raise ValidationError("Session ID contains invalid characters")

    return session_id
