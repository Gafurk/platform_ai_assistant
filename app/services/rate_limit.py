import time
from collections import defaultdict


class RateLimiter:
    """Token bucket rate limiter per session."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        max_requests: number of requests allowed per window
        window_seconds: time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """Check if identifier is within rate limit."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]

        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True

        return False

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds

        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]

        return max(0, self.max_requests - len(self.requests[identifier]))


# Global rate limiter: 10 requests per 60 seconds per session
chat_limiter = RateLimiter(max_requests=10, window_seconds=60)
