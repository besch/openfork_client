"""
DGN Client Exception Hierarchy

Custom exceptions for the DGN (Distributed GPU Network) client that provide
better error categorization and enable proper retry logic.

Exception Hierarchy:
    DGNError (base)
    ├── TransientError (retryable - network issues, 503, timeouts)
    ├── PermanentError (not retryable)
    │   ├── AuthError (token expired, invalid API key)
    │   └── ProviderError (provider not found/expired)
    │   └── UpgradeRequiredError (client must be updated)
    └── WorkflowError (workflow file issues)
"""


class DGNError(Exception):
    """Base exception for all DGN client errors."""
    pass


class TransientError(DGNError):
    """
    Transient errors that may succeed on retry.
    
    Examples:
        - Network connectivity issues
        - Server temporarily unavailable (503)
        - Request timeouts
        - Rate limiting (429)
    """
    pass


class InfrastructureError(TransientError):
    """
    Provider-side infrastructure errors that should requeue the job.

    Raised when:
        - Docker/container runtime fails
        - GPU runtime hits CUDA out-of-memory or a broken CUDA context
        - A local model server fails in a way that is provider-specific
    """
    pass


class PermanentError(DGNError):
    """
    Permanent errors that will not succeed on retry.
    
    Examples:
        - Invalid credentials
        - Resource not found
        - Permission denied
    """
    pass


class AuthError(PermanentError):
    """
    Authentication/authorization errors.
    
    Raised when:
        - Access token has expired
        - Refresh token is invalid
        - API key is revoked
    """
    pass


class ProviderError(PermanentError):
    """
    Provider-related errors.
    
    Raised when:
        - Provider registration has expired
        - Provider not found in database
        - Provider cleanup by stale provider cron
    """
    pass


class UpgradeRequiredError(PermanentError):
    """
    Client version/protocol is below the orchestrator's required policy.

    Raised when the control plane returns HTTP 426. The client should stop
    processing and let the desktop app or operator install an update.
    """

    def __init__(self, payload=None):
        self.payload = payload if isinstance(payload, dict) else {}
        message = self.payload.get("message") or "OpenFork update required"
        super().__init__(message)


class WorkflowError(DGNError):
    """
    Workflow file errors.
    
    Raised when:
        - Workflow file not found
        - Invalid workflow JSON format
        - Missing required workflow nodes
    """
    pass


class ConfigurationError(DGNError):
    """
    Configuration errors.
    
    Raised when:
        - Invalid configuration values
        - Missing required configuration
    """
    pass
