class Q3TTSError(RuntimeError):
    """Base error for actionable node and CLI failures."""


class CheckpointError(Q3TTSError):
    pass


class BackendUnavailableError(Q3TTSError):
    pass


class InsufficientVRAMError(Q3TTSError):
    pass


class CompatibilityError(Q3TTSError):
    pass

