class FastAppException(Exception):
    """Base exception for FastApp"""
    pass

class PipelineNotConfiguredError(FastAppException):
    """Pipeline is not configured"""
    pass

class PipelineNotReadyError(FastAppException):
    """Pipeline is not ready for use"""
    pass

class FileProcessingError(FastAppException):
    """Error processing files"""
    pass

class InvalidConfigurationError(FastAppException):
    """Invalid configuration provided"""
    pass