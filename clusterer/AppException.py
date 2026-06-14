class AppException(Exception):
    pass

class BuildUMAPException(AppException):
    pass
class ReduceUMAPException(AppException):
    pass

class BuildHdbscanException(AppException):
    pass
class ClusterHdbscanException(AppException):
    pass
class ProbsHdbscanException(AppException):
    pass

class ReducedToCPException(AppException):
    pass

class PeriferyException(AppException):
    pass

class NoConfigChangeException(AppException):
    pass