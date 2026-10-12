class TradingMLError(Exception):
    """Error esperado de datos, configuración o uso del motor."""


class DataError(TradingMLError):
    """Los datos no cumplen el contrato requerido."""


class LeakageError(TradingMLError):
    """El período solicitado puede contaminar la evaluación."""
