def format_float(x: float, decimal_places: int=0, decimal_sep: str=','):
    return ('{0:.' + str(decimal_places) + 'f}').format(x).replace('.', decimal_sep)