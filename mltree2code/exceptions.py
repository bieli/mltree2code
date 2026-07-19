class Mltree2codeError(Exception):
    pass


class UnsupportedModelError(Mltree2codeError):
    pass


class UnsupportedLanguageError(Mltree2codeError):
    pass


class ModelLoadError(Mltree2codeError):
    pass


class ParseError(Mltree2codeError):
    pass

