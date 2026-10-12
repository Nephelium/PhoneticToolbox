from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('ptb-api')
except PackageNotFoundError:
    # Feedback-only source deployment does not install scientific dependencies.
    __version__ = '0+source'
