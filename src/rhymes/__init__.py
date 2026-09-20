"""rhymes - turn a nursery rhyme you wrote into a sung audio track.

Deliberately empty of submodule imports: importing this package must cost
nothing, so that `rhymes --help` and `rhymes styles` stay instant and the
validation layer can reject a malformed file before anything heavy loads.
"""

__version__ = "0.1.0"
