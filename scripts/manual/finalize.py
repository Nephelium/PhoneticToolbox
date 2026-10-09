"""Synchronize the editable manual index after chapter authors finish.

Section and figure numbers are rendered by the reader from chapter order. The
chapter author owns media placement, captions, structure, and review status.
This compatibility entrypoint deliberately leaves chapter bodies untouched.
"""

from assemble import main


if __name__ == '__main__':
    main()
