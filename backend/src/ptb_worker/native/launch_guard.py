"""Keep the admission descriptor alive while our systemd client is alive.

The stdlib-only guard removes any dependency on systemd-run preserving unknown
inherited descriptors. It is an owned process, not a service or public entry.
"""
import os
import signal
import subprocess
import sys


def main():
    descriptor=int(sys.argv[1])
    os.fstat(descriptor)  # Fail before spawning if the parent did not pass it.
    arguments=sys.argv[2:]
    if not arguments or arguments[0] != 'systemd-run':
        raise ValueError('Fixed systemd-run client required')
    # Parent cleans the exact systemd unit before stopping this guard. During
    # startup/parent crash, retain the lock until the client has actually exited.
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGINT, lambda *_: None)
    child=subprocess.Popen(arguments, stdin=sys.stdin.buffer,stdout=sys.stdout.buffer,
                           stderr=sys.stderr.buffer,close_fds=True)
    return child.wait()


if __name__=='__main__':
    raise SystemExit(main())
