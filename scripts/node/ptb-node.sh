#!/bin/sh
# No sudo, global installs, shell profile edits, or persistent services.
set -eu
: "${PTB_NODE_PYTHON:?Set PTB_NODE_PYTHON to the dedicated native Linux Python}"
case "$PTB_NODE_PYTHON" in /*) ;; *) echo 'Explicit absolute Python path required' >&2; exit 2;; esac
exec "$PTB_NODE_PYTHON" -B -m ptb_node "$@"
