#!/usr/bin/env bash
# Start one SAM3 server per port in a tmux session named "sam3"; a server that stops is restarted.
#
#   ./scripts/launch_servers.sh 8007 8008
#   tmux attach -t sam3          # one window per port, Ctrl-b d to leave
#   tmux kill-session -t sam3    # stop them all

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "usage: $0 <port> [port ...]" >&2
    exit 1
fi
if tmux has-session -t sam3 2>/dev/null; then
    echo "The SAM3 servers are already running. Look at them with: tmux attach -t sam3" >&2
    echo "Stop them all first with: tmux kill-session -t sam3" >&2
    exit 1
fi

first=1
for port in "$@"; do
    run="until uv run uvicorn demo.server:app --host 0.0.0.0 --port $port; do echo 'Server stopped, restarting in 5 s'; sleep 5; done"
    if [ "$first" = 1 ]; then
        tmux new-session -d -s sam3 -n "$port" "$run"
        first=0
    else
        tmux new-window -t sam3 -n "$port" "$run"
    fi
done
echo "SAM3 servers starting on port(s) $*. Watch them with: tmux attach -t sam3"
