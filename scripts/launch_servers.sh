#!/usr/bin/env bash
# Start one SAM3 server per port, replacing the caller's own servers. The servers run in the
# background, detached from this terminal, and write to logs/sam3-<port>.log.
#
#   ./scripts/launch_servers.sh [--free-gpu] [--free-ports] <port> [port ...]
#
# --free-gpu    first remove other accounts' processes from the GPU (sudo free-gpu)
# --free-ports  first remove other accounts' listeners from the given ports (sudo free-ports)
# Both come from LeTS/scripts and need the sudo rights described there.

set -euo pipefail
cd "$(dirname "$0")/.."

free_gpu=0
free_ports=0
ports=()
for arg in "$@"; do
    case "$arg" in
        --free-gpu) free_gpu=1 ;;
        --free-ports) free_ports=1 ;;
        *) ports+=("$arg") ;;
    esac
done
if [ ${#ports[@]} -eq 0 ]; then
    echo "usage: $0 [--free-gpu] [--free-ports] <port> [port ...]" >&2
    exit 1
fi

if [ "$free_gpu" = 1 ]; then
    sudo free-gpu
fi
if [ "$free_ports" = 1 ]; then
    sudo free-ports "${ports[@]}"
fi

pkill -f "uvicorn demo.server:app" 2>/dev/null || true
for port in "${ports[@]}"; do
    for _ in $(seq 1 20); do
        ss -H -ltn "sport = :$port" | grep -q . || break
        sleep 0.5
    done
    if ss -H -ltn "sport = :$port" | grep -q .; then
        holder=$(ss -H -ltnp "sport = :$port" | grep -o 'users:.*' || echo "another account's process")
        echo "Port $port is still in use by $holder. Nothing started." >&2
        exit 1
    fi
done

mkdir -p logs
for port in "${ports[@]}"; do
    nohup setsid uv run uvicorn demo.server:app --host 0.0.0.0 --port "$port" > "logs/sam3-$port.log" 2>&1 < /dev/null &
done
echo "Mask servers starting on port(s) ${ports[*]}; they take about a minute. Logs: logs/sam3-<port>.log"
