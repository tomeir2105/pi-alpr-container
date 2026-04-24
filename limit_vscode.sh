#!/bin/bash

pids=$(ps -eo pid=,comm=,args= | grep -Ei 'Amazon Q Helper|Amazon Q|codex|code-[a-z0-9]+' | grep -v grep | awk '{print $1}')

if [ -z "$pids" ]; then
  echo "No Amazon Q / Codex processes found"
  exit 1
fi

for pid in $pids; do
  pname=$(ps -p "$pid" -o comm=)

  # kill previous cpulimit for this PID
  pkill -f "cpulimit -p $pid" >/dev/null 2>&1

  # reset priority first
  renice 0 -p "$pid" >/dev/null 2>&1

  # apply new limits
  renice +3 -p "$pid" >/dev/null 2>&1
  nohup cpulimit -p "$pid" -l 45 >/dev/null 2>&1 &

  echo "PID $pid ($pname) reset and limited"
done
