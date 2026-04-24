#!/bin/sh

./stop_watcher.sh

docker compose up -d --build

docker image prune -f
