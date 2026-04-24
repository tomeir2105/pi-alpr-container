#!/bin/sh

./stop_watcher.sh

docker compose up -d --build --force-recreate watcher


docker image prune -f
