#!/bin/bash

CONTAINER="pi-alpr-watcher"

cid=$(docker ps -q -f "name=^/${CONTAINER}$")

if [ -z "$cid" ]; then
  echo "Container ${CONTAINER} is not running"
  exit 1
fi

docker stop "$cid" && echo "Container ${CONTAINER} stopped successfully"
docker image prune -f
