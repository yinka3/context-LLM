#!/bin/bash


echo "Cleaning up"

redis-cli flushall

docker exec -i vestige-memgraph mgconsole <<EOF
MATCH (n) DETACH DELETE n;
EOF

echo "Cleaning up done for redis and docker"

echo "Clean up log files now"

CONFIRM=""

read -p "Remove logs, yes or no " CONFIRM

if [[ "$CONFIRM" == "yes" ]]; then
    echo "removing logs now..."
    rm *.log
fi


