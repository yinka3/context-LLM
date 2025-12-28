#!/bin/bash

echo "Wiping everything..."

docker compose down -v 2>/dev/null
sudo rm -rf ./redis_data 
rm -f *.log

echo Restarting containers and volumes
docker compose up -d
echo "Done."
clear
