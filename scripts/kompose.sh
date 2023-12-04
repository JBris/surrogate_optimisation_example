#!/usr/bin/env bash

###################################################################
# Constants
###################################################################

DOCKER_ENV_FILE_DEFAULT="$1"
DOCKER_ENV_FILE="${DOCKER_ENV_FILE_DEFAULT:=.env.local}"

###################################################################
# Main
###################################################################

docker compose --env-file $DOCKER_ENV_FILE --profile k8s config > docker-compose-resolved.yaml
kompose --file docker-compose-resolved.yaml --profile k8s convert --out deployment