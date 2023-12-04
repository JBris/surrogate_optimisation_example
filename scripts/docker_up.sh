#!/usr/bin/env bash

###################################################################
# Constants
###################################################################

DOCKER_ENV_FILE_DEFAULT="$1"
DOCKER_ENV_FILE="${DOCKER_ENV_FILE_DEFAULT:=.env.local}"

###################################################################
# Imports
###################################################################

. $DOCKER_ENV_FILE 

###################################################################
# Main
###################################################################

docker compose --env-file $DOCKER_ENV_FILE down
docker compose --env-file $DOCKER_ENV_FILE pull
docker compose --env-file $DOCKER_ENV_FILE build
docker compose --env-file $DOCKER_ENV_FILE up -d 