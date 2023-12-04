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

cd ./services/python

docker compose --env-file .env.build pull
docker compose --env-file .env.build build

echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

DOCKER_IMAGE_HASH=$(docker images --format "{{.ID}} {{.CreatedAt}}" | sort -rk 2 | awk 'NR==1{print $1}')
docker tag "$DOCKER_IMAGE_HASH" "$GITHUB_CONTAINER_REPO"
docker push "$GITHUB_CONTAINER_REPO"