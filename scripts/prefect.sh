#!/usr/bin/env bash

###################################################################
# Constants
###################################################################

PREFECT_FLOW_DEFAULT="$1"
PREFECT_FLOW="${PREFECT_FLOW_DEFAULT:=optimise_surrogate}"

DOCKER_ENV_FILE_DEFAULT="$2"
DOCKER_ENV_FILE="${DOCKER_ENV_FILE_DEFAULT:=.env.local}"

###################################################################
# Imports
###################################################################

. $DOCKER_ENV_FILE 

###################################################################
# Main
###################################################################

docker compose --env-file $DOCKER_ENV_FILE run --rm prefect-cli python -m flows.${PREFECT_FLOW}
