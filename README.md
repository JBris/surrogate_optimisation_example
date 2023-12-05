# Surrogate Optimisation Example 

[![Validate Pipeline](https://github.com/JBris/surrogate_optimisation_example/actions/workflows/validation.yaml/badge.svg?branch=main)](https://github.com/JBris/surrogate_optimisation_example/actions/workflows/validation.yaml) [![Generate Documentation](https://github.com/JBris/surrogate_optimisation_example/actions/workflows/docs.yaml/badge.svg?branch=main)](https://github.com/JBris/surrogate_optimisation_example/actions/workflows/docs.yaml) [![pages-build-deployment](https://github.com/JBris/surrogate_optimisation_example/actions/workflows/pages/pages-build-deployment/badge.svg?branch=gh-pages)](https://github.com/JBris/surrogate_optimisation_example/actions/workflows/pages/pages-build-deployment)

Website: [Surrogate Optimisation Example](https://jbris.github.io/surrogate_optimisation_example/)

*A Prefect pipeline demonstrating surrogate optimisation for crop system simulation data.*

# Table of contents

- [Surrogate Optimisation Example](#surrogate-optimisation-example)
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [The Prefect pipeline](#the-prefect-pipeline)
- [Python Environment](#python-environment)
  - [MLOps](#mlops)

# Introduction

The purpose of this project is to provide a simple demonstration of how to construct a Prefect pipeline, with MLOps integration, for performing surrogate optimisation of a crop system simulation model.

[Refer to the PCSE documentation.](https://pcse.readthedocs.io/en/stable/)

# The Prefect pipeline

[Prefect has been included to orchestrate the surrogate optimisation pipeline.](https://www.prefect.io/)

The pipeline is composed of the following steps:

1. Train a Gaussian process surrogate model.
2. Optimise the surrogate model using a blackbox optimisation algorithm.
3. Exploratory data analysis using Pandas Profiler.

# Python Environment

[Python dependencies are specified in this requirements.txt file.](services/python/requirements.txt). 

These dependencies are installed during the build process for the following Docker image: ghcr.io/jbris/prefect-surrogate-models:1.0.0

Execute the following command to pull the image: *docker pull ghcr.io/jbris/prefect-surrogate-models:1.0.0*

## MLOps

* [A Docker compose file has been provided to launch an MLOps stack.](docker-compose.yml)
* [See the .env file for Docker environment variables.](.env.local)
* [The docker_up.sh script can be executed to launch the Docker services.](scripts/docker_up.sh)
* [MLFlow is available for experiment tracking.](https://mlflow.org/)
* [MinIO is available for storing experiment artifacts.](https://min.io/)
