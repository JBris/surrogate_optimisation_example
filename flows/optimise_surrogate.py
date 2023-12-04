#!/usr/bin/env python

######################################
# Imports
######################################

# External
import hydra
import mlflow
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

# Internal
from .DataModels import DataModel

######################################
# Functions
######################################


@task
def load_data(
    data_dir: str, input_dir: str, parameters_file: str, outputs_file: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the parameters and outputs dataframes.

    Args:
        data_dir (str):
            The data directory.
        input_dir (str):
            The data input directory.
        parameters_file (str):
            The parameters dataframe filename.
        outputs_file (str):
            The outputs dataframe filename.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            The parameters and outputs dataframes.
    """
    parameters_file = Path(data_dir, input_dir, parameters_file)
    outputs_file = Path(data_dir, input_dir, outputs_file)

    parameters_df = pd.read_csv(str(parameters_file))
    outputs_df = pd.read_csv(str(outputs_file))

    return parameters_df, outputs_df


@flow(
    name="Optimise Surrogate",
    description="Optimise a surrogate model.",
    task_runner=SequentialTaskRunner(),
)
def optimise_surrogate_flow(data_model: DataModel) -> None:
    """
    The data processing flow.

    Args:
        data_model (DataModel):
            The directory data model.

    """
    parameters_df, outputs_df = load_data(
        data_model.data_dir,
        data_model.input_dir,
        data_model.parameters_file,
        data_model.outputs_file,
    )


######################################
# Main
######################################


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    """
    Optimise a surrogate model.

    Args:
        config (DictConfig):
            The main configuration.
    """
    DATA_CONFIG = config["data"]
    data_model = DataModel(**DATA_CONFIG)

    optimise_surrogate_flow(data_model)


if __name__ == "__main__":
    main()
