#!/usr/bin/env python

######################################
# Imports
######################################

# External

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna
from optuna.samplers import TPESampler
import pandas as pd
from pathlib import Path
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.preprocessing import MinMaxScaler
import torch
import gpytorch

# Internal
from .DataModels import DataModel, GpModel, OptimisationModel

######################################
# Functions
######################################


class MultitaskVariationalGPModel(gpytorch.models.ApproximateGP):
    """
    A multitask variational Gaussian process model.

    Args:
        gpytorch (_type_):
            An approximate, variational Gaussian process.
    """

    def __init__(self, n_col: int, num_latents: int, num_tasks: int):
        """
        Constructor.

        Args:
            n_col (int):
                The number of columns.
            num_latents (int):
                The number of latent variables.
            num_tasks (int):
                The number of tasks for multitask learning.
        """
        inducing_points = torch.rand(num_latents, n_col, n_col)

        variational_distribution = (
            gpytorch.variational.MeanFieldVariationalDistribution(
                inducing_points.size(-2), batch_shape=torch.Size([num_latents])
            )
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([num_latents])
        )
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=2.5, batch_shape=torch.Size([num_latents]), ard_num_dims=n_col
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        The forward pass.

        Args:
            x (torch.Tensor):
                The input data.

        Returns:
             gpytorch.distributions.MultivariateNormal:
                A multivariate normal random variable.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@task
def load_data(
    data_dir: str,
    input_dir: str,
    parameters_file: str,
    outputs_file: str,
    inputs: list[str],
    outputs: list[str],
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
        inputs (list[str]):
            The list of input columns.
        outputs (list[str]):
            The list of output columns.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            The parameters and outputs dataframes.
    """
    parameters_file = Path(data_dir, input_dir, parameters_file)
    outputs_file = Path(data_dir, input_dir, outputs_file)

    parameters_df = pd.read_csv(str(parameters_file))
    outputs_df = pd.read_csv(str(outputs_file))

    parameters_df = parameters_df[inputs]
    outputs_df = outputs_df[outputs]

    return parameters_df, outputs_df


@task
def to_tensor(
    parameters_df: pd.DataFrame, outputs_df: pd.DataFrame, device: torch.cuda.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dataframes to tensors.

    Args:
        parameters_df (pd.DataFrame):
            The parameter dataframe.
        outputs_df (pd.DataFrame):
            The output dataframe.
        device (torch.cuda.device):
            The tensor device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            The parameter and output tensors.
    """
    scaler = MinMaxScaler()
    scaler.fit(parameters_df)

    X = torch.from_numpy(scaler.transform(parameters_df)).float().to(device)
    Y = torch.from_numpy(outputs_df.values).float().to(device)

    return X, Y, scaler


@task
def fit_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    device: torch.cuda.device,
    num_latents: int,
    num_epochs: int,
    lr: float,
) -> tuple[
    MultitaskVariationalGPModel, gpytorch.likelihoods.MultitaskGaussianLikelihood
]:
    """
    Train a Gaussian process model.

    Args:
        X (pd.DataFrame):
            The input matrix.
        Y (pd.DataFrame):
            The output matrix.
        device (torch.cuda.device):
            The tensor and model device.
        num_latents (int):
            The number of latent variables.
        num_epochs (int):
            The number of training epochs.
        lr (float):
            The training learning rate.

    Returns:
        tuple[MultitaskVariationalGPModel, gpytorch.likelihoods.MultitaskGaussianLikelihood]:
            The trained model and likelihood.
    """
    model = MultitaskVariationalGPModel(
        n_col=X.shape[-1], num_latents=num_latents, num_tasks=Y.shape[-1]
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y.shape[-1])

    model.train().to(device)
    likelihood.train().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Y.size(0))

    epoch_interval = int(num_epochs / 10)
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y)
        loss.backward()
        optimizer.step()

        if i % epoch_interval == 0:
            print(f"Epochs: {i}  Loss: {loss.item()}")
            mlflow.log_metric("loss", loss.item(), step=i)

    print(f"Final Loss: {loss.item()}")

    model.eval()
    likelihood.eval()

    return model, likelihood

def objective(
        trial, model: MultitaskVariationalGPModel, 
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood, 
        scaler: MinMaxScaler, parameters_df_bounds: pd.DataFrame,
        device: torch.cuda.device
) -> tuple[any]:
    """
    The optimisation objective function.

    Args:
        trial (_type_): 
            The trial object.
        model (MultitaskVariationalGPModel): 
            The surrogate model.
        likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood): 
            The Gaussian process likelihood.
        scaler (MinMaxScaler): 
            The data scaler.
        parameters_df_bounds (pd.DataFrame): 
            The bounds of the input parameter dataframe.
        device (torch.cuda.device):
            The tensor device.

    Returns:
        tuple[any]: _description_
    """
    sample = {}
    for col in parameters_df_bounds.columns:
        min_val = parameters_df_bounds.iloc[0][col]
        max_val = parameters_df_bounds.iloc[-1][col]
        sample[col] = trial.suggest_float(col, min_val, max_val) 

    samples_scaled = scaler.transform(
        pd.DataFrame.from_records([ sample ])
    )

    samples_tensor = torch.from_numpy(samples_scaled).float().to(device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(samples_tensor))
        mean = predictions.mean.cpu().numpy().reshape(-1)
        mean = tuple(mean)
    
    return mean

@task
def optimise_model(
    model: MultitaskVariationalGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    scaler: MinMaxScaler, parameters_df: pd.DataFrame, outputs_df: pd.DataFrame,
    n_trials: int, n_jobs: int, params: dict, device: torch.cuda.device
) -> None:
    """
    Optimise the surrogate model.

    Args:
        model (MultitaskVariationalGPModel): 
            The surrogate model.
        likelihood (gpytorch.likelihoods.MultitaskGaussianLikelihood): 
            The Gaussian process likelihood.
        scaler (MinMaxScaler): 
            The data scaler.
        parameters_df (pd.DataFrame): 
            The input parameter dataframe.
        outputs_df (pd.DataFrame): 
            The output dataframe.
        n_trials (int): 
            The number of optimisation trials.
        n_jobs (int): 
            The number of trials to run in parallel.
        params (dict): 
            The optimisation sampler parameter dictionary.
        device (torch.cuda.device):
            The tensor device.
    """
    parameters_df_bounds = parameters_df.agg(["min", "max"])
    directions = list(np.repeat("minimize", outputs_df.shape[1]))
    sampler = TPESampler(**params)
    study = optuna.create_study(sampler=sampler, directions=directions)

    study.optimize(
        lambda trial: objective(trial, model, likelihood, scaler, parameters_df_bounds, device), 
        n_trials = n_trials, n_jobs = 1, gc_after_trial = True, 
        show_progress_bar = (n_jobs == 1)
    )

@flow(
    name="Optimise Surrogate",
    description="Optimise a surrogate model.",
    task_runner=SequentialTaskRunner(),
)
def optimise_surrogate_flow(
    data_model: DataModel, gp_model: GpModel, optimisation_model: OptimisationModel
) -> None:
    """
    The data processing flow.

    Args:
        data_model (DataModel):
            The input data data model.
        gp_model (GpModel):
            The Gaussian process data model.
        optimisation_model (OptimisationModel):
            The optimisation data model.
    """
    parameters_df, outputs_df = load_data(
        data_model.data_dir,
        data_model.input_dir,
        data_model.parameters_file,
        data_model.outputs_file,
        data_model.inputs,
        data_model.outputs,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X, Y, scaler = to_tensor(parameters_df, outputs_df, device)

    model, likelihood = fit_model(
        X, Y, device, gp_model.num_latents, gp_model.num_epochs, gp_model.lr
    )

    optimise_model(
        model, likelihood, scaler, parameters_df, outputs_df, 
        optimisation_model.n_trials, optimisation_model.n_jobs, 
        optimisation_model.params, device
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
    config = OmegaConf.to_container(config)
    DATA_CONFIG = config["data"]
    GP_CONFIG = config["gp"]
    OPTIMISE_CONFIG = config["optimise"]

    data_model = DataModel(**DATA_CONFIG)
    gp_model = GpModel(**GP_CONFIG)
    optimisation_model = OptimisationModel(**OPTIMISE_CONFIG)

    optimise_surrogate_flow(data_model, gp_model, optimisation_model)


if __name__ == "__main__":
    main()
