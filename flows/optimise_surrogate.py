#!/usr/bin/env python

######################################
# Imports
######################################

# External

import hydra
from joblib import dump as optimiser_dump 
import kaleido
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
from .DataModels import DataModel, GpModel, OptimisationModel, ExperimentModel

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

def begin_experiment(
    task: str, experiment_prefix: str, tracking_uri: str
):
    """
    Begin the experiment session.

    Args:   
        task (str):
            The name of the current task for the experiment.
        experiment_prefix (str):
            The prefix for the experiment name.
        tracking_uri (str):
            The experiment tracking URI.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = f"{experiment_prefix}_{task}"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.set_tag("task", task)

@task
def fit_model(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    outputs: list[str],
    device: torch.cuda.device,
    num_latents: int,
    num_epochs: int,
    lr: float,
    experiment_prefix: str,
    tracking_uri: str
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
        outputs (list[str]):
            A list of output names.
        device (torch.cuda.device):
            The tensor and model device.
        num_latents (int):
            The number of latent variables.
        num_epochs (int):
            The number of training epochs.
        lr (float):
            The training learning rate.
        experiment_prefix (str):
            The prefix for the experiment name.
        tracking_uri (str):
            The experiment tracking URI.

    Returns:
        tuple[MultitaskVariationalGPModel, gpytorch.likelihoods.MultitaskGaussianLikelihood]:
            The trained model and likelihood.
    """
    begin_experiment("fit_model", experiment_prefix, tracking_uri)
    
    mlflow.log_param("num_latents", num_latents)
    model = MultitaskVariationalGPModel(
        n_col=X.shape[-1], num_latents=num_latents, num_tasks=Y.shape[-1]
    )
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Y.shape[-1])

    model.train().to(device)
    likelihood.train().to(device)

    mlflow.log_param("lr", lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Y.size(0))

    mlflow.log_param("num_epochs", num_epochs)
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

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(X))
        mean = predictions.mean.cpu().numpy()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.cpu().numpy(), upper.cpu().numpy()

    mslls = gpytorch.metrics.mean_standardized_log_loss(predictions, Y).cpu().detach().numpy() 
    mses = gpytorch.metrics.mean_squared_error(predictions, Y).cpu().numpy()
    maes = gpytorch.metrics.mean_absolute_error(predictions, Y).cpu().numpy()
    coverage_errors = gpytorch.metrics.quantile_coverage_error(predictions, Y).cpu().numpy()

    for i, col in enumerate(outputs):
        mlflow.log_metric(f"msll_{col}",  mslls[i])
        mlflow.log_metric(f"mse_{col}",  mses[i])
        mlflow.log_metric(f"mae_{col}",  maes[i])
        mlflow.log_metric(f"empirical_coverage_rate_{col}",  coverage_errors[i])

    for row, col in np.ndindex(Y.shape):
        if row % int(Y.shape[0] * 0.05) != 0:
            continue
        
        mlflow.log_metric(f"Actual {outputs[col]}", Y[row, col], step = row)
        mlflow.log_metric(f"Predicted mean {outputs[col]}", mean[row, col], step = row)
        mlflow.log_metric(f"Predicted lower {outputs[col]}", lower[row, col], step = row)
        mlflow.log_metric(f"Predicted upper {outputs[col]}", upper[row, col], step = row)

    mlflow.end_run()
    return model, likelihood


def objective(
    trial,
    model: MultitaskVariationalGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    scaler: MinMaxScaler,
    parameters_df_bounds: pd.DataFrame,
    device: torch.cuda.device,
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

    samples_scaled = scaler.transform(pd.DataFrame.from_records([sample]))

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
    scaler: MinMaxScaler,
    parameters_df: pd.DataFrame,
    outputs_df: pd.DataFrame,
    n_trials: int,
    n_jobs: int,
    params: dict,
    device: torch.cuda.device,
    data_dir: str,
    output_dir: str,
    experiment_prefix: str,
    tracking_uri: str
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
        data_dir (str):
            The data directory.
        output_dir (str):
            The data output directory.
        experiment_prefix (str):
            The prefix for the experiment name.
        tracking_uri (str):
            The experiment tracking URI.
    """
    begin_experiment("optimise_model", experiment_prefix, tracking_uri)

    parameters_df_bounds = parameters_df.agg(["min", "max"])
    directions = list(np.repeat("minimize", outputs_df.shape[1]))
    sampler = TPESampler(**params)
    for k in params:
        mlflow.log_param(k, params.get(k))
    
    study = optuna.create_study(sampler=sampler, directions=directions)

    mlflow.log_param("n_trials", n_trials)
    study.optimize(
        lambda trial: objective(
            trial, model, likelihood, scaler, parameters_df_bounds, device
        ),
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=(n_jobs == 1),
    )

    outdir = str(Path(data_dir, output_dir))

    trials_df = study.trials_dataframe()
    trials_out = str(Path(outdir, "optimisation_results.csv"))
    trials_df.to_csv(trials_out, index = False)
    mlflow.log_artifact(trials_out)

    optimiser_file = str(Path(outdir, "optimiser.pkl") )
    optimiser_dump(study, optimiser_file)
    mlflow.log_artifact(optimiser_file)

    def __plot_results(plot_func, plot_name: str, i: int, col: str):
        img_file = str(Path(outdir, f"{plot_name}_{col}.png"))
        plot_func(study, target=lambda t: t.values[i], target_name=col).write_image(
            img_file
        )
        mlflow.log_artifact(img_file)
        
    for i, col in enumerate(outputs_df.columns):
        __plot_results(optuna.visualization.plot_edf, "edf", i, col)
        __plot_results(
            optuna.visualization.plot_optimization_history,
            "optimization_history",
            i,
            col,
        )
        __plot_results(
            optuna.visualization.plot_parallel_coordinate, "parallel_coordinate", i, col
        )
        __plot_results(
            optuna.visualization.plot_param_importances, "param_importances", i, col
        )
        __plot_results(optuna.visualization.plot_slice, "slice", i, col)

    mlflow.end_run()

@flow(
    name="Optimise Surrogate",
    description="Optimise a surrogate model.",
    task_runner=SequentialTaskRunner(),
)
def optimise_surrogate_flow(
    data_model: DataModel, gp_model: GpModel, optimisation_model: OptimisationModel,
    experiment_model: ExperimentModel
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
        experiment_model (OptimisationModel):
            The experiment data model.
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
        X, Y, outputs_df.columns, device, gp_model.num_latents, gp_model.num_epochs, 
        gp_model.lr, experiment_model.experiment_prefix, experiment_model.tracking_uri
    )

    optimise_model(
        model,
        likelihood,
        scaler,
        parameters_df,
        outputs_df,
        optimisation_model.n_trials,
        optimisation_model.n_jobs,
        optimisation_model.params,
        device,
        data_model.data_dir,
        data_model.output_dir,
        experiment_model.experiment_prefix,
        experiment_model.tracking_uri
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
    EXPERIMENT_CONFIG = config["experiment"]
    GP_CONFIG = config["gp"]
    OPTIMISATION_CONFIG = config["optimise"]

    data_model = DataModel(**DATA_CONFIG)
    experiment_model = ExperimentModel(**EXPERIMENT_CONFIG)
    gp_model = GpModel(**GP_CONFIG)
    optimisation_model = OptimisationModel(**OPTIMISATION_CONFIG)

    optimise_surrogate_flow(
        data_model, gp_model, optimisation_model, experiment_model
    )


if __name__ == "__main__":
    main()
