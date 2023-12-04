from pydantic import BaseModel


class DataModel(BaseModel):
    """
    The data model.

    Args:
        BaseModel (_type_):
            The Base model class.
    """
    data_dir: str
    input_dir: str
    output_dir: str
    parameters_file: str
    outputs_file: str
    inputs: list[str]
    outputs: list[str]


class GpModel(BaseModel):
    """
    The Gaussian process model.

    Args:
        BaseModel (_type_):
            The Base model class.
    """
    num_latents: int
    num_epochs: int
    lr: float


class OptimisationModel(BaseModel):
    """
    The optimisation model.

    Args:
        BaseModel (_type_):
            The Base model class.
    """
    n_trials: int
    n_jobs: int
    params: dict