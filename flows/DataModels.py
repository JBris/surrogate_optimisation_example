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
    parameters_file: str
    outputs_file: str
