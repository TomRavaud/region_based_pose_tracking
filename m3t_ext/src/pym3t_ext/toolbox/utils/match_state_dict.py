# Third-party libraries
import torch


def match_state_dict(state_dict: dict, model: torch.nn.Module) -> dict:
    """Extract the state_dict of the model from an other state_dict by matching
    their keys.

    Args:
        state_dict (dict): The state_dict from which to extract the model's
            state_dict.
        model (torch.nn.Module): The model for which to extract the state_dict.

    Returns:
        dict: The state_dict of the model.
    """
    model_state_dict = model.state_dict()
    new_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state_dict
    }

    model_state_dict.update(new_state_dict)

    return model_state_dict
