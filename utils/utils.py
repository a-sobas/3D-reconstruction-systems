import os
import torch


def save_model(model, save_dst, model_name):
    if not os.path.isdir(save_dst):
        os.makedirs(save_dst)

    torch.save(model.state_dict(), os.path.join(save_dst, model_name))


def load_model(model, dst, model_name):
    state_path = os.path.join(dst, model_name)

    assert os.path.isfile(state_path), f'{state_path} is not valid file'

    model.load_state_dict(torch.load(state_path))

    return model


def get_datasets_path():
    actual_path = os.getcwd()

    return os.path.join(actual_path, 'preprocessed_datasets')

