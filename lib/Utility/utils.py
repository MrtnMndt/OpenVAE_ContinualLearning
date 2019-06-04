import torch
import shutil
import os


def save_checkpoint(state, is_best, file_path, file_name='checkpoint.pth.tar'):
    """
    Saves the current state of the model. Does a copy of the file
    in case the model performed better than previously.

    Parameters:
        state (dict): Includes optimizer and model state dictionaries.
        is_best (bool): True if model is best performing model.
        file_path (str): Path to save the file.
        file_name (str): File name with extension (default: checkpoint.pth.tar).
    """

    save_path = os.path.join(file_path, file_name)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(file_path, 'model_best.pth.tar'))


def save_task_checkpoint(file_path, task_num):
    """
    Saves the current state of the model for a given task by copying existing checkpoint created by the
    save_checkpoint function.

    Parameters:
        file_path (str): Path to save the file,
        task_num (int): Number of task increment.
    """
    save_path = os.path.join(file_path, 'checkpoint_task_' + str(task_num) + '.pth.tar')
    shutil.copyfile(os.path.join(file_path, 'checkpoint.pth.tar'), save_path)
