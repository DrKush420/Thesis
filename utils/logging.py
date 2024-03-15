import os
import shutil
import socket
import logging
from datetime import datetime


def prepare_logdir(logdir, config_path):
    """
    Generate new subdirectory structure for logging this training session
    Ensure the following folder structure is created:
    logdir
        <machine>_<date_time>_<network_cfg_file_name>
            tb
                <tensorboard_log_files>
            checkpoints
                <checkpoints>
            <network_cfg_file_name>.py
            std.log
    """
    # Generate new subdirectory structure for logging this training session
    dir_name = socket.gethostname() + datetime.now().strftime("_%B%d_%H_%M_%S_") + os.path.splitext(os.path.basename(config_path))[0]
    dir_name = os.path.join(logdir, dir_name)
    tb_dir_name = os.path.join(dir_name, 'tb')
    checkpoints_dir_name = os.path.join(dir_name, 'checkpoints')
    os.makedirs(tb_dir_name)
    os.makedirs(checkpoints_dir_name)
    shutil.copy(config_path, dir_name)

    # log stdout to file too
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(dir_name, "std.log")),
            logging.StreamHandler()
        ]
    )

    return tb_dir_name, checkpoints_dir_name
