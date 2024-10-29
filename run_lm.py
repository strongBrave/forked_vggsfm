import os
from loguru import logger


with_ba = "python linemod_estimation.py"
without_ba = "python without_BA.py"

commands = [without_ba, with_ba]

for command in commands:
    os.system(command)
    logger.success(f"Succesfully complete command {command}")

    