import logging
import os


class Logger:
    def __init__(self, cfg) -> None:
        self.logRootDir = cfg.logSaveRootDir
        self.modelName = cfg.modelName
        self.loggerSavePath = os.path.join(cfg.logSaveRootDir, cfg.modelName, 'log.txt')
        if not os.path.exists(os.path.join(cfg.logSaveRootDir, cfg.modelName)):
            os.makedirs(os.path.join(cfg.logSaveRootDir, cfg.modelName))
        self.logger = logging.getLogger(cfg.modelName)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(self.loggerSavePath)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_string(self, string: str):
        self.logger.info(string)
        print(string)
