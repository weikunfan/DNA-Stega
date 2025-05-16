# -*- coding: utf-8 -*-

import logging
import os


class Logger(object):
    def __init__(self, log_file):
        self.logger    = logging.getLogger()
        self.formatter = logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        # 添加路径检查
        if os.path.isdir(log_file):
            raise IsADirectoryError(f"Log file path is a directory: {log_file}")
        
        # 确保父目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        try:
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(logging.INFO)
            fh.setFormatter(self.formatter)
            self.logger.addHandler(fh)

            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(self.formatter)
            self.logger.addHandler(sh)
        except Exception as e:
            print(f"Error creating log file at {log_file}: {str(e)}")
            raise

    def info(self, text):
        self.logger.info(text)
