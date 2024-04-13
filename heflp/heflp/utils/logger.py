import logging
import os
import sys
from heflp.info import (
    DEFAULT_EVAL_LOG_NAME,
    DEFAULT_EVAL_LOG_FILE,
    DEFAULT_FORMAT,
    DEFAULT_LOG_FILE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_NAME,
)
import datetime
import json
from typing import Dict, Callable,Optional

def create_id_by_timestamp(include_ms=True):
    if include_ms:
        tstamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    else:
        tstamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return tstamp

def getLogger(name=DEFAULT_LOG_NAME, loglevel=DEFAULT_LOG_LEVEL, logfile=DEFAULT_LOG_FILE):
    logger = logging.getLogger(name=name)

    if logger.handlers:
        return logger # return the logger if it already exists
    else:
        loglevel = getattr(logging, loglevel.upper(), logging.DEBUG)
        logger.setLevel(loglevel)
        fmt = DEFAULT_FORMAT
        # fmt_date = '%Y-%m-%dT%T%Z'
        fmt_date = None
        formatter = logging.Formatter(fmt, fmt_date)
        if isinstance(logfile, str):
            handler2 = logging.FileHandler(logfile) # save logs to file
            handler2.setFormatter(formatter)
            logger.addHandler(handler2)
        handler = logging.StreamHandler() # print logs
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(f"LoggerInit | Logs are stored in {logfile}")

        if logger.name == 'root':
            logger.warning('Running: %s %s',
                        os.path.basename(sys.argv[0]),
                        ' '.join(sys.argv[1:]))
        return logger

def getEvalLogger(logfile=DEFAULT_EVAL_LOG_FILE):
    return getLogger(name=DEFAULT_EVAL_LOG_NAME, loglevel='DEBUG', logfile=logfile)



class LogParser():
    def __init__(self, custom_func_dict:Optional[Dict[str, Callable]]=None) -> None:
        self.func_dict = {
            "Meta": lambda x : json.loads(x.replace("\'", "\"")),
            "Time overhead": lambda x : json.loads(x.replace("\'", "\"")),
        }
        if custom_func_dict != None:
            self.func_dict.update(custom_func_dict)

    def parse_eval_log_line(self, line:str, default_callback=lambda x:x):
        fields = line.split(' | ', 4)
        log_info = fields[0]
        filename = fields[1]
        content = fields[-1]
        if len(fields) > 3:
            title = fields[2]
            rst = self.func_dict.get(title, default_callback)(content)
        else:
            title = "notitle"
            rst = default_callback(content)
        # print(title, content)
        return log_info, filename, title, rst

    def parse_eval_log(self, log_file):
        try:
            with open(log_file, "r") as log_file:
                # Read and process each line in the log file
                l = 0
                time_overhead = dict()
                for line in log_file:
                    _, _, title, rst = self.parse_eval_log_line(line)
                    if title == "Meta":
                        meta = rst
                    elif title == 'Time overhead':
                        l += 1
                        for k,v in rst.items():
                            time_overhead[k] = time_overhead.get(k, 0) + v
                for k, v in time_overhead.items():
                    time_overhead[k] = v/l
                    # print(line.strip())  # Print the line after stripping whitespace
                return time_overhead
        except FileNotFoundError:
            print(f"Log file '{log_file}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")