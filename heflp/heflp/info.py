SUPPORT_SCHEMES = ['basic', 'flashe', 'flashev2', 'ckks', 'bfv']
SUPPORT_FRAMEWORKS = ['pytorch', 'tensorflow']
EVALUATION_MODE=True # If true, another logger named 'heflp-eval' will be enabled to log performance metrics for evaluation.

# Logger configuration
DEFAULT_LOG_NAME = 'heflp'
DEFAULT_EVAL_LOG_NAME = 'heflp-eval'
DEFAULT_LOG_LEVEL = 'DEBUG'
DEFAULT_LOG_FILE = 'heflp.log'
DEFAULT_EVAL_LOG_FILE = 'heflp-eval.log'
DEFAULT_FORMAT = "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
