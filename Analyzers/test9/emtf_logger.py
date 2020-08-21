# Copied from https://docs.python.org/2/howto/logging.html
def get_logger():
  import os
  import logging

  # create logger
  cwd = os.path.basename(os.getcwd())
  logger = logging.getLogger(cwd)
  logger.setLevel(logging.DEBUG)

  # create file handler which logs even debug messages
  fh = logging.FileHandler(cwd+'.log')
  fh.setLevel(logging.DEBUG)

  # create console handler with a higher log level
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  # create formatter and add it to the handlers
  #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s')
  fh.setFormatter(formatter)
  formatter = logging.Formatter('[%(levelname)-8s] %(message)s')
  ch.setFormatter(formatter)

  # add the handlers to the logger
  if not len(logger.handlers):
    logger.addHandler(fh)
    logger.addHandler(ch)
  return logger
