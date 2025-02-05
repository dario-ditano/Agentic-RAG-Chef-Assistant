import os
from dotenv import load_dotenv
from src.utils.common_utils import get_base_dir

# Recupera le chiavi API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
COLLECTION_NAME = "recipes"  # Nome della collection su Qdrant

def load_environment():
    """
    Load environment variables from a .env configuration file.

    This function constructs the absolute path of the configuration (.env) file
    named ``config.env``, located one directory up and inside the ``config``
    directory relative to the base directory. It then loads the environment
    variables defined in this file into the application using ``load_dotenv``. If
    the file does not exist or the variables are not set, respective defaults or
    errors might occur downstream based on application handling.

    :return: The result of the ``load_dotenv`` function indicating whether the
        configuration file was successfully loaded or not.
    :rtype: bool
    """
    # Costruisci il percorso assoluto di config.env
    dotenv_path = os.path.join(get_base_dir(), "..\config", "config.env")

    # Carica il file .env
    return load_dotenv(dotenv_path)


