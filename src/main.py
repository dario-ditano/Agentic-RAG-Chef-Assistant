from src.data.json_handler import load_and_process_json
from src.data.dataframe_utils import keep_first_n_rows
from src.qdrant.qdrant_utils import initialize_qdrant, index_recipes
from src.agents.recipe_agent import create_agent
from src.config.config import load_environment
from utils.common_utils import get_base_dir
import os

if __name__ == "__main__":

    load_environment()

    JSON_PATH = os.path.join(get_base_dir(), "..\..\data", "recipes.json")

    # Carica e processa i dati
    df = load_and_process_json(JSON_PATH)
    df_filtered = keep_first_n_rows(20, df)

    # Inizializza Qdrant e indicizza le ricette
    client = initialize_qdrant()
    index_recipes(client, df_filtered)

    # Crea l'agente
    agent = create_agent(client)

    # Ottieni una ricetta
    user_query = "Ho formaggio Stilton e pepe. Cosa posso cucinare?"
    response = agent.invoke({"input": user_query})

    print("\n🔹 RISULTATI:")
    print(response)

    client.close()
