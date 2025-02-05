from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from src.config.config import COLLECTION_NAME, QDRANT_HOST


def initialize_qdrant() -> QdrantClient:
    """
    Initialize a QdrantClient instance and set up the specified collection.

    This function establishes a connection with the Qdrant service and ensures
    the specified collection is properly configured. If the collection already
    exists, it is deleted and recreated with the specified vector parameters,
    including vector size and distance metric.

    :raises RuntimeError: If there is a failure in initializing or configuring
                          the Qdrant client.

    :return: A configured instance of the QdrantClient.
    :rtype: QdrantClient
    """
    client = QdrantClient(QDRANT_HOST)
    client.delete_collection(collection_name=COLLECTION_NAME)  # Pulisci i dati esistenti
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    return client


def index_recipes(client: QdrantClient, recipes_df):
    """
    Indexes recipes into a vector database by generating embeddings for each recipe's
    ingredients and upserting them into the specified collection.

    This function uses OpenAI embeddings to convert recipes' ingredients into
    numerical vectors. It constructs a list of `PointStruct` objects, where each
    object contains a unique identifier, its corresponding embedding vector, and
    a payload representation of the recipe. These `PointStruct` objects are then
    upserted into a specified collection within the QdrantClient vector database.

    :param client: The QdrantClient instance used for managing the vector database.
    :type client: QdrantClient
    :param recipes_df: A dataframe containing recipes, where each row represents
        a recipe and each column represents a property of the recipe such as
        ingredients.
    :type recipes_df: pandas.DataFrame
    :return: None
    """
    embeddings = OpenAIEmbeddings()
    points = [
        PointStruct(id=idx, vector=embeddings.embed_query(row["ingredients"]), payload=row.to_dict())
        for idx, row in recipes_df.iterrows()
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indicizzate {len(recipes_df)} ricette nel database vettoriale.")


def find_recipes(client: QdrantClient, user_ingredients: str, top_k: int = 3):
    """
    Find recipes based on user's provided ingredients using semantic search.

    This function utilizes a vector similarity search to discover recipes that are
    most relevant to the input ingredients provided by the user. It employs OpenAI's
    embedding model to convert ingredient descriptions into a query vector, which is
    then used to perform a search in Qdrant's vector database. The resulting recipes
    are returned in descending order of relevance up to a maximum limit specified by
    the user.

    :param client: Instance of QdrantClient used to interact with the vector database.
    :type client: QdrantClient
    :param user_ingredients: A string of ingredients provided by the user to query the recipes.
    :type user_ingredients: str
    :param top_k: Optional integer specifying the maximum number of recipes to return,
                  default is 3.
    :type top_k: int
    :return: A list of recipe dictionaries, each containing recipe details like title,
             ingredients, and instructions. If no results are found, a default message
             is returned indicating no recipes were found.
    :rtype: list of dict
    """
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(user_ingredients)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    if not results:
        return [{"title": "Nessuna ricetta trovata", "ingredients": "",
                 "instructions": "Non ho trovato ricette con questi ingredienti."}]

    return [hit.payload for hit in results]
