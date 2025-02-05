import json
import pandas as pd


def load_and_process_json(json_file_path: str) -> pd.DataFrame:
    """
    Loads a JSON file containing recipe data, processes it to extract relevant
    fields, and returns a pandas DataFrame with the processed data.

    :param json_file_path: Path to the JSON file containing the recipe data.
    :type json_file_path: str
    :return: A pandas DataFrame containing processed recipe data, including the
        title, concatenated ingredient list, and instructions for each recipe.
    :rtype: pd.DataFrame
    """
    with open(json_file_path, "r") as f:
        recipes = json.load(f)

    processed_recipes = [
        {
            "title": recipe_data["title"],
            "ingredients": ", ".join(recipe_data["ingredients"]),
            "instructions": recipe_data["instructions"]
        }
        for recipe_id, recipe_data in recipes.items()
    ]
    return pd.DataFrame(processed_recipes)
