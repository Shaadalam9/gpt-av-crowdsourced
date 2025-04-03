import pandas as pd
import os
import math
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import common
from custom_logger import CustomLogger
from logmod import logs
from ollama import OllamaClient

# Set up logging and plotly template
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)
template = common.get_configs('plotly_template')

client = OllamaClient()


class Analysis:
    def __init__(self):
        pass

    def process_csv_files(self, output=common.get_configs("output")):
        # Define subfolders to check
        sub_folders = ["with_memory", "without_memory"]

        # Process each subfolder
        for sub_folder in sub_folders:
            folder_path = os.path.join(output, sub_folder)
            if not os.path.exists(folder_path):
                logger.info(f"Folder not found: {folder_path}")
                continue

            # Create an "analysed" subfolder within the current folder if it doesn't exist
            analysed_folder = os.path.join(folder_path, "analysed")
            if not os.path.exists(analysed_folder):
                os.makedirs(analysed_folder)

            # Iterate through files in the current subfolder
            for filename in os.listdir(folder_path):
                if filename.startswith("output_") and filename.endswith(".csv"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        df = pd.read_csv(file_path)
                        # Identify the columns to analyse (all except "image")
                        columns_to_analyze = [col for col in df.columns if col != "image"]

                        # Iterate over each row and each text column
                        for index, row in df.iterrows():
                            for col in columns_to_analyze:
                                text = row[col]

                                if text == "":
                                    rating = math.nan

                                else:
                                    # Generate a rating using the client API (one call per cell)
                                    rating = client.generate_simple_text(sentence=text, model="deepseek-r1:1.5b")
                                # Save the rating in the same column (this keeps the same column header)
                                df.at[index, col] = rating

                        # Define the output file path in the "analysed" folder using the same filename
                        output_file_path = os.path.join(analysed_folder, filename)
                        df.to_csv(output_file_path, index=False)
                        logger.info(f"Saved analysed file: {output_file_path}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")

    def plot_with_memory(self):
        pass


# Example usage
if __name__ == "__main__":
    analysis = Analysis()
    analysis.process_csv_files()
