import os
import glob
import math
import shutil
import pandas as pd
import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: F401
import common
from custom_logger import CustomLogger
from logmod import logs
from ollama import OllamaClient

# Paths
output_path = common.get_configs("output")
data_path = common.get_configs("data")


class Analysis:
    """
    A class for processing CSV files, averaging LLM results, 
    and plotting comparisons between eHMI means and LLM scores.
    """

    def __init__(self):
        # Set up logging and plotly template
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)
        self.template = common.get_configs('plotly_template')

        # Initialize the client for generating text ratings
        self.client = OllamaClient()

        # Constants (moved from module-level)
        self.save_png = True
        self.save_eps = True
        self.base_height_per_row = 20  # Adjust as needed
        self.flag_size = 12
        self.text_size = 12
        self.scale = 1  # scale=3 hangs often

    def save_plotly_figure(self, fig, filename, width=1600, height=900, save_final=True):
        """Saves a Plotly figure as HTML, PNG, and EPS formats.

        Args:
            fig (plotly.graph_objs.Figure): Plotly figure object.
            filename (str): Name of the file (without extension) to save.
            width (int, optional): Width of the image in pixels. Defaults to 1600.
            height (int, optional): Height of the image in pixels. Defaults to 900.
            save_final (bool, optional): Whether to save a copy to the final folder.
        """
        output_folder = "_output"
        output_final = "figures"
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_final, exist_ok=True)

        self.logger.info(f"Saving html file for {filename}.")
        py.offline.plot(fig, filename=os.path.join(output_folder, filename + ".html"))
        if save_final:
            py.offline.plot(fig, filename=os.path.join(output_final, filename + ".html"), auto_open=False)

        try:
            if self.save_png:
                self.logger.info(f"Saving png file for {filename}.")
                fig.write_image(os.path.join(output_folder, filename + ".png"),
                                width=width, height=height, scale=self.scale)
                if save_final:
                    shutil.copy(os.path.join(output_folder, filename + ".png"),
                                os.path.join(output_final, filename + ".png"))
            if self.save_eps:
                self.logger.info(f"Saving eps file for {filename}.")
                fig.write_image(os.path.join(output_folder, filename + ".eps"),
                                width=width, height=height)
                if save_final:
                    shutil.copy(os.path.join(output_folder, filename + ".eps"),
                                os.path.join(output_final, filename + ".eps"))
        except ValueError:
            self.logger.error(f"Value error raised when attempting to save image {filename}.")

    def process_csv_files(self, output=common.get_configs("output")):
        """
        Process CSV files in the defined subfolders ('with_memory' and 'without_memory')
        by generating ratings for each text cell (excluding the 'image' column) and saving 
        the results in an 'analysed' subfolder.
        """
        sub_folders = ["with_memory", "without_memory"]

        for sub_folder in sub_folders:
            folder_path = os.path.join(output, sub_folder)
            if not os.path.exists(folder_path):
                self.logger.info(f"Folder not found: {folder_path}")
                continue

            analysed_folder = os.path.join(folder_path, "analysed")
            os.makedirs(analysed_folder, exist_ok=True)
            csv_files = glob.glob(os.path.join(folder_path, "output_*.csv"))

            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    columns_to_analyze = [col for col in df.columns if col != "image"]

                    for index, row in df.iterrows():
                        for col in columns_to_analyze:
                            text = row[col]
                            rating = math.nan if text == "" else self.client.generate_simple_text(
                                sentence=text, model="deepseek-r1:1.5b"
                            )
                            df.at[index, col] = rating

                    output_file_path = os.path.join(analysed_folder, os.path.basename(file_path))
                    df.to_csv(output_file_path, index=False)
                    self.logger.info(f"Saved analysed file: {output_file_path}")
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")

    def average_llm_results(self, folder_path, image_column="image", output_csv_path=None):
        """
        Aggregate CSV files from a folder, compute the average for each numeric column grouped 
        by the image column, and optionally save the result to a CSV file.
        """
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        dataframes = []

        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
            except Exception as e:
                self.logger.error(f"Error reading {file}: {e}")

        if not dataframes:
            self.logger.error("No CSV files were successfully read.")
            return None

        combined_df = pd.concat(dataframes, ignore_index=True)

        if image_column not in combined_df.columns:
            raise ValueError(f"The column '{image_column}' was not found in the CSV files.")

        numeric_columns = combined_df.select_dtypes(include='number').columns.tolist()
        avg_df = combined_df.groupby(image_column)[numeric_columns].mean().reset_index()

        if output_csv_path:
            avg_df.to_csv(output_csv_path, index=False)

        return avg_df

    def _prepare_merged_df(self, mapping_csv_path, ehmi_csv_path, avg_df):
        """
        Helper function to read and merge mapping and eHMI CSV files with the averaged DataFrame.
        """
        mapping_df = pd.read_csv(mapping_csv_path)
        ehmi_df = pd.read_csv(ehmi_csv_path).rename(columns={"mean": "ehmi_mean"})

        mapping_df["id"] = mapping_df["id"].astype(str)
        avg_df["image"] = avg_df["image"].astype(str).str.replace("image_", "", regex=False)

        mapping_df["text"] = mapping_df["text"].str.upper()
        ehmi_df["eHMI"] = ehmi_df["eHMI"].str.upper()

        merged_df = pd.merge(avg_df, mapping_df, left_on="image", right_on="id", how="inner")
        merged_df = pd.merge(merged_df, ehmi_df, left_on="text", right_on="eHMI", how="inner")
        return merged_df

    def plot_ehmi_vs_llm(self, mapping_csv_path, ehmi_csv_path, avg_df, memory_type):
        """
        Plots the eHMI mean (from eHMIs.csv) versus various LLM score columns (from avg_df)
        for each text category (derived from mapping.csv).
        """
        merged_df = self._prepare_merged_df(mapping_csv_path, ehmi_csv_path, avg_df)

        numeric_columns = [col for col in avg_df.columns if col != "image"]
        fig = go.Figure()

        for col in numeric_columns:
            fig.add_trace(go.Scatter(
                x=merged_df["ehmi_mean"],
                y=merged_df[col],
                mode='markers',
                name=col,
                text=merged_df["text"]
            ))

        fig.update_layout(
            title="Comparison of eHMI Mean vs LLM Score Columns by Text",
            xaxis_title="eHMI Mean",
            yaxis_title="LLM Score",
            legend_title="LLM Score Column"
        )
        self.save_plotly_figure(fig, f"merged_{memory_type}", save_final=False)

    def plot_individual_ehmi_vs_llm(self, mapping_csv_path, ehmi_csv_path, avg_df, memory_type):
        """
        Creates individual scatter plots of the eHMI mean versus each LLM score column.
        Returns a dictionary with column names as keys and corresponding Plotly figures as values.
        """
        mapping_df = pd.read_csv(mapping_csv_path)
        ehmi_df = pd.read_csv(ehmi_csv_path).rename(columns={"mean": "ehmi_mean"})

        mapping_df["id"] = mapping_df["id"].astype(str)
        avg_df["image"] = avg_df["image"].astype(str).str.replace("image_", "", regex=False)

        mapping_df["text"] = mapping_df["text"].str.upper()
        ehmi_df["eHMI"] = ehmi_df["eHMI"].str.upper()

        merged_df = pd.merge(avg_df, mapping_df, left_on="image", right_on="id", how="inner")
        merged_df = pd.merge(merged_df, ehmi_df, left_on="text", right_on="eHMI", how="inner")
        numeric_columns = [col for col in avg_df.columns if col != "image"]

        for col in numeric_columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged_df["ehmi_mean"],
                y=merged_df[col],
                mode='markers',
                name=col,
                text=merged_df["text"]
            ))
            fig.update_layout(
                title=f"eHMI Mean vs {col}",
                xaxis_title="eHMI Mean",
                yaxis_title=col
            )

            self.save_plotly_figure(fig, f"scatter_plot_{col}_{memory_type}", save_final=False)


# Example usage
if __name__ == "__main__":
    analysis = Analysis()
    # Loop over both configurations
    for memory_type in ["with_memory", "without_memory"]:
        folder_path = os.path.join(output_path, memory_type, "analysed")

        # Skip if folder doesn't exist or is empty
        if not os.path.isdir(folder_path) or not os.listdir(folder_path):
            analysis.logger.error(f"Skipping {memory_type}: folder is missing or empty.")
            continue

        analysis.logger.info(f"Processing: {memory_type}")

        avg_df = analysis.average_llm_results(
            folder_path=os.path.join(output_path, memory_type, "analysed")
        )

        analysis.plot_ehmi_vs_llm(
            mapping_csv_path=os.path.join(data_path, "mapping.csv"),
            ehmi_csv_path=os.path.join(data_path, "ehmis.csv"),
            avg_df=avg_df,
            memory_type=memory_type
        )

        figures = analysis.plot_individual_ehmi_vs_llm(
            mapping_csv_path=os.path.join(data_path, "mapping.csv"),
            ehmi_csv_path=os.path.join(data_path, "ehmis.csv"),
            avg_df=avg_df,
            memory_type=memory_type
        )
