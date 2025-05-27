# Standard libraries
import os
import re
import math
import pandas as pd
import subprocess
import requests
import time
import json
import socket
import base64
import common
from logmod import logs
from datetime import datetime
import hashlib
from custom_logger import CustomLogger


# LangChain for conversation memory
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class OllamaClient:
    """
    A client for interacting with an Ollama server for LLM+vision processing.
    Handles image encoding, prompt generation, conversation memory tracking, 
    logging of interactions, and saving results to CSV.
    """

    def __init__(self,
                 model_name="gemma3:1b",
                 host="localhost",
                 port=11434,
                 use_history=True,
                 max_memory_messages=6):
        """
        Initialise the OllamaClient with the given parameters.

        Args:
            model_name (str): Name of the model to use.
            host (str): Host where the Ollama server is running.
            port (int): Port number of the Ollama server.
            use_history (bool): Flag to determine if conversation history should be used.
            max_memory_messages (int): Maximum number of messages to store in conversation memory.
        """
        # Setup logging configuration from common configs.
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.template = common.get_configs('plotly_template')

        # Initialise instance attributes.
        self.model_name = model_name
        self.host = host
        self.port = port
        self.first_run = True  # Indicator for the first run (for history usage)
        self.url = f"http://{self.host}:{self.port}/api/generate"
        self.history_file = os.path.join(common.get_configs("output"), "ollama_image_history.json")
        self.memory_file = os.path.join(common.get_configs("output"), "ollama_memory.json")
        # Default output_file is now overridden in generate() based on seed.
        self.output_file = os.path.join(common.get_configs("output"), "output.csv")

        self.use_history = use_history
        self.max_memory_messages = max_memory_messages
        # Initialise the conversation memory using LangChain.
        self.memory = ConversationBufferMemory(return_messages=True)

        # Ensure the Ollama server is running and the model is available.
        self.ensure_server_running()
        self.ensure_model_available()
        # Load conversation memory from file if it exists.
        self.load_memory()

    @staticmethod
    def delete_old_file():
        """
        Delete old history and memory files if they exist.
        This helps ensure a clean state between runs.
        """
        output_folder = common.get_configs("output")
        history_path = os.path.join(output_folder, "ollama_image_history.json")
        memory_path = os.path.join(output_folder, "ollama_memory.json")

        if os.path.exists(history_path):
            os.remove(history_path)
        if os.path.exists(memory_path):
            os.remove(memory_path)

    def is_port_open(self):
        """
        Check if the configured port is open on the host.

        Returns:
            bool: True if the port is open, False otherwise.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # connect_ex returns 0 if connection is successful (port is open)
            return s.connect_ex((self.host, self.port)) == 0

    def start_ollama(self):
        """
        Start the Ollama server as a subprocess.
        This method calls the 'ollama serve' command.
        """
        logger.info("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"])
        # Allow some time for the server to start.
        time.sleep(2)

    def ensure_server_running(self):
        """
        Ensure that the Ollama server is running.
        If the server is not running, it attempts to start it.
        Exits the program if the server fails to start.
        """
        if not self.is_port_open():
            self.start_ollama()
            if not self.is_port_open():
                logger.error("Ollama failed to start.")
                exit(1)

    def model_exists(self):
        """
        Check if the specified model is available by listing models from Ollama.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return self.model_name in result.stdout
        except Exception as e:
            logger.error(f"Failed to check model existence: {e}")
            return False

    def pull_model(self):
        """
        Pull the specified model from Ollama if it is not available locally.
        Exits the program if the model cannot be pulled.
        """
        logger.info(f"Pulling model '{self.model_name}'...")
        try:
            subprocess.run(["ollama", "pull", self.model_name], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error pulling model '{self.model_name}': {e}")
            exit(1)

    def ensure_model_available(self):
        """
        Ensure that the desired model is available locally.
        If the model does not exist, it will be pulled.
        """
        if not self.model_exists():
            self.pull_model()

    def encode_image_to_base64(self, image_path):
        """
        Encode an image file to a base64 string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64-encoded string of the image.
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def hash_image(self, file_path):
        """
        Compute the MD5 hash of an image file.

        Args:
            file_path (str): Path to the image file.

        Returns:
            str: MD5 hash of the image.
        """
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def log_interaction(self, prompt, image_path, response):
        """
        Log the interaction details including prompt, image info, and response.

        Args:
            prompt (str): The prompt used in the interaction.
            image_path (str): Path to the image file.
            response (str): Response generated by the model.
        """
        image_hash = self.hash_image(image_path)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "image_path": image_path,
            "image_hash": image_hash,
            "response": response
        }

        try:
            # Load existing history if available.
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []

        # Append the new entry to the history.
        history.append(entry)

        # Save the updated history back to file.
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def load_memory(self):
        """
        Load conversation memory from the memory file, limiting to the last max_memory_messages.
        """
        try:
            with open(self.memory_file, "r") as f:
                messages = json.load(f)
                full_list = messages_from_dict(messages)
                # Limit the memory to the most recent messages.
                self.memory.chat_memory.messages = full_list[-self.max_memory_messages:]
        except FileNotFoundError:
            # If no memory file exists, proceed without loading.
            pass

    def save_memory(self):
        """
        Save the current conversation memory to the memory file.
        """
        messages = messages_to_dict(self.memory.chat_memory.messages)
        with open(self.memory_file, "w") as f:
            json.dump(messages, f, indent=2)

    def generate(self, prompt, image_paths, output_csv=None, use_history=None, seed=42):
        """
        Process the given image_paths using the provided prompt, generate responses, and update the output CSV file.

        If an output file named output_{seed}.csv already exists, it is loaded and a new column
        (named after the model) is added/updated without deleting existing data.

        Args:
            prompt (str): The prompt or instruction for image processing.
            image_paths (list): List of image file paths to process.
            output_csv (str, optional): CSV file to save the results. Defaults to None.
            use_history (bool, optional): Whether to include conversation history in the prompt.
                                          Defaults to None, which uses the instance setting.
            seed (int, optional): Seed value for randomness and reproducibility. Defaults to 42.
        """
        # If use_history is not provided, use the instance default.
        if use_history is None:
            use_history = self.use_history

        # Set output file based on the seed if not explicitly provided.
        if output_csv is None:
            output_csv = os.path.join(common.get_configs("output"), f"output_{seed}.csv")

        # If no images are provided, exit the function.
        if not image_paths:
            logger.error("No supported image files provided.")
            return

        # Attempt to load an existing CSV, or create a new DataFrame if the file doesn't exist.
        try:
            df = pd.read_csv(output_csv)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["image"])

        # Ensure the current model's column exists in the DataFrame.
        if self.model_name not in df.columns:
            df[self.model_name] = pd.NA

        # Process each image in the provided list.
        for img_path in image_paths:
            image_name = os.path.basename(img_path)

            # Check if this image has already been processed for the current model.
            if pd.notna(df.loc[df["image"] == image_name, self.model_name]).any():  # type: ignore
                logger.info(f"Skipping '{image_name}' (already processed).")
                continue

            # Encode the image in base64 format.
            b64_image = self.encode_image_to_base64(img_path)
            full_response = ""

            # If using history and not the first run, format conversation history for the prompt.
            if use_history and not self.first_run:
                formatted_history = ""
                for message in self.memory.chat_memory.messages:
                    if message.__class__.__name__ == "HumanMessage":
                        formatted_history += f"History - Human: {message.content}\n"
                    elif message.__class__.__name__ == "AIMessage":
                        formatted_history += f"History - AI: {message.content}\n"

                # Build the full prompt by combining the base prompt, history introduction, and instructions.
                full_prompt = (
                    f"{common.get_configs('base_prompt')}\n\n"
                    f"{common.get_configs('history_intro')}\n"
                    f"{formatted_history}\n"
                    f"{common.get_configs('current_image_instruction')}"
                )
            else:
                full_prompt = prompt

            # Prepare the data payload for the request.
            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "images": [b64_image],
                "options": {
                    "temperature": common.get_configs("temperature"),
                    "seed": seed
                }
            }

            try:
                # Send the POST request to the Ollama server.
                response = requests.post(self.url, json=data, stream=True)

                if response.status_code == 200:
                    print(f"\n[{image_name}] Generated text: ", end="", flush=True)

                    # Process streamed lines from the response.
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            result = json.loads(decoded_line)
                            generated_text = result.get("response", "")
                            full_response += generated_text
                            print(generated_text, end="", flush=True)

                    # If using conversation history, update and save the memory.
                    if use_history:
                        self.memory.chat_memory.add_user_message(prompt)
                        self.memory.chat_memory.add_ai_message(full_response.strip())
                        # Limit the history to the most recent messages.
                        self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_memory_messages:]
                        self.save_memory()

                    # Mark that the first run has completed.
                    self.first_run = False

                    # Log the interaction details.
                    self.log_interaction(prompt, img_path, full_response.strip())

                    # Update the DataFrame: either update the existing row or append a new row.
                    if image_name in df["image"].values:
                        df.loc[df["image"] == image_name, self.model_name] = full_response.strip()
                    else:
                        new_row = {"image": image_name, self.model_name: full_response.strip()}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                else:
                    logger.error(f"Request failed with status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send request to Ollama: {e}")

        # Ensure the output directory exists and save the results to CSV.
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"\nSaved results to {output_csv}")

    def generate_simple_text(self, sentence, model="deepseek-r1:14b", prompt="Tell me a story.",):
        """
        Generate text using a simple prompt and stream the response from the model.

        Args:
            prompt (str): The text prompt to send to the model.
            model (str): The model to use for generation (e.g., 'deepseek-r1:1.5b').

        Returns:
            tuple: (full generated text, extracted number or NaN)
        """
        # Switch to the new model for this request
        self.model_name = model

        # Ensure server and model are ready
        self.ensure_server_running()
        self.ensure_model_available()

        # Prompt
        prompt = f"""
            Read the following sentence carefully and extract the number mentioned in it.
            Only return the number (as digits), without any additional explanation or units.

            Sentence: "{sentence}"
            """

        url = f"http://{self.host}:{self.port}/api/generate"
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }

        full_response = ""

        try:
            response = requests.post(url, json=data, stream=True)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        result = json.loads(decoded_line)
                        generated_text = result.get("response", "")
                        full_response += generated_text
            else:
                full_response = f"Request failed with status code {response.status_code}"
        except requests.exceptions.RequestException as e:
            full_response = f"Request failed: {e}"

        # Extract number from the generated text
        number = self.extract_number(full_response)

        if number is None:
            number = math.nan

        return number

    @staticmethod
    def extract_number(text):
        """
        Extracts and returns the first number found in the provided text.
        Returns an int if the number is an integer, or a float if it contains a decimal.
        """
        pattern = r"[-+]?\d*\.\d+|[-+]?\d+"
        match = re.search(pattern, text)
        if match:
            num_str = match.group(0)
            return float(num_str) if '.' in num_str else int(num_str)
        return None


if __name__ == "__main__":
    # This block can be used for testing the OllamaClient functionality.
    # For now, it is left empty.
    pass
