# Standard libraries
import os
import pandas as pd
import subprocess
import requests
import time
import json
import socket
import base64
import common
from logmod import logs
import random
from datetime import datetime
import hashlib

# LangChain for conversation memory
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

output_folder = common.get_configs("output")
os.makedirs(output_folder, exist_ok=True)


class OllamaClient:
    """
    A client for interacting with an Ollama server for LLM+vision processing.
    Handles image encoding, prompt generation, memory tracking, and result logging.
    """

    def __init__(self,
                 model_name="gemma3:1b",
                 host="localhost",
                 port=11434,
                 use_history=True,
                 max_memory_messages=6):
        """
        Initialise the OllamaClient.

        Args:
            model_name (str): Name of the LLM model to use.
            host (str): Server host.
            port (int): Server port.
            use_history (bool): Whether to use conversation history.
            max_memory_messages (int): Max number of memory messages to retain.
        """
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.template = common.get_configs('plotly_template')

        self.model_name = model_name
        self.host = host
        self.port = port
        self.first_run = True
        self.url = f"http://{self.host}:{self.port}/api/generate"
        self.history_file = os.path.join(output_folder, "ollama_image_history.json")
        self.memory_file = os.path.join(output_folder, "ollama_memory.json")
        self.output_file = os.path.join(output_folder, "output.csv")

        self.use_history = use_history
        self.max_memory_messages = max_memory_messages
        self.memory = ConversationBufferMemory(return_messages=True)

        self.ensure_server_running()
        self.ensure_model_available()
        self.load_memory()

    @staticmethod
    def delete_old_file():
        """
        Deletes the old history files.
        """

        if os.path.exists(os.path.join(output_folder, "ollama_image_history.json")):
            os.remove(os.path.join(output_folder, "ollama_image_history.json"))

        if os.path.exists(os.path.join(output_folder, "ollama_memory.json")):
            os.remove(os.path.join(output_folder, "ollama_memory.json"))


    def is_port_open(self):
        """
        Check if the Ollama server port is open.

        Returns:
            bool: True if the port is open, else False.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0

    def start_ollama(self):
        """
        Start the Ollama server.
        """
        print("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(2)

    def ensure_server_running(self):
        """
        Ensure the Ollama server is running; start it if necessary.
        """
        if not self.is_port_open():
            self.start_ollama()
            if not self.is_port_open():
                print("Ollama failed to start.")
                exit(1)

    def model_exists(self):
        """
        Check if the required model is available locally.

        Returns:
            bool: True if model exists, else False.
        """
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return self.model_name in result.stdout
        except Exception as e:
            print(f"Failed to check model existence: {e}")
            return False

    def pull_model(self):
        """
        Pull the required model from Ollama.
        """
        print(f"Pulling model '{self.model_name}'...")
        try:
            subprocess.run(["ollama", "pull", self.model_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error pulling model '{self.model_name}': {e}")
            exit(1)

    def ensure_model_available(self):
        """
        Ensure the specified model is available.
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
        Compute an MD5 hash of an image file.

        Args:
            file_path (str): Path to the image file.

        Returns:
            str: MD5 hash of the image.
        """
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def log_interaction(self, prompt, image_path, response):
        """
        Log the interaction with prompt, image path, and response.

        Args:
            prompt (str): User prompt.
            image_path (str): Image file path.
            response (str): Model-generated response.
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
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []

        history.append(entry)

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

    def load_memory(self):
        """
        Load memory from file to restore conversation context.
        """
        try:
            with open(self.memory_file, "r") as f:
                messages = json.load(f)
                full_list = messages_from_dict(messages)
                self.memory.chat_memory.messages = full_list[-self.max_memory_messages:]
        except FileNotFoundError:
            pass

    def save_memory(self):
        """
        Save current memory to file.
        """
        messages = messages_to_dict(self.memory.chat_memory.messages)
        with open(self.memory_file, "w") as f:
            json.dump(messages, f, indent=2)

    def generate(self, prompt, image_folder="data", output_csv=None, use_history=None, seed=42):
        """
        Run prompt + image through Ollama model and log results.

        Args:
            prompt (str): Text prompt to send with each image.
            image_folder (str): Folder containing images to process.
            output_csv (str): File to store responses.
            use_history (bool): Whether to use conversational memory.
        """
        if use_history is None:
            use_history = self.use_history
        if output_csv is None:
            output_csv = self.output_file

        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(supported_extensions)
        ]

        if not image_paths:
            print(f"No supported image files found in {image_folder}")
            return

        try:
            df = pd.read_csv(output_csv)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["image"])

        if self.model_name not in df.columns:
            df[self.model_name] = pd.NA

        random.seed(seed)
        selected_images = random.sample(image_paths, len(image_paths))

        for img_path in selected_images:
            image_name = os.path.basename(img_path)

            if pd.notna(df.loc[df["image"] == image_name, self.model_name]).any():
                print(f"Skipping '{image_name}' (already processed).")
                continue

            b64_image = self.encode_image_to_base64(img_path)
            full_response = ""

            if use_history and not self.first_run:
                # Format the conversation history (only content is extracted)
                formatted_history = ""
                for message in self.memory.chat_memory.messages:
                    if message.__class__.__name__ == "HumanMessage":
                        formatted_history += f"History - Human: {message.content}\n"
                    elif message.__class__.__name__ == "AIMessage":
                        formatted_history += f"History - AI: {message.content}\n"

                full_prompt = (
                    f"{common.get_configs('base_prompt')}\n\n"
                    f"{common.get_configs('history_intro')}\n"
                    f"{formatted_history}\n"
                    f"{common.get_configs('current_image_instruction')}"
                )
            else:
                full_prompt = prompt

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
                response = requests.post(self.url, json=data, stream=True)

                if response.status_code == 200:
                    print(f"\n[{image_name}] Generated text: ", end="", flush=True)

                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            result = json.loads(decoded_line)
                            generated_text = result.get("response", "")
                            full_response += generated_text
                            print(generated_text, end="", flush=True)

                    if use_history:
                        self.memory.chat_memory.add_user_message(prompt)
                        self.memory.chat_memory.add_ai_message(full_response.strip())
                        self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_memory_messages:]
                        self.save_memory()
                    
                    self.first_run = False

                    self.log_interaction(prompt, img_path, full_response.strip())

                    if image_name in df["image"].values:
                        df.loc[df["image"] == image_name, self.model_name] = full_response.strip()
                    else:
                        new_row = {"image": image_name, self.model_name: full_response.strip()}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                else:
                    print(f"Request failed with status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to send request to Ollama: {e}")

        output_path = os.path.join(output_folder, f"output_csv_{seed}.csv")
        os.makedirs(output_folder, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")


# Entry point for standalone execution
if __name__ == "__main__":
    pass
