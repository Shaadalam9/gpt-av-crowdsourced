import os
import pandas as pd
import subprocess
import requests
import time
import json
import socket
import base64
import common
from custom_logger import CustomLogger
from logmod import logs
import random
from datetime import datetime
import hashlib

from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict


class OllamaClient:
    """
    A client interface to manage interaction with the Ollama LLM server.
    Handles server initialization, model setup, image-based prompting, and response history tracking.
    """

    def __init__(self, model_name="gemma3:1b", host="localhost", port=11434):
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)
        self.template = common.get_configs('plotly_template')

        self.model_name = model_name
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}/api/generate"

        self.history_file = "ollama_image_history.json"
        self.memory_file = "ollama_memory.json"
        self.memory = ConversationBufferMemory(return_messages=True)

        self.ensure_server_running()
        self.ensure_model_available()
        self.load_memory()

    def is_port_open(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0

    def start_ollama(self):
        print("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(2)

    def ensure_server_running(self):
        if not self.is_port_open():
            self.start_ollama()
            if not self.is_port_open():
                print("Ollama failed to start.")
                exit(1)

    def model_exists(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return self.model_name in result.stdout
        except Exception as e:
            print(f"Failed to check model existence: {e}")
            return False

    def pull_model(self):
        print(f"Pulling model '{self.model_name}'...")
        try:
            subprocess.run(["ollama", "pull", self.model_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error pulling model '{self.model_name}': {e}")
            exit(1)

    def ensure_model_available(self):
        if not self.model_exists():
            self.pull_model()

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def hash_image(self, file_path):
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def log_interaction(self, prompt, image_path, response):
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
        try:
            with open(self.memory_file, "r") as f:
                messages = json.load(f)
                self.memory.chat_memory.messages = messages_from_dict(messages)
        except FileNotFoundError:
            pass

    def save_memory(self):
        messages = messages_to_dict(self.memory.chat_memory.messages)
        with open(self.memory_file, "w") as f:
            json.dump(messages, f, indent=2)

    def generate(self, prompt, image_folder="data", output_csv="output.csv"):
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(supported_extensions)
        ]

        if not image_paths:
            print(f"No supported image files found in {image_folder}")
            return

        random.seed(common.get_configs("random_seed"))
        selected_images = random.sample(image_paths, len(image_paths))
        records = []

        for img_path in selected_images:
            b64_image = self.encode_image_to_base64(img_path)
            full_response = ""

            # Build prompt with memory
            history_context = self.memory.load_memory_variables({}).get("history", "")
            full_prompt = f"{history_context}\nUser: {prompt}\nAssistant:"

            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "images": [b64_image]
            }

            try:
                response = requests.post(self.url, json=data, stream=True)

                if response.status_code == 200:
                    print("Generated text: ", end="", flush=True)

                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")
                            result = json.loads(decoded_line)
                            generated_text = result.get("response", "")
                            full_response += generated_text
                            print(generated_text, end="", flush=True)

                    # Save interaction
                    self.memory.chat_memory.add_user_message(prompt)
                    self.memory.chat_memory.add_ai_message(full_response.strip())
                    self.save_memory()
                    self.log_interaction(prompt, img_path, full_response.strip())

                    records.append({
                        "image": os.path.basename(img_path),
                        self.model_name: full_response.strip()
                    })

                else:
                    print(f"Request failed with status code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to send request to Ollama: {e}")

        if records:
            df = pd.DataFrame(records)
            df.to_csv(output_csv, index=False)
            print(f"\nSaved results to {output_csv}")


# Example usage
if __name__ == "__main__":
    client = OllamaClient(model_name=common.get_configs("model_name"))
    prompt = common.get_configs("prompt")
    client.generate(prompt=prompt, image_folder=common.get_configs("data"))
