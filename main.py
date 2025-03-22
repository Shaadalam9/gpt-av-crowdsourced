import os
import subprocess
import requests
import time
import json
import socket
import base64
from datetime import datetime
import common
from custom_logger import CustomLogger
from logmod import logs


class OllamaClient:
    """
    A client interface to manage interaction with the Ollama LLM server.
    Handles server initialization, model setup, and text generation requests.
    """

    def __init__(self, model_name="gemma3:1b", host="localhost", port=11434):
        """
        Initialize the OllamaClient.

        Args:
            model_name (str): Name of the model to use.
            host (str): Host address where Ollama server is running.
            port (int): Port number where Ollama server listens.
        """
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)
        self.template = common.get_configs('plotly_template')

        self.model_name = model_name
        self.host = host
        self.port = port
        self.url = f"http://{self.host}:{self.port}/api/generate"

        self.ensure_server_running()
        self.ensure_model_available()

    def is_port_open(self):
        """Check if the specified port is open (i.e., Ollama server is running)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0

    def start_ollama(self):
        """Start the Ollama server as a background process."""
        print("Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"])
        time.sleep(2)  # Wait a moment for the server to start

    def ensure_server_running(self):
        """Ensure Ollama server is running; start it if not."""
        if not self.is_port_open():
            self.start_ollama()
            if not self.is_port_open():
                print("Ollama failed to start.")
                exit(1)

    def model_exists(self):
        """Check if the specified model exists in the Ollama server's model list."""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            return self.model_name in result.stdout
        except Exception as e:
            print(f"Failed to check model existence: {e}")
            return False

    def pull_model(self):
        """Pull the specified model from the Ollama registry."""
        print(f"Pulling model '{self.model_name}'...")
        try:
            subprocess.run(["ollama", "pull", self.model_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error pulling model '{self.model_name}': {e}")
            exit(1)

    def ensure_model_available(self):
        """Ensure the specified model is available locally; pull it if not."""
        if not self.model_exists():
            self.pull_model()

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def generate(self, prompt, image_path="data/image_0.jpg"):
        """
        Send a prompt to the model and stream the generated response.

        Args:
            prompt (str): The input text prompt to send to the model.
        """
        b64_image = self.encode_image_to_base64(image_path)
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "images": [b64_image]
            # "options": {
            # }
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
                        print(generated_text, end="", flush=True)
            else:
                print(f"Request failed with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to send request to Ollama: {e}")


# Example usage
if __name__ == "__main__":
    client = OllamaClient(model_name=common.get_configs("model_name"))
    client.generate(prompt=common.get_configs("prompt"), image_path="data/image_0.jpg")
