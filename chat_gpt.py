import os
import pandas as pd
import base64
import json
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore
import common
from custom_logger import CustomLogger
from logmod import logs
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


class ImageAnalyser:
    """
    A class to analyse images using OpenAI's GPT model and vision capabilities.
    Optionally maintains conversation history to provide context-aware responses.
    """

    def __init__(self, image_path, prompt="What do you see on this image?", use_history=True, max_memory_messages=6):
        """
        Initialise the ImageAnalyser class.

        Args:
            image_path (str): Path to the image file to analyse.
            prompt (str): Prompt to send along with the image.
            use_history (bool): Whether to include past conversation history.
            max_memory_messages (int): Number of past messages to retain in memory.
        """
        load_dotenv()
        self.image_path = image_path
        self.prompt = prompt
        self.first_run = True
        self.client = OpenAI(api_key=common.get_secrets("OPENAI_API_KEY"))
        self.use_history = use_history
        self.max_memory_messages = max_memory_messages
        self.memory_file = "chatgpt_memory.json"
        self.memory = ConversationBufferMemory(return_messages=True)
        self.load_memory()

    def encode_image(self):
        """
        Encodes the image file into a base64 string for API submission.

        Returns:
            str: Base64-encoded image string.

        Raises:
            Exception: If image file cannot be read or encoded.
        """
        try:
            with open(self.image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                logger.info("Image successfully encoded.")
                return encoded
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise

    def load_memory(self):
        """
        Loads conversation history from file into memory.
        Only the most recent messages up to `max_memory_messages` are retained.
        """
        try:
            with open(self.memory_file, "r") as f:
                messages = json.load(f)
                full_list = messages_from_dict(messages)
                self.memory.chat_memory.messages = full_list[-self.max_memory_messages:]
        except FileNotFoundError:
            # No memory file exists yet — this is fine for first-time usage.
            pass

    def save_memory(self):
        """
        Saves the current conversation history to a file.
        """
        messages = messages_to_dict(self.memory.chat_memory.messages)
        with open(self.memory_file, "w") as f:
            json.dump(messages, f, indent=2)

    def analyse_image(self, model_name="gpt-4o", seed=42):
        """
        Sends the image and prompt (optionally with history) to OpenAI for analysis.

        Args:
            model_name (str): Column name to store output.
            seed (int): Random seed used to name the CSV file.

        Returns:
            str or None: AI's response to the image, or None if an error occurs.
        """
        output_csv = f"output_csv_{seed}"
        base64_image = self.encode_image()

        # Build prompt including history if enabled
        if self.use_history and not self.first_run:
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
            full_prompt = self.prompt

        try:
            # Submit image and prompt to OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )

            # Extract AI's response
            content = response.choices[0].message.content

            # Update memory with the latest exchange
            if self.use_history:
                self.memory.chat_memory.add_user_message(self.prompt)
                self.memory.chat_memory.add_ai_message(content)
                self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_memory_messages:]
                self.save_memory()
            self.first_run = False

            # Save to CSV
            image_name = os.path.basename(self.image_path)
            try:
                df = pd.read_csv(output_csv)
            except FileNotFoundError:
                df = pd.DataFrame(columns=["image"])

            if model_name not in df.columns:
                df[model_name] = pd.NA

            if image_name in df["image"].values:
                df.loc[df["image"] == image_name, model_name] = content
            else:
                new_row = {"image": image_name, model_name: content}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            df.to_csv(output_csv, index=False)
            print(f"\nSaved GPT-4 Vision output for {image_name} to {output_csv}")
            return content

        except Exception as e:
            logger.error(f"An error occurred during image analysis: {e}")
            return None


# Example usage
if __name__ == "__main__":
    pass
