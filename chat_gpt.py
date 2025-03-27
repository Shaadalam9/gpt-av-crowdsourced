import os
import base64
from dotenv import load_dotenv
from openai import OpenAI
import common
from custom_logger import CustomLogger
from logmod import logs

class ImageAnalyser:
    def __init__(self, image_path, prompt="What do you see on this image?"):
        load_dotenv()
        self.logger = CustomLogger().get_logger(__name__)
        self.image_path = image_path
        self.prompt = prompt
        self.client = OpenAI(api_key=common.get_secrets("OPENAI_API_KEY"))

    def encode_image(self):
        try:
            with open(self.image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
                self.logger.info("Image successfully encoded.")
                return encoded
        except Exception as e:
            self.logger.error(f"Failed to encode image: {e}")
            raise

    def analyze_image(self):
        base64_image = self.encode_image()
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
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
            content = response.choices[0].message.content
            self.logger.info("Image analysis completed.")
            return content
        except Exception as e:
            self.logger.error(f"An error occurred during image analysis: {e}")
            return None

# Example usage
if __name__ == "__main__":
    pass