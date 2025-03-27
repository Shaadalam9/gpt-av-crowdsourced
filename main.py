from deepseek import VisualQuestionAnswering
from ollama import OllamaClient
from chat_gpt import ImageAnalyser
import common
import os
import subprocess


def run_ollama():
    """
    Run the OllamaClient for image analysis using various models.

    Loads configuration settings, iterates through each model, and generates outputs
    for images in the specified folder. Optionally deletes conversation history files.
    """
    print("\n--- Running OllamaClient ---\n")
    model_list = common.get_configs("model_names")
    prompt = common.get_configs("prompt")

    for model in model_list:
        print(f"\n--- Processing with model: {model} ---\n")
        client = OllamaClient(
            model_name=model,
            use_history=common.get_configs("use_history"),
            max_memory_messages=common.get_configs("max_memory_messages")
        )
        client.generate(
            prompt=prompt,
            image_folder=common.get_configs("data"),
            seed=common.get_configs("random_seed")
        )
        # Optionally delete memory/history file after each run
        if common.get_configs("delete_history_files"):
            OllamaClient.delete_old_file()


def run_vqa():
    """
    Run Visual Question Answering using the DeepSeek-VL2 model.

    Sends each image to the model with a question and stores the response.
    """
    print("\n--- Running Visual Question Answering (DeepSeek) ---\n")
    image_dir = common.get_configs("data")
    question = common.get_configs("prompt")
    use_history = common.get_configs("use_history")
    max_memory = common.get_configs("max_memory_messages")
    seed = common.get_configs("random_seed")

    vqa = VisualQuestionAnswering(use_history=use_history, max_memory_messages=max_memory)

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_path = os.path.join(image_dir, image_file)
            answer = vqa.ask_question(image_path, question, seed=seed)
            print(f"\n[{image_file}] DeepSeek Response: {answer}")



def run_chatgpt():
    """
    Run GPT-4 Vision-based image analysis via the ImageAnalyser.

    Sends each image in a folder to the OpenAI model and stores the result.
    """
    print("\n--- Running GPT-4 Vision Analyser ---\n")
    image_dir = common.get_configs("data")
    prompt = common.get_configs("prompt")
    use_history = common.get_configs("use_history")
    max_memory = common.get_configs("max_memory_messages")
    seed = common.get_configs("random_seed")

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_path = os.path.join(image_dir, image_file)
            analyser = ImageAnalyser(
                image_path=image_path,
                prompt=prompt,
                use_history=use_history,
                max_memory_messages=max_memory
            )
            result = analyser.analyse_image(seed=seed)
            if result:
                print(f"\n[{image_file}] GPT-4 Vision Response: {result}")


if __name__ == "__main__":
    # run_ollama()
    # Stop the 'ollama' service
    # subprocess.run(["systemctl", "stop", "ollama"], check=True)
    run_vqa()
    # run_chatgpt()
