from deepseek import VisualQuestionAnswering
from ollama import OllamaClient
from chat_gpt import GPT_ImageAnalyser
import common
import os
import random

# Create the output folder if it doesn't exist already.
output_folder = common.get_configs("output")
os.makedirs(output_folder, exist_ok=True)


def get_shuffled_images(folder_path, seed):
    """
    Retrieve and shuffle image file paths from a given folder.

    Args:
        folder_path (str): Path to the folder containing image files.
        seed (int): Seed value for random shuffling to ensure reproducibility.

    Returns:
        list: A list of shuffled image file paths.
    """
    # Define the supported image file extensions.
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    # Get the full paths for files in the folder that have a supported extension.
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_extensions)
    ]
    # Seed the random number generator to ensure reproducibility.
    random.seed(seed)
    # Return the shuffled list of image paths.
    return random.sample(image_paths, len(image_paths))


def run_ollama(prompt, image_dir, seed, use_history, max_memory):
    """
    Run the OllamaClient on a set of images with the provided prompt.

    This function retrieves the list of model names from the configuration,
    shuffles the images using the provided seed, and processes each image
    with every model by generating responses.

    Args:
        prompt (str): The prompt or question to be processed by the models.
        image_dir (str): Directory path containing the images.
        seed (int): Seed value for random operations.
        use_history (bool): Whether to use conversation history.
        max_memory (int): Maximum number of memory messages to be used.
    """
    print("\n--- Running OllamaClient ---\n")
    # Get the list of model names from the configuration.
    model_list = common.get_configs("model_names")
    # Get the shuffled list of image paths.
    image_paths = get_shuffled_images(image_dir, seed)

    # Process each model on the list of images.
    for model in model_list:
        print(f"\n--- Processing with model: {model} ---\n")
        client = OllamaClient(
            model_name=model,
            use_history=use_history,
            max_memory_messages=max_memory
        )
        # Generate output using the client with the specified prompt and images.
        client.generate(
            prompt=prompt,
            image_paths=image_paths,
            seed=seed
        )
        # Optionally delete old history files based on the configuration.
        if common.get_configs("delete_history_files"):
            OllamaClient.delete_old_file()


def run_vqa(prompt, image_dir, seed, use_history, max_memory):
    """
    Run visual question answering using DeepSeek's VisualQuestionAnswering.

    For each image, this function sends the prompt and prints out the generated answer.

    Args:
        prompt (str): The question or prompt for the visual QA model.
        image_dir (str): Directory containing the images to process.
        seed (int): Seed value for any random operations.
        use_history (bool): Whether to use conversation history in processing.
        max_memory (int): Maximum number of memory messages allowed.
    """
    print("\n--- Running Visual Question Answering (DeepSeek) ---\n")
    # Initialize the VisualQuestionAnswering instance with history settings.
    vqa = VisualQuestionAnswering(use_history=use_history, max_memory_messages=max_memory)
    # Retrieve and shuffle the list of image paths.
    image_paths = get_shuffled_images(image_dir, seed)

    # Process each image and print the answer from DeepSeek.
    for image_path in image_paths:
        image_file = os.path.basename(image_path)
        answer = vqa.ask_question(image_path, prompt, seed=seed)
        print(f"\n[{image_file}] DeepSeek Response: {answer}")


def run_chatgpt(prompt, image_dir, seed, use_history, max_memory):
    """
    Run image analysis using GPT-4 Vision Analyser.

    For each image, this function instantiates the analyser, processes the image
    with the given prompt, and prints the result if available.

    Args:
        prompt (str): The prompt for GPT-4 vision analysis.
        image_dir (str): Directory containing images to be analysed.
        seed (int): Seed for random operations.
        use_history (bool): Whether to use conversation history.
        max_memory (int): Maximum number of history messages allowed.
    """
    print("\n--- Running GPT-4 Vision Analyser ---\n")
    # Get the shuffled list of image paths.
    image_paths = get_shuffled_images(image_dir, seed)

    # Process each image individually.
    for image_path in image_paths:
        image_file = os.path.basename(image_path)
        analyser = GPT_ImageAnalyser(
            image_path=image_path,
            prompt=prompt,
            use_history=use_history,
            max_memory_messages=max_memory
        )
        result = analyser.analyse_image(seed=seed)
        if result:
            print(f"\n[{image_file}] GPT-4 Vision Response: {result}")


if __name__ == "__main__":
    # Retrieve configuration settings from the common configuration module.
    image_dir = common.get_configs("data")
    prompt = common.get_configs("prompt")
    use_history = common.get_configs("use_history")
    max_memory = common.get_configs("max_memory_messages")
    seed_list = common.get_configs("random_seed")

    # Loop over each seed in the configuration to run the processes.
    for seed in seed_list:
        print(f"\n============================\nRunning for seed: {seed}\n============================")
        run_ollama(prompt, image_dir, seed, use_history, max_memory)
        run_vqa(prompt, image_dir, seed, use_history, max_memory)
        run_chatgpt(prompt, image_dir, seed, use_history, max_memory)
