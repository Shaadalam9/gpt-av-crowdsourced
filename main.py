from ollama import OllamaClient
from chat_gpt import GPT_ImageAnalyser
from deepseek import VisualQuestionAnswering
import common
import os
import random
import subprocess
import signal
import platform
from custom_logger import CustomLogger
from logmod import logs

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# Create the output folder if it doesn't exist already.
if common.get_configs("use_history"):
    folder_name = "with_memory"
else:
    folder_name = "without_memory"

# Define the output directory and create the specific subdirectory
dir_path = os.path.join(common.get_configs("output"), folder_name)
os.makedirs(dir_path, exist_ok=True)
logger.info(f"Created folder: {dir_path}")


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


def run_ollama(prompt, image_paths, seed, use_history, max_memory):
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
    logger.info("\n--- Running OllamaClient ---\n")
    # Get the list of model names from the configuration.
    model_list = common.get_configs("model_names")

    # Process each model on the list of images.
    for model in model_list:
        logger.info(f"\n--- Processing with model: {model} ---\n")
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


def kill_ollama_linux():
    """
    Finds and kills all 'ollama' processes on Linux/macOS using pgrep and SIGKILL.
    """
    try:
        # Get list of process IDs matching 'ollama'
        result = subprocess.run(["pgrep", "ollama"], capture_output=True, text=True)
        pids = result.stdout.strip().split()

        if not pids or pids == ['']:
            logger.info("No 'ollama' process found on Linux/macOS.")
            return

        # Kill each process found
        for pid in pids:
            logger.info(f"Killing ollama process with PID: {pid}")
            os.kill(int(pid), signal.SIGKILL)

        logger.info("All 'ollama' processes terminated on Linux/macOS.")

    except Exception as e:
        logger.error(f"Error on Linux/macOS: {e}")


def kill_ollama_windows():
    """
    Finds and kills all 'ollama.exe' processes on Windows using taskkill.
    """
    try:
        # Use taskkill to forcibly terminate all ollama.exe processes
        result = subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], capture_output=True, text=True)

        # Check if the command was successful
        if "SUCCESS" in result.stdout.upper():
            logger.info("All 'ollama.exe' processes terminated on Windows.")
        elif "not found" in result.stdout.lower():
            logger.info("No 'ollama.exe' process found on Windows.")
        else:
            logger.info(result.stdout.strip())

    except Exception as e:
        logger.error(f"Error on Windows: {e}")


def kill_ollama():
    """
    Main function that checks the operating system and calls the appropriate method
    to kill 'ollama' processes.
    """
    current_os = platform.system()

    if current_os == "Linux" or current_os == "Darwin":
        kill_ollama_linux()
    elif current_os == "Windows":
        kill_ollama_windows()
    else:
        logger.error(f"Unsupported OS: {current_os}")


def run_vqa(prompt, image_paths, seed, use_history, max_memory):
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
    logger.info("\n--- Running Visual Question Answering (DeepSeek) ---\n")

    # Initialise the VisualQuestionAnswering instance with history settings.
    vqa = VisualQuestionAnswering(use_history=use_history, max_memory_messages=max_memory)

    # Process each image and print the answer from DeepSeek.
    for image_path in image_paths:
        image_file = os.path.basename(image_path)
        answer = vqa.ask_question(image_path, prompt, seed=seed)
        logger.info(f"\n[{image_file}] DeepSeek Response: {answer}")


def run_chatgpt(prompt, image_paths, seed, use_history, max_memory):
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
    logger.info("\n--- Running GPT-4 Vision Analyser ---\n")

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
            logger.info(f"\n[{image_file}] GPT-4 Vision Response: {result}")


if __name__ == "__main__":
    # Retrieve configuration settings from the common configuration module.
    image_dir = common.get_configs("crowdsourced_data")
    prompt = common.get_configs("prompt")
    use_history = common.get_configs("use_history")
    max_memory = common.get_configs("max_memory_messages")
    seed_list = common.get_configs("random_seed")

    # Loop over each seed in the configuration to run the processes.
    for seed in seed_list:
        # Get the shuffled list ONCE
        shuffled_images = get_shuffled_images(image_dir, seed)

        logger.info(f"\n============================\nRunning for seed: {seed}\n============================")
        run_ollama(prompt, shuffled_images, seed, use_history, max_memory)
        kill_ollama()
        run_vqa(prompt, shuffled_images, seed, use_history, max_memory)
        run_chatgpt(prompt, shuffled_images, seed, use_history, max_memory)
