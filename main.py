from deepseek import VisualQuestionAnswering
from ollama import OllamaClient
from chat_gpt import ImageAnalyser
import common
import os
import random

output_folder = common.get_configs("output")
os.makedirs(output_folder, exist_ok=True)


def get_shuffled_images(folder_path, seed):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported_extensions)
    ]
    random.seed(seed)
    return random.sample(image_paths, len(image_paths))


def run_ollama(prompt, image_dir, seed, use_history, max_memory):
    print("\n--- Running OllamaClient ---\n")
    model_list = common.get_configs("model_names")
    image_paths = get_shuffled_images(image_dir, seed)

    for model in model_list:
        print(f"\n--- Processing with model: {model} ---\n")
        client = OllamaClient(
            model_name=model,
            use_history=use_history,
            max_memory_messages=max_memory
        )
        client.generate(
            prompt=prompt,
            image_paths=image_paths,
            seed=seed
        )
        if common.get_configs("delete_history_files"):
            OllamaClient.delete_old_file()


def run_vqa(prompt, image_dir, seed, use_history, max_memory):
    print("\n--- Running Visual Question Answering (DeepSeek) ---\n")
    vqa = VisualQuestionAnswering(use_history=use_history, max_memory_messages=max_memory)
    image_paths = get_shuffled_images(image_dir, seed)

    for image_path in image_paths:
        image_file = os.path.basename(image_path)
        answer = vqa.ask_question(image_path, prompt, seed=seed)
        print(f"\n[{image_file}] DeepSeek Response: {answer}")


def run_chatgpt(prompt, image_dir, seed, use_history, max_memory):
    print("\n--- Running GPT-4 Vision Analyser ---\n")
    image_paths = get_shuffled_images(image_dir, seed)

    for image_path in image_paths:
        image_file = os.path.basename(image_path)
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
    image_dir = common.get_configs("data")
    prompt = common.get_configs("prompt")
    use_history = common.get_configs("use_history")
    max_memory = common.get_configs("max_memory_messages")
    seed_list = common.get_configs("random_seed")

    for seed in seed_list:
        print(f"\n============================\nRunning for seed: {seed}\n============================")
        run_ollama(prompt, image_dir, seed, use_history, max_memory)
        run_vqa(prompt, image_dir, seed, use_history, max_memory)
        run_chatgpt(prompt, image_dir, seed, use_history, max_memory)
