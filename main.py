# by Md Shadab Alam (md_shadab_alam@outlook.com)
from DeepSeek_VL2.deepseek import VisualQuestionAnswering
from ollama import OllamaClient
from chat_gpt import ImageAnalyser
import common

def run_ollama():
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
        if common.get_configs("delete_history_files"):
            OllamaClient.delete_old_file()

def run_vqa():
    print("\n--- Running Visual Question Answering (DeepSeek) ---\n")
    vqa = VisualQuestionAnswering()
    image_path = input("Enter image path for DeepSeek VQA: ")
    question = input("Ask a question about the image: ")
    answer = vqa.ask_question(image_path, question)
    print(f"Answer: {answer}")

def run_chatgpt():
    print("\n--- Running GPT-4 Vision Analyzer ---\n")
    image_path = input("Enter image path for GPT analysis: ")
    prompt = input("Enter prompt for GPT analysis: ")
    analyzer = ImageAnalyzer(image_path=image_path, prompt=prompt)
    result = analyzer.analyze_image()
    if result:
        print(f"GPT-4 Vision Response: {result}")

if __name__ == "__main__":
    run_ollama()
    run_vqa()
    run_chatgpt()
