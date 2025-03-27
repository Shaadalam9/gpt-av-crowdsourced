import os
import json
import pandas as pd
import torch
import common
from transformers import AutoModelForCausalLM  # noqa:F401
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

# Disable parallelism in tokeniser warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

output_folder = common.get_configs("output")
os.makedirs(output_folder, exist_ok=True)


class VisualQuestionAnswering:
    """
    A class that enables visual question answering using DeepSeek VL2.
    Supports contextual conversations by maintaining chat memory.
    """

    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny", use_history=True, max_memory_messages=6):
        """
        Initialises the VisualQuestionAnswering model and processor.

        Args:
            model_path (str): HuggingFace model identifier or local path.
            use_history (bool): Whether to utilise previous conversation history.
            max_memory_messages (int): Maximum number of memory messages to retain.
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_history = use_history
        self.first_run = True
        self.max_memory_messages = max_memory_messages
        self.memory_file = os.path.join(output_folder, "deepseek_memory.json")
        self.memory = ConversationBufferMemory(return_messages=True)
        self.load_memory()
        self._load_model_and_processor()

    def load_memory(self):
        """
        Loads conversation history from a local file.
        Only the most recent `max_memory_messages` are retained.
        """
        try:
            with open(self.memory_file, "r") as f:
                messages = json.load(f)
                full_list = messages_from_dict(messages)
                self.memory.chat_memory.messages = full_list[-self.max_memory_messages:]
        except FileNotFoundError:
            # Memory file does not exist yet — no action needed
            pass

    def save_memory(self):
        """
        Saves the current chat memory to a file for persistence across sessions.
        """
        messages = messages_to_dict(self.memory.chat_memory.messages)
        with open(self.memory_file, "w") as f:
            json.dump(messages, f, indent=2)

    def _load_model_and_processor(self):
        """
        Loads the DeepSeek model and associated processor and tokeniser.
        Transfers the model to the correct device with bfloat16 precision.
        """
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer  # type: ignore

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).to(self.device).eval()  # type: ignore

    def ask_question(self, image_path, question, model_name="deepseek-vl2", seed=42):
        """
        Asks a question about the given image using the VQA model and logs output to CSV.

        Args:
            image_path (str): Path to the image file.
            question (str): Question to ask.
            model_name (str): Model name to use as CSV column.
            seed (int): Random seed used for output CSV name.

        Returns:
            str: The model's answer.
        """

        output_csv = os.path.join(output_folder, f"output_csv_{seed}.csv")

        # Format conversation prompt with memory history if enabled
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
            full_prompt = question

        # Construct the conversation format expected by the model
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{full_prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        # Load and prepare image for processing
        pil_images = load_pil_images(conversation)
        inputs = self.processor(conversations=conversation,
                                images=pil_images,
                                force_batchify=True,
                                system_prompt=""
                                ).to(self.model.device)  # type: ignore

        # Prepare image and text input embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)

        # Generate response from model
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        # Decode the model's response
        decoded = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        answer = decoded.split("<|Assistant|>")[-1].strip()

        # Save updated memory if required
        if self.use_history:
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_memory_messages:]
            self.save_memory()
        
        self.first_run = False

        # Save to CSV
        image_name = os.path.basename(image_path)
        try:
            df = pd.read_csv(output_csv)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["image"])

        if model_name not in df.columns:
            df[model_name] = pd.NA

        if image_name in df["image"].values:
            df.loc[df["image"] == image_name, model_name] = answer
        else:
            new_row = {"image": image_name, model_name: answer}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(output_csv, index=False)
        print(f"\nSaved DeepSeek-VL2 output for {image_name} to {output_csv}")
        return answer


# Example usage
if __name__ == "__main__":
    pass
