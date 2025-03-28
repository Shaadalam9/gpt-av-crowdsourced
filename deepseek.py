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

# Disable parallelism in tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VisualQuestionAnswering:
    """
    A class that enables visual question answering using DeepSeek VL2.
    Supports contextual conversations by maintaining chat memory.
    """

    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny", use_history=True, max_memory_messages=6):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_history = use_history
        self.first_run = True
        self.max_memory_messages = max_memory_messages
        self.memory_file = os.path.join(common.get_configs("output"), "deepseek_memory.json")
        self.memory = ConversationBufferMemory(return_messages=True)
        self.load_memory()
        self._load_model_and_processor()

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as f:
                messages = json.load(f)
                full_list = messages_from_dict(messages)
                self.memory.chat_memory.messages = full_list[-self.max_memory_messages:]
        except FileNotFoundError:
            pass

    def save_memory(self):
        messages = messages_to_dict(self.memory.chat_memory.messages)
        with open(self.memory_file, "w") as f:
            json.dump(messages, f, indent=2)

    def _load_model_and_processor(self):
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer  # type: ignore

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).to(self.device).eval()  # type: ignore

    def ask_question(self, image_path, question, model_name="deepseek-vl2", seed=42):
        output_csv = os.path.join(common.get_configs("output"), f"output_csv_{seed}.csv")

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

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{full_prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        pil_images = load_pil_images(conversation)
        inputs = self.processor(conversations=conversation,
                                images=pil_images,
                                force_batchify=True,
                                system_prompt=""
                                ).to(self.model.device)  # type: ignore

        inputs_embeds = self.model.prepare_inputs_embeds(**inputs)

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

        decoded = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        answer = decoded.split("<|Assistant|>")[-1].strip()

        if self.use_history:
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_memory_messages:]
            self.save_memory()

        self.first_run = False

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


if __name__ == "__main__":
    pass
