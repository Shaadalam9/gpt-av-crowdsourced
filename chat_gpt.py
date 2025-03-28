import os
import json
import base64
import pandas as pd
import common
from openai import OpenAI  # type: ignore
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict


class ImageAnalyser:
    def __init__(self, image_path, prompt, use_history=True, max_memory_messages=6):
        self.image_path = image_path
        self.prompt = prompt
        self.use_history = use_history
        self.max_memory_messages = max_memory_messages
        self.first_run = True
        self.output_csv = os.path.join(common.get_configs("output"),
                                       f"output_csv_{common.get_configs('random_seed')}.csv")
        self.model_name = "gpt-4-vision-preview"
        self.client = OpenAI()
        self.memory_file = os.path.join(common.get_configs("output"), "chatgpt_memory.json")
        self.memory = ConversationBufferMemory(return_messages=True)
        self.load_memory()

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

    def analyse_image(self, seed=42):
        with open(self.image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
            temperature=common.get_configs("temperature")
        )

        content = response.choices[0].message.content

        if self.use_history:
            self.memory.chat_memory.add_user_message(self.prompt)
            self.memory.chat_memory.add_ai_message(content)
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[-self.max_memory_messages:]
            self.save_memory()

        self.first_run = False

        image_name = os.path.basename(self.image_path)
        try:
            df = pd.read_csv(self.output_csv)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["image"])

        if self.model_name not in df.columns:
            df[self.model_name] = pd.NA

        if image_name in df["image"].values:
            df.loc[df["image"] == image_name, self.model_name] = content
        else:
            new_row = {"image": image_name, self.model_name: content}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(self.output_csv, index=False)
        print(f"\nSaved GPT-4 Vision output for {image_name} to {self.output_csv}")
        return content


if __name__ == "__main__":
    pass
