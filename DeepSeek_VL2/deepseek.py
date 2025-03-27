import os
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# Disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VisualQuestionAnswering:
    def __init__(self, model_path="deepseek-ai/deepseek-vl2-tiny"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_and_processor()

    def _load_model_and_processor(self):
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer

        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = self.model.to(torch.bfloat16).to(self.device).eval()

    def ask_question(self, image_path, question):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        pil_images = load_pil_images(conversation)
        inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

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
        return answer

# Example usage
if __name__ == "__main__":
    pass