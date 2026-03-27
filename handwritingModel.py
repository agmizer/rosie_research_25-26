import base64
import urllib.request

from io import BytesIO
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

#need for Qwen 3B
import torch
from pdf2image import convert_from_path
from qwen_vl_utils import process_vision_info


class Handwriting: 

    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    
    def read_page(self, path_to_pdf, page):
        # Render 1 page to an image
        images = convert_from_path(
            path_to_pdf, 
            first_page=page + 1, 
            last_page=page + 1,
            dpi=200 # 200 is a good balance for OCR quality vs memory
        )
        if not images:
            return ""
        
        image = images[0]

        # Build the full prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        # Constraints to keep VRAM under control
                        "min_pixels": 256 * 28 * 28,
                        "max_pixels": 1024 * 28 * 28, 
                    },
                    {"type": "text", "text": "Transcribe the handwriting in this image accurately. Output only the text."},
                ],
            }
        ]
    

        # Apply the chat template and processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)


        # Generate the output
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
            # Trim the prompt tokens from the output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
      

        # 5. Cleanup to free VRAM
        del inputs
        torch.cuda.empty_cache()

        return output_text[0]