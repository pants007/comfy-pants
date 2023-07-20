from clip_interrogator import Config, Interrogator
import numpy as np
from PIL import Image


class CLIPInterrogator:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "interrogate"

    CATEGORY = "STRING"

    def interrogate(self, image):
        img_np = image.squeeze(0).numpy()
        ci = Interrogator(
            Config(chunk_size=512, clip_model_name="ViT-L-14/openai"))
        img = Image.fromarray((img_np * 255).astype(np.uint8)).convert("RGB")
        res = ci.interrogate(img)
        return (res,)


NODE_CLASS_MAPPINGS = {
    "CLIPInterrogator": CLIPInterrogator,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPInterrogator Node": "CLIPInterrogator Node"
}
