import os
import tomli


def get_config():
    file_path = os.path.abspath(__file__)
    path = os.path.join(os.path.dirname(file_path),
                        'stylepile.toml')
    with open(path, "rb") as f:
        style = tomli.load(f)
    return style


STYLE = get_config()
ART_TYPES = STYLE['art-type']
CONCEPTS = ['None'] + sorted(STYLE['concepts']['concepts'])
ARTISTS = ['None'] + sorted(STYLE['artists']['artists'])
ART_MOVEMENTS = ['None'] + sorted(STYLE['art-movements']['art-movements'])
COLORS = ['None'] + sorted(STYLE['colors']['colors'])
DIRECTIONS = ['None'] + sorted(STYLE['directions']['directions'])
MOODS = ['None'] + sorted(STYLE['moods']['moods'])
DEFAULTS = STYLE['defaults']


class CLIPTextEncodeAIO:
    @ classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "positive_prompt": ("STRING", {"multiline": True}),
            "opt_prompt": ("STRING", {"default": ""}),
            "concept": (CONCEPTS,),
            "strength_concept": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "type": (list(ART_TYPES.keys()),),
            "strength_type": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "artist": (ARTISTS,),
            "strength_artist": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "movement": (ART_MOVEMENTS,),
            "strength_movement": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "color": (COLORS,),
            "strength_color": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "mood": (MOODS,),
            "strength_mood": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "direction": (DIRECTIONS,),
            "strength_direction": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
            "negative_prompt": ("STRING", {"multiline": True})
        }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, positive_prompt, opt_prompt, concept, strength_concept, type, strength_type, artist, strength_artist, movement, strength_movement,
               color, strength_color, mood, strength_mood, direction, strength_direction, negative_prompt):
        art_type_positive = ART_TYPES[type]['positive']
        art_type_negative = ART_TYPES[type]['negative']
        art_type_pos_str = f'({art_type_positive}:{strength_type})'
        default_positive = DEFAULTS['positive']
        default_negative = DEFAULTS['negative']
        artist_str = f'(made by {artist}:{strength_artist:.2f})' if artist != 'None' else ''
        movement_str = f'(in the style of {movement}:{strength_movement:.2f})' if movement != 'None' else ''
        color_str = f'({color}:{strength_color:.2f})' if color != 'None' else ''
        mood_str = f'({mood}:{strength_mood:.2f})' if mood != 'None' else ''
        direction_str = f'({direction}:{strength_direction:.2f})' if direction != 'None' else ''
        positive_text = f'{opt_prompt}, {positive_prompt}, {art_type_positive}, {default_positive},{artist_str},{movement_str}, {color_str}, {mood_str}, {direction_str}'
        negative_text = f'{negative_prompt}, {art_type_negative}, {default_negative}'
        tokens = clip.tokenize(positive_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        tokens = clip.tokenize(negative_text)
        cond1, pooled1 = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], [[cond1, {"pooled_output": pooled1}]],)


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeAIO": CLIPTextEncodeAIO,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeStylePile Node": "CLIPTextEncodeStylePile Node"
}
