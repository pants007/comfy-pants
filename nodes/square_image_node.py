import torch


class MakeSquare:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "padding_mode": (['replicate', 'constant'],),
                "constant_fill": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "square"

    # OUTPUT_NODE = False

    CATEGORY = "image"

    def square(self, image, padding_mode, constant_fill):
        # do some processing on the image, in this example I just invert it
        image = image.squeeze()
        h, w, c = image.shape
        image_channel_rev = image.permute(2, 0, 1)
        if h < w:
            pad_top = (w - h) // 2
            pad_bottom = pad_top
            new_img = torch.nn.functional.pad(
                image_channel_rev, [0, 0, pad_top, pad_bottom], mode=padding_mode, value=constant_fill)
        elif w < h:
            pad_left = (h - w) // 2
            pad_right = pad_left
            new_img = torch.nn.functional.pad(
                image_channel_rev, [pad_left, pad_right, 0, 0], mode=padding_mode, value=constant_fill)
        else:
            new_img = image_channel_rev
        new_img = new_img.permute(2, 1, 0)
        new_img = new_img.transpose(0, 1)
        return (new_img.unsqueeze(0),)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image Make Square": MakeSquare
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Image Make Square Node": "MakeSquare Node"
}
