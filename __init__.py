from .nodes import square_image_node, text_encode_aio
NODE_CLASS_MAPPINGS = {
    **square_image_node.NODE_CLASS_MAPPINGS,
    **text_encode_aio.NODE_CLASS_MAPPINGS}
