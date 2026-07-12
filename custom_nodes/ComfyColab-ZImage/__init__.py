from .nodes import ZImageTurboBundleLoader


NODE_CLASS_MAPPINGS = {
    "ComfyColabZImageTurboBundleLoader": ZImageTurboBundleLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabZImageTurboBundleLoader": "Z-Image Turbo Bundle Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
