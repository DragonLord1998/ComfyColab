from .nodes import (
    Krea2BundleLoader,
    QwenImageEdit2511BundleLoader,
    ZImageTurboBundleLoader,
)


NODE_CLASS_MAPPINGS = {
    "ComfyColabZImageTurboBundleLoader": ZImageTurboBundleLoader,
    "ComfyColabQwenImageEdit2511BundleLoader": QwenImageEdit2511BundleLoader,
    "ComfyColabKrea2BundleLoader": Krea2BundleLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabZImageTurboBundleLoader": "Z-Image Turbo Bundle Loader",
    "ComfyColabQwenImageEdit2511BundleLoader": "Qwen Image Edit 2511 Bundle Loader",
    "ComfyColabKrea2BundleLoader": "Krea 2 Bundle Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
