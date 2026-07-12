from .nodes import (
    Flux2DevBundleLoader,
    Flux2Klein4BBundleLoader,
    Flux2Klein9BBundleLoader,
    Krea2BundleLoader,
    QwenImageEdit2511BundleLoader,
    ZImageTurboBundleLoader,
)


NODE_CLASS_MAPPINGS = {
    "ComfyColabZImageTurboBundleLoader": ZImageTurboBundleLoader,
    "ComfyColabQwenImageEdit2511BundleLoader": QwenImageEdit2511BundleLoader,
    "ComfyColabKrea2BundleLoader": Krea2BundleLoader,
    "ComfyColabFlux2Klein4BBundleLoader": Flux2Klein4BBundleLoader,
    "ComfyColabFlux2Klein9BBundleLoader": Flux2Klein9BBundleLoader,
    "ComfyColabFlux2DevBundleLoader": Flux2DevBundleLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabZImageTurboBundleLoader": "Z-Image Turbo Bundle Loader",
    "ComfyColabQwenImageEdit2511BundleLoader": "Qwen Image Edit 2511 Bundle Loader",
    "ComfyColabKrea2BundleLoader": "Krea 2 Bundle Loader",
    "ComfyColabFlux2Klein4BBundleLoader": "FLUX.2 Klein 4B Bundle Loader",
    "ComfyColabFlux2Klein9BBundleLoader": "FLUX.2 Klein 9B Bundle Loader",
    "ComfyColabFlux2DevBundleLoader": "FLUX.2 Dev Bundle Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
