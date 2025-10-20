"""
Centralized configuration for model checkpoint paths.

Fill these paths with your local .pth files before running.
Nothing is loaded at import; consumers should handle missing paths.
"""

# ——— Inference interfaces used by tools.py ———


VITIME_MODEL_PATH: str | None = "your_Path"  
