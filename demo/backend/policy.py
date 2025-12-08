import os


class Policy:
    """Very small placeholder policy class.

    In a real project this would load a torch model from `model_path` and
    produce generations. This stub echoes the prompt with a friendly message.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._loaded = False
        self._load()

    def _load(self):
        # If model file exists, mark loaded. Otherwise remain a lightweight stub.
        if self.model_path and os.path.exists(self.model_path):
            self._loaded = True
        else:
            self._loaded = False

    def generate(self, prompt: str) -> str:
        if self._loaded:
            # Real model inference would happen here
            return f"[policy model loaded] Echo: {prompt}"
        else:
            return f"[stub policy] Echo: {prompt}"
