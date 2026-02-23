# =====================================================================
# brain_loader.py - Singleton Model Loader (GPU/CPU Auto-Switch)
# =====================================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config

class MedGemmaLoader:
    """
    Singleton class to ensure the 4B model is only loaded once.
    Detects if an NVIDIA GPU is available and uses 4-bit quantization (bitsandbytes)
    to minimize VRAM usage and maximize speed. Falls back to CPU if no GPU found.
    """
    _instance = None
    _is_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedGemmaLoader, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance

    def load_model(self, hf_token: str):
        if self._is_loaded:
            return

        print(f"Loading Tokenizer for {config.MEDGEMMA_MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MEDGEMMA_MODEL_PATH, 
            token=hf_token
        )

        # Proactive GPU compatibility check (prevents bitsandbytes from corrupting CUDA state)
        gpu_compatible = False
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            print(f"NVIDIA GPU detected: {gpu_name} (compute capability {major}.{minor})")
            
            if major <= 9:  # PyTorch currently supports up to sm_90 (Hopper/Ada)
                gpu_compatible = True
            else:
                print(f"[SKIP GPU] Compute capability {major}.{minor} (Blackwell) is not yet supported by this PyTorch build.")
                print("GPU acceleration will activate automatically once PyTorch adds sm_120 support.")
                print("Using CPU mode for now.\n")

        if gpu_compatible:
            try:
                print("Loading Model in 4-bit quantization mode...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.MEDGEMMA_MODEL_PATH,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=hf_token
                )
                print("Model successfully loaded on GPU!")
                self._is_loaded = True
                return
            except Exception as e:
                print(f"\n[GPU LOADING FAILED] {e}")
                print("Falling back to CPU mode...\n")

        print("Loading Model on CPU (bfloat16, ~8GB RAM)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MEDGEMMA_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cpu",
            token=hf_token
        )
        print("Model successfully loaded on CPU!")
        
        self._is_loaded = True

    def get_model_and_tokenizer(self):
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded yet. Call load_model(hf_token) first.")
        return self.model, self.tokenizer
