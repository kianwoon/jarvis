import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_name: str = "Qwen/Qwen-7B-Chat"):
    """Download and setup the Qwen model."""
    try:
        logger.info(f"Starting download of {model_name}")
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Download model
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        
        logger.info(f"Successfully downloaded {model_name}")
        logger.info(f"Model size: {model.get_memory_footprint() / 1024**3:.2f} GB")
        logger.info(f"Model device: {model.device}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, tokenizer = download_model()
        logger.info("Model download completed successfully!")
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        exit(1) 