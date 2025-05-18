from typing import Dict, Any, Optional, List, AsyncGenerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import logging
from app.core.config import get_settings
from app.llm.base import LLMConfig, LLMResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCache:
    """Singleton cache for LLM models and tokenizers."""
    _instance = None
    _models: Dict[str, Any] = {}
    _tokenizers: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str) -> AutoModelForCausalLM:
        """Get or load model from cache, using MPS if available, and 8-bit quantization if on CUDA."""
        try:
            if model_name not in self._models:
                logger.info(f"Loading model: {model_name}")
                # Device selection
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    logger.info("Using MPS device (Apple Silicon GPU)")
                    self._models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map={"": device},
                        trust_remote_code=True,
                        torch_dtype=torch.float16,  # float16 for MPS
                        low_cpu_mem_usage=True
                    )
                    logger.warning("8-bit quantization is not supported on MPS. Using float16 instead.")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                    logger.info("Using CUDA device (NVIDIA GPU)")
                    self._models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        trust_remote_code=True,
                        load_in_8bit=True,  # 8-bit quantization for CUDA
                        low_cpu_mem_usage=True
                    )
                else:
                    device = torch.device("cpu")
                    logger.info("Using CPU device")
                    self._models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map={"": device},
                        trust_remote_code=True,
                        torch_dtype=torch.float32,  # Use float32 for CPU
                        low_cpu_mem_usage=True
                    )
                    logger.warning("8-bit quantization is not supported on CPU. Using float32 instead.")
                logger.info(f"Model {model_name} loaded successfully on {device}")
            return self._models[model_name]
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def get_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Get or load tokenizer from cache."""
        try:
            if model_name not in self._tokenizers:
                logger.info(f"Loading tokenizer: {model_name}")
                self._tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                # Robustly set pad token
                if self._tokenizers[model_name].pad_token is None:
                    self._tokenizers[model_name].pad_token = self._tokenizers[model_name].eos_token
                    self._tokenizers[model_name].pad_token_id = self._tokenizers[model_name].eos_token_id
                logger.info(f"Tokenizer {model_name} loaded successfully")
            return self._tokenizers[model_name]
        except Exception as e:
            logger.error(f"Error loading tokenizer {model_name}: {str(e)}")
            raise

class QwenInference:
    """Qwen LLM inference service."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.settings = get_settings()
        self.cache = ModelCache()
        logger.info(f"Initializing QwenInference with model: {config.model_name}")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer from cache."""
        try:
            self.model = self.cache.get_model(self.config.model_name)
            self.tokenizer = self.cache.get_tokenizer(self.config.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error in _load_model: {str(e)}")
            raise
    
    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Generate text from the model."""
        try:
            logger.info(f"Generating response for prompt: {prompt[:50]}...")
            
            # Ensure padding token is set
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Adjust based on model context window
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode and return
            response_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            logger.info("Generation completed successfully")
            return LLMResponse(
                text=response_text,
                metadata={
                    "model": self.config.model_name,
                    "generation_config": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "max_tokens": self.config.max_tokens
                    }
                }
            )
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate streaming text from the model."""
        try:
            logger.info(f"Starting streaming generation for prompt: {prompt[:50]}...")
            
            # Ensure padding token is set
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)
            
            # Setup streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_special_tokens=True
            )
            
            # Prepare generation kwargs
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Start generation in background thread
            thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs
            )
            thread.start()
            
            # Stream the output
            for text in streamer:
                yield LLMResponse(
                    text=text,
                    metadata={
                        "model": self.config.model_name,
                        "streaming": True
                    }
                )
            
            thread.join()
            logger.info("Streaming generation completed successfully")
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise RuntimeError(f"Streaming generation failed: {str(e)}")
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        # TODO: Implement embedding generation
        # This will require a separate embedding model
        raise NotImplementedError("Embedding generation not implemented yet") 