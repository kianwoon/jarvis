import httpx
from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from typing import AsyncGenerator, List, Dict, Union, Optional
import json
import os
import logging
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

logger = logging.getLogger(__name__)

class OllamaLLM(BaseLLM):
    def __init__(self, config: LLMConfig, base_url: str = None):
        super().__init__(config)
        
        # If no base_url provided, get from settings
        if not base_url:
            from app.core.llm_settings_cache import get_main_llm_full_config
            main_llm_config = get_main_llm_full_config()
            base_url = main_llm_config.get('model_server', '')
            
            if not base_url:
                # Use environment variable as last resort
                base_url = os.environ.get("OLLAMA_BASE_URL", "")
                
            if not base_url:
                raise ValueError("Model server URL must be configured in LLM settings or provided as parameter")
        
        # Docker environment detection and URL conversion
        is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER")
        
        # Convert localhost to host.docker.internal for Docker if needed
        if is_docker and "localhost" in base_url:
            original_url = base_url
            base_url = base_url.replace("localhost", "host.docker.internal")
            logger.info(f"[OllamaLLM] Docker detected - converted URL from {original_url} to {base_url}")
        
        self.base_url = base_url
        self.model_name = config.model_name
        self.is_docker = is_docker
        
        # Log initialization details for debugging
        logger.debug(f"[OllamaLLM] Initialized with model={config.model_name}, base_url={base_url}, docker={is_docker}")

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using chat endpoint for better prompt handling.
        
        This method now uses the chat endpoint internally to properly separate
        system prompts from user prompts.
        """
        
        # Use chat endpoint with message format
        return await self.chat(prompt, **kwargs)

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Generate stream using chat endpoint for better prompt handling.
        
        This method now uses the chat endpoint internally to properly separate
        system prompts from user prompts.
        """
        # Use chat_stream endpoint with message format
        async for response in self.chat_stream(prompt, **kwargs):
            yield response

    async def embed(self, text: str):
        raise NotImplementedError("Embedding is not supported for OllamaLLM.")
    
    def _convert_prompt_to_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Convert a single prompt string to messages format.
        
        If system_prompt is provided, it will be used as the system message.
        Otherwise, we'll try to extract it from the prompt if it follows certain patterns.
        """
        messages = []
        
        
        # If system prompt is explicitly provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return messages
        
        # Try to detect if the prompt has a system prompt section
        # Look for common patterns that indicate system instructions
        llm_settings = get_llm_settings()
        default_system_prompt = llm_settings.get('main_llm', {}).get('system_prompt', '')
        
        if default_system_prompt and prompt.startswith(default_system_prompt):
            # Extract system prompt and remaining content
            remaining_prompt = prompt[len(default_system_prompt):].strip()
            messages.append({"role": "system", "content": default_system_prompt})
            if remaining_prompt:
                messages.append({"role": "user", "content": remaining_prompt})
        else:
            # No clear separation, treat entire prompt as user message
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def chat(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Chat completion using Ollama's /api/chat endpoint.
        
        Args:
            prompt: Either a string (will be converted to messages) or a list of message dicts
            **kwargs: Additional arguments like system_prompt for string prompts
        """
        
        # Get context length from settings
        llm_settings = get_llm_settings()
        context_length = llm_settings.get("context_length", 128000)
        
        # Convert prompt to messages if it's a string
        if isinstance(prompt, str):
            system_prompt = kwargs.get('system_prompt')
            messages = self._convert_prompt_to_messages(prompt, system_prompt)
        else:
            messages = prompt
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
                "num_ctx": context_length,
            }
        }
        
        
        # Use centralized timeout configuration
        from app.core.timeout_settings_cache import get_timeout_value
        http_timeout = get_timeout_value("api_network", "http_request_timeout", 30)
        
        
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            # Chat endpoint returns message object
            message = data.get("message", {})
            return LLMResponse(
                text=message.get("content", ""),
                metadata={
                    "model": self.model_name,
                    "done": data.get("done", True),
                    "role": message.get("role", "assistant")
                }
            )
    
    async def chat_stream(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Streaming chat completion using Ollama's /api/chat endpoint."""
        # Get context length from settings
        llm_settings = get_llm_settings()
        context_length = llm_settings.get("context_length", 128000)
        
        # Convert prompt to messages if it's a string
        if isinstance(prompt, str):
            system_prompt = kwargs.get('system_prompt')
            messages = self._convert_prompt_to_messages(prompt, system_prompt)
        else:
            messages = prompt
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
                "num_ctx": context_length,
            }
        }
        
        
        try:
            from app.core.timeout_settings_cache import get_timeout_value
            timeout = get_timeout_value("llm_ai", "llm_streaming_timeout", 120)
            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise Exception(f"Ollama API error {response.status_code}: {error_text.decode()}")
                    
                    token_count = 0
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data_json = json.loads(line)
                            message = data_json.get("message", {})
                            if message and "content" in message:
                                token_count += 1
                                content = message['content']
                                yield LLMResponse(
                                    text=message["content"],
                                    metadata={
                                        "model": self.model_name,
                                        "streaming": True,
                                        "role": message.get("role", "assistant")
                                    }
                                )
                            elif data_json.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            continue
                        except Exception as e:
                            continue
                    
        except httpx.ConnectError as e:
            raise Exception(f"Cannot connect to Ollama server at {self.base_url}. Is it running?")
        except httpx.TimeoutException as e:
            raise Exception(f"Request to Ollama timed out. Model might be loading or overloaded.")
        except Exception as e:
            raise

class JarvisLLM:
    def __init__(self, mode=None, max_tokens=None, base_url: str = None):
        # Default to 'non-thinking' if mode is not specified
        self.mode = mode if mode in ("thinking", "non-thinking") else "non-thinking"
        
        # If no base_url provided, get from settings
        if not base_url:
            from app.core.llm_settings_cache import get_main_llm_full_config
            main_llm_config = get_main_llm_full_config()
            base_url = main_llm_config.get('model_server', '')
            
            if not base_url:
                # Use environment variable as last resort
                base_url = os.environ.get("OLLAMA_BASE_URL", "")
                
            if not base_url:
                raise ValueError("Model server URL must be configured in LLM settings")
        
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.llm = self._build_llm(self.mode, self.max_tokens)

    def _build_llm(self, mode, max_tokens):
        settings = get_llm_settings()
        
        # Use helper function to get full LLM configuration
        mode_settings = get_main_llm_full_config(settings)
        
        # Check required fields in mode config
        required_params = ["model", "max_tokens", "temperature", "top_p"]
        missing = [f for f in required_params if f not in mode_settings or mode_settings[f] is None]
        if missing:
            raise RuntimeError(f"Missing required LLM config fields in {mode} mode: {', '.join(missing)}")
        # Handle both string and int values for max_tokens
        if max_tokens is not None:
            max_tokens_value = int(max_tokens)
        else:
            max_tokens_raw = mode_settings["max_tokens"]
            try:
                max_tokens_value = int(max_tokens_raw)
            except (ValueError, TypeError):
                max_tokens_value = 16384
                
        config = LLMConfig(
            model_name=mode_settings["model"],
            temperature=float(mode_settings["temperature"]),
            top_p=float(mode_settings["top_p"]),
            max_tokens=max_tokens_value
        )
        return OllamaLLM(config, base_url=self.base_url)

    def set_mode(self, mode=None, max_tokens=None):
        self.mode = mode if mode in ("thinking", "non-thinking") else "non-thinking"
        self.max_tokens = max_tokens
        self.llm = self._build_llm(self.mode, self.max_tokens)

    async def invoke(self, prompt: str) -> str:
        response = await self.llm.generate(prompt)
        return response.text 