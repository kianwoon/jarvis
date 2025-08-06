import httpx
from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from typing import AsyncGenerator, List, Dict, Union, Optional
import json
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

class OllamaLLM(BaseLLM):
    def __init__(self, config: LLMConfig, base_url: str = "http://localhost:11434"):
        print(f"[OLLAMA_INIT] ============ OllamaLLM INITIALIZING ============")
        print(f"[OLLAMA_INIT] Model: {config.model_name}")
        print(f"[OLLAMA_INIT] Base URL: {base_url}")
        print(f"[OLLAMA_INIT] Max Tokens: {config.max_tokens}")
        print(f"[OLLAMA_INIT] Temperature: {config.temperature}")
        import traceback
        print(f"[OLLAMA_INIT] Called from:\n{''.join(traceback.format_stack()[-5:-1])}")
        print(f"[OLLAMA_INIT] ============================================")
        super().__init__(config)
        self.base_url = base_url
        self.model_name = config.model_name

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using chat endpoint for better prompt handling.
        
        This method now uses the chat endpoint internally to properly separate
        system prompts from user prompts.
        """
        # Debug: Check if this is related to query expansion
        is_query_expansion = any(term in prompt.lower() for term in ['alternative', 'query', 'search', 'rephrase', 'variation'])
        if is_query_expansion:
            print(f"[QUERY_EXPANSION_OLLAMA] GENERATE method called for query expansion")
            print(f"[QUERY_EXPANSION_OLLAMA] Prompt: {prompt[:200]}...")
            print(f"[QUERY_EXPANSION_OLLAMA] Base URL: {self.base_url}")
            print(f"[QUERY_EXPANSION_OLLAMA] Model: {self.model_name}")
        
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
        
        # Debug logging
        print(f"[DEBUG CONVERT] Converting prompt to messages")
        print(f"[DEBUG CONVERT] System prompt provided: {system_prompt is not None}")
        print(f"[DEBUG CONVERT] Prompt length: {len(prompt)} chars")
        
        # If system prompt is explicitly provided
        if system_prompt:
            print(f"[DEBUG CONVERT] Using provided system prompt: {system_prompt[:100]}...")
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return messages
        
        # Try to detect if the prompt has a system prompt section
        # Look for common patterns that indicate system instructions
        llm_settings = get_llm_settings()
        default_system_prompt = llm_settings.get('main_llm', {}).get('system_prompt', '')
        print(f"[DEBUG CONVERT] Default system prompt from settings: {default_system_prompt[:100] if default_system_prompt else 'None'}...")
        
        if default_system_prompt and prompt.startswith(default_system_prompt):
            # Extract system prompt and remaining content
            remaining_prompt = prompt[len(default_system_prompt):].strip()
            print(f"[DEBUG CONVERT] Detected system prompt in prompt string, extracting...")
            messages.append({"role": "system", "content": default_system_prompt})
            if remaining_prompt:
                messages.append({"role": "user", "content": remaining_prompt})
        else:
            # No clear separation, treat entire prompt as user message
            print(f"[DEBUG CONVERT] No system prompt detected, treating entire prompt as user message")
            messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def chat(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> LLMResponse:
        """Chat completion using Ollama's /api/chat endpoint.
        
        Args:
            prompt: Either a string (will be converted to messages) or a list of message dicts
            **kwargs: Additional arguments like system_prompt for string prompts
        """
        # Debug: Check if this is related to query expansion
        if isinstance(prompt, str):
            is_query_expansion = any(term in prompt.lower() for term in ['alternative', 'query', 'search', 'rephrase', 'variation'])
            if is_query_expansion:
                print(f"[QUERY_EXPANSION_OLLAMA] CHAT method called for query expansion")
                print(f"[QUERY_EXPANSION_OLLAMA] Prompt: {prompt[:200]}...")
                print(f"[QUERY_EXPANSION_OLLAMA] Base URL: {self.base_url}")
                print(f"[QUERY_EXPANSION_OLLAMA] Model: {self.model_name}")
        
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
        
        # Debug logging for messages
        print(f"[DEBUG OLLAMA CHAT] Sending {len(messages)} messages to Ollama")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content_preview = msg.get('content', '')[:200] + '...' if len(msg.get('content', '')) > 200 else msg.get('content', '')
            print(f"[DEBUG OLLAMA CHAT] Message {i+1} - Role: {role}")
            print(f"[DEBUG OLLAMA CHAT] Message {i+1} - Content preview: {content_preview}")
            print(f"[DEBUG OLLAMA CHAT] Message {i+1} - Full length: {len(msg.get('content', ''))} chars")
        
        # Use centralized timeout configuration
        from app.core.timeout_settings_cache import get_timeout_value
        http_timeout = get_timeout_value("api_network", "http_request_timeout", 30)
        
        # Add logging for HTTP request
        print(f"[OLLAMA_HTTP] Making POST request to {self.base_url}/api/chat")
        print(f"[OLLAMA_HTTP] Model: {self.model_name}")
        print(f"[OLLAMA_HTTP] Payload preview: {str(payload)[:300]}...")
        
        async with httpx.AsyncClient(timeout=http_timeout) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            print(f"[OLLAMA_HTTP] Response status: {response.status_code}")
            response.raise_for_status()
            
            data = response.json()
            # Chat endpoint returns message object
            message = data.get("message", {})
            print(f"[OLLAMA_HTTP] Response received, content length: {len(message.get('content', ''))}")
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
        
        print(f"[DEBUG] Ollama chat payload - model: {self.model_name}, num_predict: {self.config.max_tokens}, num_ctx: {context_length}")
        # CRITICAL: Explicit high-token logging to detect truncation issues
        if self.config.max_tokens >= 8000:
            print(f"[OLLAMA HIGH TOKENS] CRITICAL: Using {self.config.max_tokens} tokens - should prevent truncation")
            print(f"[OLLAMA HIGH TOKENS] Model: {self.model_name}")
            print(f"[OLLAMA HIGH TOKENS] Context length: {context_length}")
        print(f"[DEBUG] Ollama URL: {self.base_url}/api/chat")
        
        print(f"[DEBUG] Attempting connection to {self.base_url}/api/chat")
        print(f"[DEBUG] Payload: {json.dumps(payload, indent=2)[:500]}...")
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout
                print(f"[DEBUG] HTTP client created, making request...")
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    print(f"[DEBUG] HTTP response status: {response.status_code}")
                    if response.status_code != 200:
                        error_text = await response.aread()
                        print(f"[ERROR] Ollama API error {response.status_code}: {error_text.decode()}")
                        raise Exception(f"Ollama API error {response.status_code}: {error_text.decode()}")
                    
                    token_count = 0
                    print(f"[DEBUG] Starting to read streaming response...")
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data_json = json.loads(line)
                            message = data_json.get("message", {})
                            if message and "content" in message:
                                token_count += 1
                                content = message['content']
                                if token_count <= 5:  # Log first 5 tokens
                                    print(f"[DEBUG] Token {token_count}: '{content}'")
                                elif token_count % 50 == 0:  # Log every 50th token
                                    print(f"[DEBUG] Generated {token_count} tokens so far...")
                                
                                # CRITICAL: Check for potential truncation patterns
                                if token_count >= 2500 and self.config.max_tokens >= 8000:
                                    print(f"[TRUNCATION CHECK] Token {token_count}: Still generating (target: {self.config.max_tokens})")
                                if "distribut..." in content or (content.strip().endswith("...") and len(content.strip()) > 10):
                                    print(f"[TRUNCATION DETECTED] Response appears to be cut off at token {token_count}: '{content[-50:]}'")
                                    print(f"[TRUNCATION DETECTED] Expected {self.config.max_tokens} tokens, got truncation pattern")
                                yield LLMResponse(
                                    text=message["content"],
                                    metadata={
                                        "model": self.model_name,
                                        "streaming": True,
                                        "role": message.get("role", "assistant")
                                    }
                                )
                            elif data_json.get("done", False):
                                print(f"[DEBUG] Stream completed. Total tokens generated: {token_count}")
                                if self.config.max_tokens >= 8000:
                                    print(f"[HIGH TOKEN COMPLETION] Expected {self.config.max_tokens}, generated {token_count} tokens")
                                    if token_count < (self.config.max_tokens * 0.7):
                                        print(f"[TOKEN WARNING] Generated {token_count} tokens, but expected up to {self.config.max_tokens}. Possible early termination?")
                        except json.JSONDecodeError as e:
                            print(f"[WARNING] JSON decode error on line: '{line}' - {e}")
                            continue
                        except Exception as e:
                            print(f"[ERROR] Unexpected error processing stream line: {e}")
                            print(f"[ERROR] Problematic line: '{line}'")
                            continue
                    
                    if token_count == 0:
                        print(f"[ERROR] No tokens generated! Check if model is loaded and responding.")
        except httpx.ConnectError as e:
            print(f"[ERROR] Connection failed to {self.base_url}: {e}")
            raise Exception(f"Cannot connect to Ollama server at {self.base_url}. Is it running?")
        except httpx.TimeoutException as e:
            print(f"[ERROR] Request timed out: {e}")
            raise Exception(f"Request to Ollama timed out. Model might be loading or overloaded.")
        except Exception as e:
            print(f"[ERROR] Unexpected error in chat_stream: {e}")
            raise

class JarvisLLM:
    def __init__(self, mode=None, max_tokens=None, base_url: str = "http://localhost:11434"):
        # Default to 'non-thinking' if mode is not specified
        self.mode = mode if mode in ("thinking", "non-thinking") else "non-thinking"
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
                print(f"[WARNING] Invalid max_tokens in settings: {max_tokens_raw}, using 16384")
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
        # Debug: Track JarvisLLM invoke calls
        import traceback
        caller_info = traceback.format_stack()[-2]
        print(f"[JARVIS_LLM_INVOKE] Called from: {caller_info.strip()}")
        print(f"[JARVIS_LLM_INVOKE] Prompt: {prompt[:200]}...")
        
        response = await self.llm.generate(prompt)
        return response.text 