"""
Language Model wrapper for various LLM providers (OpenAI-compatible API only).

Supports:
- OpenAI API (including o1, o4-mini, etc.)
- SiliconFlow API
- 百炼平台 (Bailian / DashScope compatible-mode)
- Local LLM service (e.g. http://localhost:8000/v1/chat/completions, init with model_path="local_llm")

All providers use the OpenAI client; local vLLM is no longer supported.
"""

import threading
import time
import os
from typing import List, Dict, Optional, Union, Any
import openai
from openai import OpenAI

os.environ["SiliconFlow_API_KEY"] = "sk-thccdvsucvfhwxkhyplvgffhuufrayuqspwdotmytxxwighg"

# API key should be set via environment variable: SILICONFLOW_API_KEY
# Example: export SILICONFLOW_API_KEY="your-api-key"

class LLMConfig:
    """Configuration class for LLM initialization."""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        temperature: float = 0.1,
        top_k: int = 50,
        top_p: float = 1.0,
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize LLM configuration.
        
        Args:
            model_path: Model identifier (e.g., "openai/gpt-4", "siliconflow/deepseek-ai/DeepSeek-V3", "siliconflow/Pro/moonshotai/Kimi-K2.5")
            batch_size: Batch size for batch inference
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_tokens: Maximum tokens to generate
            system_prompt: Default system prompt
            api_key: API key (if None, will try to read from environment)
            base_url: Base URL for API (if None, will use default)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries (seconds)
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are an expert in traffic management. Use your knowledge and reasoning skills to "
            "solve the traffic signal timing problem. DO NOT write any code."
        )
        self.api_key = api_key
        self.base_url = base_url
        self.retry_count = retry_count
        self.retry_delay = retry_delay


class LLM:
    """
    Unified interface for LLM providers via OpenAI-compatible API only.
    
    Supports OpenAI API, SiliconFlow API, Bailian, and local LLM service.
    """

    # Provider configurations
    PROVIDER_CONFIGS = {
        "openai": {
            "api_key_envs": ["OPENAI_API_KEY"],
            "base_url": "https://api.openai.com/v1",
        },
        "siliconflow": {
            "api_key_envs": ["SILICONFLOW_API_KEY", "SiliconFlow_API_KEY"],
            "base_url": "https://api.siliconflow.cn/v1",
        },
        "local_llm": {
            "api_key_envs": [],  # No API key required for local service
            "base_url": "http://localhost:8000/v1",
        },
        "bailian": {
            "api_key_envs": ["DASHSCOPE_API_KEY", "BAILIAN_API_KEY"],
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
    }

    def __init__(
        self,
        model_path: str,
        batch_size: int = 16,
        temperature: float = 0.1,
        top_k: int = 50,
        top_p: float = 1.0,
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize LLM instance.
        
        Args:
            model_path: Model identifier. Format:
                - "openai/model_name" for OpenAI API
                - "siliconflow/institute/model_name" for SiliconFlow API
                - "siliconflow/Pro/institute/model_name" for SiliconFlow Pro models (e.g., "siliconflow/Pro/moonshotai/Kimi-K2.5")
                - "bailian/model_name" or "bailian/institute/model_name" for 百炼平台
                - "local_llm" or "v6/local_llm" for local LLM service (OpenAI-compatible endpoint)
            batch_size: Batch size for batch inference
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            max_tokens: Maximum tokens to generate
            system_prompt: Default system prompt
            api_key: API key (if None, will try to read from environment)
            base_url: Base URL for API (if None, will use default). For local_llm, default is http://localhost:8000/v1
            retry_count: Number of retries on failure
            retry_delay: Delay between retries (seconds)

        Local LLM: use model_path="local_llm" to use a local server at http://localhost:8000/v1/chat/completions.
        No API key or model name is required by the server; base_url can be overridden if your server runs elsewhere.
        """
        # Parse model path (only API providers supported)
        self.model_path = model_path
        path_parts = model_path.split("/")
        # Support local LLM: "local_llm" or "v6/local_llm" (version prefix)
        path_stripped = model_path.strip().lower()
        if path_stripped == "local_llm":
            self.provider = "local_llm"
            self.institute = None
            self.model_name = "local_llm"
            self.local_llm_version = "v4"  # default when no version given
        elif len(path_parts) == 2 and path_parts[-1].lower() == "local_llm":
            self.provider = "local_llm"
            self.institute = None
            self.model_name = "local_llm"
            self.local_llm_version = path_parts[0].strip()  # e.g. "v6"
        else:
            self.local_llm_version = None
            self.provider = path_parts[0].lower() if len(path_parts) > 1 else "local"
            self.institute = path_parts[-2] if len(path_parts) > 2 else None
            self.model_name = path_parts[-1]
        
        # If base_url is provided and model_path has no provider prefix, allow custom provider
        # This allows using third-party APIs with just model name like "o4-mini" instead of "openai/o4-mini"
        if base_url is not None and len(path_parts) == 1:
            # Use "custom" provider for third-party APIs with custom base_url
            self.provider = "custom"
            self.model_name = path_parts[0]
            self.institute = None
        elif self.provider not in self.PROVIDER_CONFIGS:
            raise ValueError(
                f"Only API providers are supported: {list(self.PROVIDER_CONFIGS.keys())}. "
                f"Got model_path={model_path!r}. Local vLLM is no longer supported. "
                f"If using a third-party API, provide base_url parameter to use model name without provider prefix."
            )

        # Configuration
        self.batch_size = batch_size
        self.system_prompt = system_prompt or (
            "You are an expert in traffic management. Use your knowledge and reasoning skills to "
            "solve the traffic signal timing problem. DO NOT write any code."
        )
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # Initialize model
        self.model, self.generation_kwargs, self.use_api = self._initialize_model(
            model_path, top_k, top_p, temperature, max_tokens, api_key, base_url
        )

    def _initialize_model(
        self,
        model_path: str,
        top_k: int,
        top_p: float,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str],
        base_url: Optional[str],
    ):
        """Initialize the LLM via OpenAI-compatible API (no local vLLM)."""
        generation_kwargs = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Handle custom provider (third-party API with custom base_url)
        if self.provider == "custom":
            # For custom provider, base_url must be provided
            if base_url is None:
                raise ValueError(
                    "base_url must be provided when using model name without provider prefix. "
                    f"Got model_path={model_path!r} without base_url."
                )
            # For custom provider, try to get API key from common environment variables if not provided
            if api_key is None:
                # Try common API key environment variables
                for env_name in ["OPENAI_API_KEY", "API_KEY"]:
                    api_key = os.environ.get(env_name)
                    if api_key:
                        break
                if api_key is None:
                    # Use a placeholder if no API key found (some third-party APIs might not require it)
                    api_key = "not-needed"
        else:
            config = self.PROVIDER_CONFIGS[self.provider]

            # Get API key (not required for local_llm)
            if api_key is None:
                api_key_envs = config.get("api_key_envs", [])
                for env_name in api_key_envs:
                    api_key = os.environ.get(env_name)
                    if api_key:
                        break
                if api_key is None and self.provider != "local_llm":
                    env_list = ", ".join(api_key_envs) if api_key_envs else "environment variables"
                    raise ValueError(
                        f"API key not provided and none of {env_list} found in environment variables"
                    )
                if api_key is None and self.provider == "local_llm":
                    api_key = "not-needed"  # Placeholder, local service does not use it

            # Get base URL
            if base_url is None:
                base_url = config["base_url"]

        # Initialize OpenAI client
        model = OpenAI(api_key=api_key, base_url=base_url, default_headers=None if "openai" not in self.provider else {"x-foo": "true"})
        return model, generation_kwargs, True

    def _get_api_model_name(self) -> str:
        """Get the formatted model name for API calls."""
        if self.provider == "custom":
            # For custom provider, return model name as-is
            return self.model_name
        if self.provider == "local_llm":
            version = self.local_llm_version or "v4"
            return f"../LlamaFactory/saves/qwen3-8b/full/sft/qwen3-8b-chatcity-{version}"
        if self.provider == "siliconflow":
            # Support Pro models: siliconflow/Pro/moonshotai/Kimi-K2.5 -> Pro/moonshotai/Kimi-K2.5
            # Parse the model_path to reconstruct the full path after "siliconflow/"
            path_parts = self.model_path.split("/")
            if len(path_parts) > 1 and path_parts[0].lower() == "siliconflow":
                # Return everything after "siliconflow/"
                return "/".join(path_parts[1:])
            return f"{self.institute}/{self.model_name}" if self.institute else self.model_name
        if self.provider == "bailian":
            return f"{self.institute}/{self.model_name}" if self.institute else self.model_name
        elif "deepseek" in self.model_name.lower():
            return f"deepseek-ai/{self.model_name}"
        else:
            return self.model_name

    def _needs_thinking_mode(self) -> bool:
        """Check if the model needs thinking mode enabled."""
        model_lower = self.model_name.lower()
        return (
            "deepseek-v3.2" in model_lower or
            ("qwen3" in model_lower and "thinking" not in model_lower and "instruct" not in model_lower) or
            "minmax" in model_lower
        )

    def _create_messages(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Create message list from query or use provided messages."""
        if messages is not None:
            return messages

        if query is None:
            raise ValueError("Either 'query' or 'messages' must be provided")

        return [
            {
                "role": "system",
                "content": system_prompt or self.system_prompt,
            },
            {
                "role": "user",
                "content": query
            }
        ]

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> Any:
        """Make API call with proper error handling and retries."""
        model_name = self._get_api_model_name()

        for attempt in range(self.retry_count):
            try:
                # Special handling for OpenAI o4-mini (no temperature)
                if self.provider == "openai" and self.model_name == "o4-mini":
                    kwargs = {
                        "model": model_name,
                        "messages": messages,
                        "max_completion_tokens": self.generation_kwargs["max_tokens"],
                    }
                elif self.provider == "openai":
                    kwargs = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": self.generation_kwargs["temperature"],
                        "max_completion_tokens": self.generation_kwargs["max_tokens"],
                    }
                elif self.provider == "local_llm":
                    kwargs = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": self.generation_kwargs["temperature"],
                        "max_tokens": self.generation_kwargs["max_tokens"]
                    }
                else:
                    # SiliconFlow or other providers
                    max_tokens = self.generation_kwargs["max_tokens"]
                    if "glm-z1-9b" in model_name.lower():
                        max_tokens = 8000

                    kwargs = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": self.generation_kwargs["temperature"],
                        "max_tokens": max_tokens,
                    }

                # Enable thinking mode for specific models (deepseek-V3.2, Qwen3, MinMax)
                # o4-mini does not support enable_think, so skip extra_body for it
                if self.provider.lower() != "openai":
                    kwargs["extra_body"] = {"enable_thinking": False}

                return self.model.chat.completions.create(**kwargs)

            except Exception as e:
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise RuntimeError(f"API call failed after {self.retry_count} attempts: {e}")

    def generate(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            query: User query string (if messages not provided)
            messages: List of message dictionaries with 'role' and 'content' keys
            system_prompt: System prompt (overrides default)

        Returns:
            Generated response text (includes reasoning if available)
        """
        messages = self._create_messages(query, messages, system_prompt)
        response = self._call_api(messages, stream=False)

        # Check for reasoning content in response
        collected_response = ""
        if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content:
            collected_response = "<think>\n"
            collected_response += response.choices[0].message.reasoning_content
            collected_response += "\n</think>\n"

        # Add main content
        if hasattr(response.choices[0].message, 'content') and response.choices[0].message.content:
            collected_response += response.choices[0].message.content

        return collected_response if collected_response else ""

    def generate_stream(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Generate a streaming response from the LLM.

        Args:
            query: User query string (if messages not provided)
            messages: List of message dictionaries with 'role' and 'content' keys
            system_prompt: System prompt (overrides default)

        Yields:
            Response chunks as they are generated
        """
        messages = self._create_messages(query, messages, system_prompt)
        stream = self._call_api(messages, stream=True)

        collected_response = "<think>\n"
        reasoning_finish_flag = False

        for chunk in stream:
            # Check for main content first (as in user's example)
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                if not reasoning_finish_flag:
                    collected_response += "</think>\n"
                    reasoning_finish_flag = True
                token = chunk.choices[0].delta.content
                collected_response += token
                yield token
            # Then check for reasoning content
            elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                token = chunk.choices[0].delta.reasoning_content
                collected_response += token
                # Don't yield reasoning content to user

    def batch_generate(
        self,
        queries: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Generate responses for multiple queries in batch.

        Args:
            queries: List of query strings
            messages_list: List of message lists (one per query)
            system_prompt: System prompt (overrides default)

        Returns:
            List of generated responses
        """
        if queries is not None:
            messages_list = [
                self._create_messages(query=q, system_prompt=system_prompt)
                for q in queries
            ]
        elif messages_list is None:
            raise ValueError("Either 'queries' or 'messages_list' must be provided")

        all_responses = []
        batch_messages = []

        for i, messages in enumerate(messages_list):
            batch_messages.append(messages)

            if len(batch_messages) == self.batch_size or i == len(messages_list) - 1:
                # Use threading for API calls
                model_name = self._get_api_model_name()
                threads = []
                responses = [None] * len(batch_messages)

                for j, msg in enumerate(batch_messages):
                    thread = threading.Thread(
                        target=self._threading_generate,
                        args=(model_name, msg, responses, j)
                    )
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                all_responses.extend(responses)
                batch_messages = []

        return all_responses

    def _threading_generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        response_list: List[Optional[str]],
        idx: int,
    ):
        """Thread worker for batch API generation."""
        response_list[idx] = ""

        for attempt in range(2):  # 2 retries for threading
            try:
                time.sleep(self.retry_delay)
                stream = self._call_api(messages, stream=True)

                collected_response = "<think>\n"
                reasoning_finish_flag = False

                for chunk in stream:
                    # Check for main content first (as in user's example)
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        if not reasoning_finish_flag:
                            collected_response += "</think>\n"
                            reasoning_finish_flag = True
                        token = chunk.choices[0].delta.content
                        collected_response += token
                    # Then check for reasoning content
                    elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        token = chunk.choices[0].delta.reasoning_content
                        collected_response += token

                response_list[idx] = collected_response
                print(f"\nSuccess [{idx}].")
                break

            except Exception as e:
                if attempt < 1:
                    continue
                print(f"Error in threading_generate [{idx}]: {e}")
                response_list[idx] = ""

    # Backward compatibility aliases
    def inference(self, query: str, system_prompt: Optional[str] = None) -> str:
        """Backward compatibility: alias for generate()."""
        return self.generate(query=query, system_prompt=system_prompt)

    def inference_messages(self, messages: List[Dict[str, str]]) -> str:
        """Backward compatibility: alias for generate()."""
        return self.generate(messages=messages)

    def batch_inference(self, queries: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """Backward compatibility: alias for batch_generate()."""
        return self.batch_generate(queries=queries, system_prompt=system_prompt)

    def batch_inference_messages(self, messages_list: List[List[Dict[str, str]]]) -> List[str]:
        """Backward compatibility: alias for batch_generate()."""
        return self.batch_generate(messages_list=messages_list)
