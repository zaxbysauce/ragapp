"""
Model availability checker for Ollama endpoints.
"""
from typing import Dict, Any

import httpx

from app.config import settings


class ModelCheckerError(Exception):
    """Exception raised for model checker errors."""
    pass


class ModelChecker:
    """Checks availability of embedding and chat models via Ollama tags API."""
    
    def __init__(self, timeout: float = 10.0):
        """
        Initialize the model checker.
        
        Args:
            timeout: Request timeout in seconds (default: 10.0)
        """
        self.timeout = timeout
    
    async def check_models(self) -> Dict[str, Any]:
        """
        Check availability of configured embedding and chat models.
        
        Calls Ollama's /api/tags endpoint for both embedding and chat URLs
        to verify the configured models are available.
        
        Returns:
            Dictionary with 'embedding_model' and 'chat_model' keys,
            each containing a dict with 'available' (bool) and 'error' (str or None).
        
        Example:
            {
                'embedding_model': {'available': True, 'error': None},
                'chat_model': {'available': False, 'error': 'Model not found'}
            }
        """
        result = {
            'embedding_model': {'available': False, 'error': None},
            'chat_model': {'available': False, 'error': None}
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Check embedding model
            result['embedding_model'] = await self._check_model_availability(
                client,
                settings.ollama_embedding_url,
                settings.embedding_model
            )
            
            # Check chat model
            result['chat_model'] = await self._check_model_availability(
                client,
                settings.ollama_chat_url,
                settings.chat_model
            )
        
        return result
    
    async def _check_model_availability(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Check if a specific model is available at the given Ollama endpoint.
        
        Args:
            client: httpx AsyncClient instance
            base_url: Base URL of the Ollama endpoint
            model_name: Name of the model to check
        
        Returns:
            Dictionary with 'available' (bool) and 'error' (str or None)
        """
        url = f"{base_url.rstrip('/')}/api/tags"
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Parse models from response
            models = data.get('models', [])
            if not isinstance(models, list):
                return {
                    'available': False,
                    'error': "Invalid response format: 'models' is not a list"
                }
            
            # Check if model name matches any available model
            # Model names in Ollama may include tags (e.g., "qwen2.5:32b")
            # We check for exact match or if model_name is a prefix
            available_model_names = [m.get('name', '') for m in models]
            
            for available_name in available_model_names:
                if available_name == model_name or available_name.startswith(f"{model_name}:"):
                    return {'available': True, 'error': None}
            
            return {
                'available': False,
                'error': f"Model '{model_name}' not found. Available models: {', '.join(available_model_names) or 'none'}"
            }
        
        except httpx.TimeoutException:
            return {
                'available': False,
                'error': f"Request timed out after {self.timeout}s"
            }
        except httpx.HTTPStatusError as e:
            return {
                'available': False,
                'error': f"HTTP error {e.response.status_code}: {e.response.text}"
            }
        except httpx.RequestError as e:
            return {
                'available': False,
                'error': f"Request failed: {str(e)}"
            }
        except Exception as e:
            return {
                'available': False,
                'error': f"Unexpected error: {str(e)}"
            }


# Singleton instance for convenience
model_checker = ModelChecker()
