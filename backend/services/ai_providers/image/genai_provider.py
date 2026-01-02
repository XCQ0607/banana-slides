"""
Google GenAI SDK implementation for image generation

Supports two modes:
- Google AI Studio: Uses API key authentication
- Vertex AI: Uses GCP service account authentication
"""
import logging
from typing import Optional, List
from google import genai
from google.genai import types
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import ImageProvider
from config import get_config

logger = logging.getLogger(__name__)


class GenAIImageProvider(ImageProvider):
    """Image generation using Google GenAI SDK (supports both AI Studio and Vertex AI)"""

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model: str = "gemini-3-pro-image-preview",
        vertexai: bool = False,
        project_id: str = None,
        location: str = None
    ):
        """
        Initialize GenAI image provider

        Args:
            api_key: Google API key (for AI Studio mode)
            api_base: API base URL (for proxies like aihubmix, AI Studio mode only)
            model: Model name to use
            vertexai: If True, use Vertex AI instead of AI Studio
            project_id: GCP project ID (required for Vertex AI mode)
            location: GCP region (for Vertex AI mode, default: us-central1)
        """
        timeout_ms = int(get_config().GENAI_TIMEOUT * 1000)

        if vertexai:
            # Vertex AI mode - uses service account credentials from GOOGLE_APPLICATION_CREDENTIALS
            logger.info(f"Initializing GenAI image provider in Vertex AI mode, project: {project_id}, location: {location}")
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location or 'us-central1',
                http_options=types.HttpOptions(timeout=timeout_ms)
            )
        else:
            # AI Studio mode - uses API key
            http_options = types.HttpOptions(
                base_url=api_base,
                timeout=timeout_ms
            ) if api_base else types.HttpOptions(timeout=timeout_ms)

            self.client = genai.Client(
                http_options=http_options,
                api_key=api_key
            )

        self.model = model
    
    @retry(
        stop=stop_after_attempt(get_config().GENAI_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_image(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "2K",
        enable_thinking: bool = True
    ) -> Optional[Image.Image]:
        """
        Generate image using Google GenAI SDK
        
        Args:
            prompt: The image generation prompt
            ref_images: Optional list of reference images
            aspect_ratio: Image aspect ratio
            resolution: Image resolution (supports "1K", "2K", "4K")
            enable_thinking: If True, enable thinking chain mode (may generate multiple images)
            
        Returns:
            Generated PIL Image object, or None if failed
        """
        try:
            # Build contents list with prompt and reference images
            contents = []
            
            # Add reference images first (if any)
            if ref_images:
                for ref_img in ref_images:
                    contents.append(ref_img)
            
            # Add text prompt
            contents.append(prompt)
            
            logger.debug(f"Calling GenAI API for image generation with {len(ref_images) if ref_images else 0} reference images...")
            logger.debug(f"Config - aspect_ratio: {aspect_ratio}, resolution: {resolution}, enable_thinking: {enable_thinking}")
            
            # Build config
            # Build config
            image_config_args = {'aspect_ratio': aspect_ratio}
            logging.info(f"DEBUG: GenAIProvider - Received resolution={resolution}")
            if resolution:
                image_config_args['image_size'] = resolution
                logging.info(f"DEBUG: GenAIProvider - Added image_size={resolution} to config")
            else:
                logging.info("DEBUG: GenAIProvider - Skipped image_size (resolution is None or empty)")
                
            config_params = {
                'response_modalities': ['TEXT', 'IMAGE'],
                'image_config': types.ImageConfig(**image_config_args)
            }
            
            # Add thinking config if enabled
            if enable_thinking:
                config_params['thinking_config'] = types.ThinkingConfig(
                    include_thoughts=True
                )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**config_params)
            )
            
            logger.debug("GenAI API call completed")
            
            # Extract the final image from the response.
            # Earlier images are usually low resolution drafts 
            # Therefore, always use the last image found.
            last_image = None
            
            for i, part in enumerate(response.parts):
                if part.text is not None:
                    logger.debug(f"Part {i}: TEXT - {part.text[:100] if len(part.text) > 100 else part.text}")
                else:
                    try:
                        logger.debug(f"Part {i}: Attempting to extract image...")
                        image = part.as_image()
                        if image:
                            logger.debug(f"Successfully extracted image from part {i}")
                            last_image = image
                    except Exception as e:
                        logger.debug(f"Part {i}: Failed to extract image - {str(e)}")
            
            # Return the last image found (highest quality in thinking chain scenarios)
            if last_image:
                return last_image
            
            # No image found in response
            error_msg = "No image found in API response. "
            if response.parts:
                error_msg += f"Response had {len(response.parts)} parts but none contained valid images."
            else:
                error_msg += "Response had no parts."
            
            raise ValueError(error_msg)
            
        except Exception as e:
            logger.warning(f"SDK generation failed: {str(e)}. Attempting raw HTTP fallback...")
            try:
                return self._generate_image_raw(prompt, ref_images, aspect_ratio, resolution, enable_thinking)
            except Exception as raw_e:
                error_detail = f"Both SDK and Raw HTTP generation failed. SDK Error: {str(e)}. Raw Error: {str(raw_e)}"
                logger.error(error_detail, exc_info=True)
                raise Exception(error_detail) from e

    def _generate_image_raw(
        self,
        prompt: str,
        ref_images: Optional[List[Image.Image]] = None,
        aspect_ratio: str = "16:9",
        resolution: str = "2K",
        enable_thinking: bool = True
    ) -> Optional[Image.Image]:
        """
        Generate image using raw HTTP request (robust fallback)
        """
        import requests
        import json
        import base64
        from io import BytesIO

        logger.info("Starting raw HTTP image generation...")

        # Construct URL
        if hasattr(self.client, '_api_key') and self.client._api_key:
             # AI Studio mode
            base_url = "https://generativelanguage.googleapis.com"
            if hasattr(self.client, '_http_options') and self.client._http_options.base_url:
                base_url = self.client._http_options.base_url.rstrip('/')
            
            url = f"{base_url}/v1beta/models/{self.model}:generateContent?key={self.client._api_key}"
        else:
            # Vertex AI mode (not fully supported in raw fallback yet, but we can try)
            # For now, if we are here, it's likely AI Studio / Proxy usage
            raise NotImplementedError("Raw HTTP fallback currently only supports AI Studio/API Key mode")

        # Construct Payload
        contents_parts = []
        
        # Add reference images
        if ref_images:
            for img in ref_images:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                contents_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_str
                    }
                })

        # Add prompt
        contents_parts.append({"text": prompt})

        payload = {
            "contents": [{"parts": contents_parts}],
            "generationConfig": {
                "response_modalities": ["TEXT", "IMAGE"],
                "image_config": {"aspect_ratio": aspect_ratio}
            }
        }

        if resolution:
             payload["generationConfig"]["image_config"]["image_size"] = resolution

        if enable_thinking:
             payload["generationConfig"]["thinking_config"] = {"include_thoughts": True}

        # Send Request
        try:
            logger.debug(f"Sending raw POST request to {url.split('?')[0]}...")
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            # Loose Parsing Logic
            try:
                data = response.json()
            except:
                # If not JSON, maybe it's raw text?
                raise Exception(f"Invalid JSON response: {response.text[:200]}")

            # 1. Try Standard Google Format (candidates -> content -> parts -> inline_data)
            try:
                if 'candidates' in data:
                    for candidate in data['candidates']:
                        if 'content' in candidate and 'parts' in candidate['content']:
                            for part in candidate['content']['parts']:
                                if 'inline_data' in part:
                                    b64_data = part['inline_data']['data']
                                    return Image.open(BytesIO(base64.b64decode(b64_data)))
                                # Handle camelCase variation just in case
                                if 'inlineData' in part:
                                    b64_data = part['inlineData']['data']
                                    return Image.open(BytesIO(base64.b64decode(b64_data)))
            except Exception as e:
                logger.debug(f"Failed to parse standard format: {e}")

            # 2. Try Imagen Format (predictions -> bytesBase64Encoded)
            try:
                if 'predictions' in data:
                    for pred in data['predictions']:
                        if 'bytesBase64Encoded' in pred:
                            return Image.open(BytesIO(base64.b64decode(pred['bytesBase64Encoded'])))
            except Exception as e:
                logger.debug(f"Failed to parse Imagen format: {e}")

            # 3. Try 'images' array (common proxy format)
            try:
                if 'images' in data and isinstance(data['images'], list) and len(data['images']) > 0:
                    return Image.open(BytesIO(base64.b64decode(data['images'][0])))
            except Exception as e:
                logger.debug(f"Failed to parse 'images' array: {e}")

            # 4. Try 'data' field (single base64 string)
            try:
                if 'data' in data and isinstance(data['data'], str):
                     return Image.open(BytesIO(base64.b64decode(data['data'])))
            except Exception as e:
                logger.debug(f"Failed to parse 'data' field: {e}")

            # 5. Try 'output' field
            try:
                if 'output' in data:
                     # Could be list or string
                     if isinstance(data['output'], list) and len(data['output']) > 0:
                         return Image.open(BytesIO(base64.b64decode(data['output'][0])))
                     if isinstance(data['output'], str):
                         return Image.open(BytesIO(base64.b64decode(data['output'])))
            except Exception as e:
                logger.debug(f"Failed to parse 'output' field: {e}")

            logger.error(f"Could not find image data in response: {str(data)[:500]}")
            raise ValueError("No recognizable image data found in response")

        except Exception as e:
            logger.error(f"Raw HTTP request failed: {str(e)}")
            raise

