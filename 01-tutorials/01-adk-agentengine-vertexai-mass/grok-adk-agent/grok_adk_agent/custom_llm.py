# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dotenv import load_dotenv
from typing import Any, Dict, AsyncGenerator
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.auth import default
import google.auth.transport.requests


load_dotenv()
_, default_project_id = google.auth.default()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", default_project_id)
LOCATION = os.getenv("VERTEXAI_LOCATION", "global")
API_BASE = f"https://{LOCATION}/aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"

if LOCATION == "global":
    API_BASE = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"

credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

#---------------------------------------------------------------#
# Custom Model Wrapper for Grok
#---------------------------------------------------------------#
class GrokLlm(LiteLlm):
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Dev-Server-safe wrapper around LiteLlm.
        Overrides serialization to prevent `adk web` from crashing on llm_client.
        Required until PR#5375 (https://github.com/google/adk-python/pull/5375) is merged in adk-python.
        """
        return {
            "model": self.model,
            "type": "LiteLlm (Web Safe Workaround)"
        }

#---------------------------------------------------------------#
# Custom OpenAI Compatible LLM
#---------------------------------------------------------------#
class CustomOpenAICompatibleLlm(BaseLlm):
    #---------------------------------------------------------------#
    # Constructor
    #---------------------------------------------------------------#
    def __init__(self, model:str, **kwargs):
        """
        Constructor for the custom OpenAI compatible LLM.
        Args:
            model (str): The model string.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model=model, **kwargs)
        self.model = model
        
    #---------------------------------------------------------------#
    # Generate Content Async
    #---------------------------------------------------------------#
    async def generate_content_async( self, llm_request: LlmRequest, stream: bool = True) -> AsyncGenerator[LlmResponse, None]:
        """
        Generates content asynchronously using the Grok model.
        Args:
            llm_request (LlmRequest): The LLM request.
            stream (bool): Whether to stream the content.
        Returns:
            AsyncGenerator[LlmResponse, None]: Async generator for LLM responses.
        """
        _model = self.model
        if(llm_request.model):
            _model = llm_request.model
        
        llm_engine = GrokLlm(
            model=_model,
            api_key=self._get_token(),
            api_base=API_BASE
        )

        async for chunk in llm_engine.generate_content_async(llm_request, stream):
            yield chunk
        
    #---------------------------------------------------------------#
    # Generate Access Token
    #---------------------------------------------------------------#
    def _get_token(self) -> str:
        """
        Returns access token to call VertexAI MaaS endpoint after refresh if needed.
        Args:
            None
        Returns:
            str: Access token.
        """
        if not credentials.valid:
            credentials.refresh(google.auth.transport.requests.Request())
        return credentials.token