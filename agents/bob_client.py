"""
bob_client.py — Shared IBM watsonx.ai client factory.

All agents import `get_model()` from here to avoid duplicating
credential setup and to make it easy to swap model IDs via env vars.
"""
import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()


def get_model() -> ModelInference:
    """Return a ready-to-use watsonx ModelInference instance."""
    api_key    = os.environ["WATSONX_API_KEY"]
    url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    project_id = os.environ["WATSONX_PROJECT_ID"]
    model_id   = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

    credentials = Credentials(api_key=api_key, url=url)

    return ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
    )


def chat(model: ModelInference, prompt: str) -> str:
    """
    Send a single user message and return the assistant text.

    Keeps the call-site identical across all agents:
        text = chat(model, prompt)
    """
    messages = [{"role": "user", "content": prompt}]
    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]
