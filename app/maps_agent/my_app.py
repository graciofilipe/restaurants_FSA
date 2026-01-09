import os

# Enable Telemetry globally
os.environ["GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"] = "true"
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

from vertexai.agent_engines import AdkApp as BaseAdkApp
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from .agent import app as adk_app

class AdkApp(BaseAdkApp):
    def __init__(self, *args, env_vars=None, **kwargs):
        # Env vars already set globally, but keep logic if passed explicitly
        if env_vars:
            for k, v in env_vars.items():
                os.environ[k] = v
        super().__init__(*args, **kwargs)

app = AdkApp(
    app=adk_app,
    artifact_service=InMemoryArtifactService(),
    env_vars={
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
    }
)
