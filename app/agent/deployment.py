import os
from vertexai.agent_engines import AdkApp as BaseAdkApp
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from .maps_agent import app as adk_app

class AdkApp(BaseAdkApp):
    def __init__(self, *args, env_vars=None, **kwargs):
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
