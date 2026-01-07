#!/usr/bin/env python3
import argparse
import sys
import vertexai
from vertexai.preview import reasoning_engines
import subprocess
import os
import json
import logging
import time
import threading
from datetime import datetime
import requests
import google.auth
import google.auth.transport.requests

# Default configuration
PROJECT_ID = "filipegracio-ai-learning"
LOCATION = "us-central1"
STAGING_BUCKET = "filipegracio-ai-learning-agent-engine"
LOG_NAME = "agent-deployments"
POLL_INTERVAL = 30 # seconds
DEPLOY_TIMEOUT = 1800 # 30 minutes

def setup_cloud_logging():
    """Sets up Google Cloud Logging."""
    try:
        import google.cloud.logging
        client = google.cloud.logging.Client(project=PROJECT_ID)
        client.setup_logging()
        logger = logging.getLogger(LOG_NAME)
        logger.setLevel(logging.INFO)
        return logger
    except ImportError:
        print("Warning: google-cloud-logging not found. Skipping Cloud Logging.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Failed to setup Cloud Logging: {e}", file=sys.stderr)
        return None

logger = setup_cloud_logging()

def log(message, level="info"):
    """Logs to both console and Cloud Logging."""
    print(message)
    if logger:
        if level == "error":
            logger.error(message)
        else:
            logger.info(message)

def get_agent_status(agent_id):
    """Retrieves the status of the Reasoning Engine."""
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        resources = reasoning_engines.ReasoningEngine.list()
        for r in resources:
            if r.resource_name.endswith(f"/{agent_id}"):
                state = "UNKNOWN"
                if hasattr(r, "state"):
                    state = str(r.state)
                elif hasattr(r, "gca_resource") and hasattr(r.gca_resource, "state"):
                    state = str(r.gca_resource.state)
                return state, r
        return "NOT_FOUND", None
    except Exception as e:
        log(f"Error checking status: {e}", "error")
        return "ERROR", None

def get_auth_token():
    """Retrieves Google auth token."""
    credentials, project = google.auth.default()
    auth_request = google.auth.transport.requests.Request()
    credentials.refresh(auth_request)
    return credentials.token

def verify_agent_health(agent_id):
    """Sends a ping query to the agent to verify functionality."""
    log(f"Verifying health for agent {agent_id}...")
    try:
        token = get_auth_token()
        # Use streamQuery as ADK agents typically support it
        url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{agent_id}:streamQuery"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Simple ping query
        payload = {
            "input": {
                "message": {
                    "role": "user",
                    "parts": [{"text": "Hello"}]
                },
                "user_id": "health_check_user"
            }
        }
        
        # Stream=True to handle streaming response, but we check status code first
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        
        if response.status_code == 200:
            log(f"Health Check PASSED: Agent responded with 200 OK.")
            return True
        else:
            # Consume content to get error message
            log(f"Health Check FAILED: Status {response.status_code} - {response.text}", "error")
            return False
            
    except Exception as e:
        log(f"Health Check FAILED with exception: {e}", "error")
        return False

def get_target_agent_id(display_name: str):
    """
    Finds the most recent Reasoning Engine with the given display_name.
    Returns (target_id, stale_ids).
    """
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    resources = reasoning_engines.ReasoningEngine.list()
    
    # Filter by display_name
    matches = [r for r in resources if r.display_name == display_name]
    
    if not matches:
        return None, []
    
    # Sort by create_time (latest first)
    matches.sort(key=lambda r: r.create_time, reverse=True)
    
    target = matches[0]
    target_id = target.resource_name.split("/")[-1]
    
    stale_ids = [r.resource_name.split("/")[-1] for r in matches[1:]]
    
    return target_id, stale_ids

def stream_logs(process):
    """Reads stdout from process and logs it."""
    for line in iter(process.stdout.readline, ''):
        if line:
            log(line.strip())
    process.stdout.close()

def deploy_agent(agent_dir: str, display_name: str, target_id: str = None):
    """
    Executes the adk deploy command with monitoring.
    """
    # Resolve paths
    abs_agent_dir = os.path.abspath(agent_dir)
    parent_dir = os.path.dirname(abs_agent_dir)
    agent_folder_name = os.path.basename(abs_agent_dir)
    adk_path = os.path.abspath(".venv/bin/adk")
    
    cmd = [
        adk_path, "deploy", "agent_engine",
        "--project", PROJECT_ID,
        "--region", LOCATION,
        "--staging_bucket", STAGING_BUCKET,
        "--trace_to_cloud",
        "--adk_app_object", "app",
        "--adk_app", "deployment.py"
    ]
    
    if target_id:
        cmd.extend(["--agent_engine_id", target_id])
    
    # Prepare environment variables
    env_vars = {}
    
    # 1. Read agent's existing .env if it exists
    agent_env_path = os.path.join(abs_agent_dir, ".env")
    if os.path.exists(agent_env_path):
        try:
             with open(agent_env_path, "r") as f:
                 for line in f:
                     line = line.strip()
                     if line and not line.startswith("#") and "=" in line:
                         k, v = line.split("=", 1)
                         env_vars[k] = v
        except Exception as e:
             log(f"Warning: Failed to read agent .env: {e}", "error")

    # 2. Read telemetry config (System Policy overrides)
    config_path = "conductor/telemetry_config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                telemetry_vars = config.get("environment_variables", {})
                log(f"Loading telemetry config from {config_path}: {telemetry_vars}")
                env_vars.update(telemetry_vars)
        except Exception as e:
            log(f"Warning: Failed to load telemetry config: {e}", "error")
    else:
        log(f"Warning: Telemetry config not found at {config_path}", "error")

    # Write temp .env file inside the agent dir (so it's relative to execution context)
    # We write it to absolute path
    temp_env_path = os.path.join(abs_agent_dir, ".env.deploy.tmp")
    try:
        with open(temp_env_path, "w") as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
        
        # We need to pass the env file path.
        # Since we are running from parent_dir, and temp file is in agent_folder_name
        cmd.extend(["--env_file", os.path.join(agent_folder_name, ".env.deploy.tmp")])
        
        cmd.append(agent_folder_name)

        log(f"Running command: {' '.join(cmd)} in cwd: {parent_dir}")
        
        # Use Popen to stream output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            env=os.environ.copy(),
            cwd=parent_dir # Set CWD to parent of agent folder
        )
        
        # Start log consumer thread
        log_thread = threading.Thread(target=stream_logs, args=(process,))
        log_thread.daemon = True
        log_thread.start()
        
        # Monitoring Loop
        start_time = time.time()
        last_poll = start_time
        
        while process.poll() is None:
            # Check Timeout
            if time.time() - start_time > DEPLOY_TIMEOUT:
                log("Deployment timed out! Terminating process.", "error")
                process.terminate()
                sys.exit(1)
            
            # Poll API every POLL_INTERVAL
            if target_id and (time.time() - last_poll > POLL_INTERVAL):
                status, _ = get_agent_status(target_id)
                log(f"Monitoring Agent {target_id}: State={status}")
                if "FAILED" in str(status):
                    log("Agent entered FAILED state. Deployment might have failed on backend.", "error")
                last_poll = time.time()
            
            time.sleep(1)
            
        log_thread.join(timeout=5)
        
        if process.returncode != 0:
            log(f"Deployment failed with return code: {process.returncode}", "error")
            sys.exit(process.returncode)
        
        log(f"Deployment successful.")
        
        # Post-deployment Verification
        if not target_id:
            # Creation mode: Find the new agent ID
            log("Finding new agent ID...")
            new_id, _ = get_target_agent_id(display_name)
            if new_id:
                target_id = new_id
                log(f"Found new agent ID: {target_id}")
            else:
                log("Could not find new agent ID.", "error")
        
        if target_id:
            verify_agent_health(target_id)
        
    finally:
        # Cleanup
        if os.path.exists(temp_env_path):
            os.remove(temp_env_path)

def cleanup_stale_agents(stale_ids: list[str]):
    """
    Deletes older Reasoning Engine instances using the API client.
    """
    if not stale_ids:
        log("No stale agents to clean up.")
        return

    from google.cloud import aiplatform_v1
    
    client = aiplatform_v1.ReasoningEngineServiceClient(
        client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    )

    log(f"Cleaning up {len(stale_ids)} stale agent(s): {stale_ids}")
    for engine_id in stale_ids:
        try:
            resource_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{engine_id}"
            request = aiplatform_v1.DeleteReasoningEngineRequest(name=resource_name, force=True)
            operation = client.delete_reasoning_engine(request=request)
            log(f"Delete operation started for agent: {engine_id} (force=True)")
            operation.result() # Wait for completion
            log(f"Deleted stale agent: {engine_id}")
        except Exception as e:
            log(f"Failed to delete agent {engine_id}: {e}", "error")

def delete_agent_by_id(agent_id: str):
    cleanup_stale_agents([agent_id])

def main():
    parser = argparse.ArgumentParser(description="Deploy an ADK agent to Vertex AI Reasoning Engine.")
    parser.add_argument("agent_dir", help="The directory of the agent to deploy.", nargs='?')
    parser.add_argument("--cleanup", action="store_true", help="Delete stale versions of the agent.")
    parser.add_argument("--delete-id", help="Delete a specific agent ID and exit.")
    
    args = parser.parse_args()
    
    if args.delete_id:
        delete_agent_by_id(args.delete_id)
        return

    if not args.agent_dir:
        log("Error: agent_dir is required unless --delete-id is specified.", "error")
        sys.exit(1)
    
    # Deriving display_name from directory name
    display_name = os.path.basename(args.agent_dir.strip("/"))
    
    log(f"Looking for existing deployments for: {display_name}")
    target_id, stale_ids = get_target_agent_id(display_name)
    
    if target_id:
        log(f"Found existing agent ID: {target_id}. Updating...")
    else:
        log("No existing agent found. Creating new...")
    
    deploy_agent(args.agent_dir, display_name, target_id)
    
    if args.cleanup:
        cleanup_stale_agents(stale_ids)

if __name__ == "__main__":
    main()
