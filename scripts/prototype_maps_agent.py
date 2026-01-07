import os
import sys
from app.agent.maps_agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

def main():
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
    location = "us-central1"
    
    print(f"Initializing Agent for project {project_id}...")
    # root_agent is already instantiated in maps_agent.py
    agent = root_agent

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, session_service=session_service)
    
    print("Restaurant Agent (Maps Grounding) Prototype")
    print("Type 'exit' to quit.")
    
    session_id = "cli-session-1"
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            print("Agent: Thinking...")
            result = runner.run(user_input, session_id=session_id)
            
            # Try to extract text if it's an object
            if hasattr(result, 'text'):
                print(f"Agent: {result.text}")
            elif hasattr(result, 'content'):
                 print(f"Agent: {result.content}")
            else:
                print(f"Agent: {result}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
