import anyio
import os
from claude_agent_sdk import query, ClaudeAgentOptions

# Load key from environment or .env
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

async def run_agent():
    # Options control what the agent can touch on your Windows system
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Bash"], # Allows reading files and running cmd
        permission_mode='acceptEdits'    # Claude asks before making changes
    )

    prompt = "Create a script 'test.py' that calculates the first 10 Fibonacci numbers."
    
    async for message in query(prompt=prompt, options=options):
        if hasattr(message, 'content'):
            for block in message.content:
                if hasattr(block, 'text'):
                    print(f"Claude: {block.text}")

if __name__ == "__main__":
    anyio.run(run_agent)
