from agents.base import BaseAgent

class ClaudeAgent(BaseAgent):
    allowed_tools: list[str]
    permissions: str

    def __init__(self, allowed_tools, permissions):
        self.allowed_tools = allowed_tools
        self.permissions = permissions

    def run(cmd):
        pass #TODO

    def close():
        pass #TODO
