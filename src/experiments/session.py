from agents.base import Agent
from agents.claude import ClaudeAgent
from agents.codex import CodexAgent


class Session:
    agent: Agent

    def __init__(self, agent_name, permissions, allowed_tools):
        if agent_name == "claude":
            self.agent = ClaudeAgent(permissions=permissions, allowed_tools=allowed_tools)
        elif agent_name == "codex":
            self.agent = CodexAgent(permissions=permissions, allowed_tools=allowed_tools)

    def run_experiment():
        pass #TODO