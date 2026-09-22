
class BaseAgent:
    allowed_tools: list[str]
    permissions: str

    def __init__(self):
        pass

    def run(prompt):
        raise NotImplementedError

    def end():
        raise NotImplementedError