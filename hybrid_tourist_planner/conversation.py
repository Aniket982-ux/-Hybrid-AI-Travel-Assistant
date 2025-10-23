_conversation_history = []
MAX_HISTORY_TURNS = 2
# Each turn consists of a user message and a system response.
def add_to_history(role: str, content: str):
    """Add message to sliding window."""
    _conversation_history.append({"role": role, "content": content})
    if len(_conversation_history) > MAX_HISTORY_TURNS * 2:
        _conversation_history.pop(0)
        _conversation_history.pop(0)

def get_conversation_history() -> str:
    """Retrieve conversation history."""
    if not _conversation_history:
        return "No previous conversation."
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in _conversation_history])

def clear_history():
    """Clear conversation history."""
    _conversation_history.clear()
