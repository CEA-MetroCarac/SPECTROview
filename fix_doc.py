with open("docs/developer/ai_agent.md", "r") as f:
    text = f.read()

old_text = "The rolling context sent to the LLM is capped at the last **6 pairs** (12 messages) to prevent context overflow."
new_text = "The rolling context sent to the LLM is configurable via `max_context_messages` (defaulting to no cap, meaning all messages in the conversation are sent)."
text = text.replace(old_text, new_text)

with open("docs/developer/ai_agent.md", "w") as f:
    f.write(text)
