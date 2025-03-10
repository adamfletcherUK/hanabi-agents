from typing import List


def format_discussion(discussion_history: List[str]) -> str:
    """Format discussion history for display."""
    if not discussion_history:
        return "No discussion yet"

    # Limit the length of each message to avoid extremely long prompts
    # Set to None to disable truncation
    max_message_length = 500
    formatted_messages = []

    for i, msg in enumerate(discussion_history):
        if max_message_length and len(msg) > max_message_length:
            short_msg = msg[:max_message_length] + "... (truncated)"
        else:
            short_msg = msg

        formatted_messages.append(f"Message {i+1}: {short_msg}")

    return "\n".join(formatted_messages)
