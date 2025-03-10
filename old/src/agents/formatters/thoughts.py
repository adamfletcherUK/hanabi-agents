from typing import List


def format_thoughts(thoughts: List[str]) -> str:
    """Format current thoughts for display."""
    if not thoughts:
        return "No thoughts yet"

    # Limit the length of each thought to avoid extremely long prompts
    # Set to None to disable truncation
    max_thought_length = 500
    formatted_thoughts = []

    for i, thought in enumerate(thoughts):
        if max_thought_length and len(thought) > max_thought_length:
            short_thought = thought[:max_thought_length] + \
                "... (truncated)"
        else:
            short_thought = thought

        formatted_thoughts.append(f"Thought {i+1}: {short_thought}")

    return "\n".join(formatted_thoughts)
