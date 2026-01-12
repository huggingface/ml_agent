"""Reliability checks for job submissions and other operations"""

from agent.utils.terminal_display import Colors


def check_training_script_save_pattern(script: str) -> str | None:
    """Check if a training script properly saves models."""
    has_from_pretrained = "from_pretrained" in script
    has_push_to_hub = "push_to_hub" in script

    if has_from_pretrained and not has_push_to_hub:
        return f"\n{Colors.RED}WARNING: We've detected that no model will be saved at the end of this training script. Please ensure this is what you want.{Colors.RESET}"
    elif has_from_pretrained and has_push_to_hub:
        return f"\n{Colors.GREEN}We've detected that a model will be pushed to hub at the end of this training.{Colors.RESET}"

    return None
