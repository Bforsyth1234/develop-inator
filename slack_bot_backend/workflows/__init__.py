"""Workflow orchestration boundary for QUESTION/ACTION/CONFIGURE flows."""

__all__ = ["ActionWorkflow", "ConfigureWorkflow", "IntentWorkflow", "QuestionWorkflow"]


def __getattr__(name: str):
    if name == "ActionWorkflow":
        from .action import ActionWorkflow

        return ActionWorkflow
    if name == "ConfigureWorkflow":
        from .configure import ConfigureWorkflow

        return ConfigureWorkflow
    if name == "IntentWorkflow":
        from .intent import IntentWorkflow

        return IntentWorkflow
    if name == "QuestionWorkflow":
        from .question import QuestionWorkflow

        return QuestionWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
