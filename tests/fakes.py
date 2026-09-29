"""Test doubles shared across test files."""

from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedLLM(GenericFakeChatModel):
    """Returns pre-written messages in order. bind_tools is a no-op because the script decides."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])
