"""Tests for conversation management."""

from src.core.conversation import Conversation, Message, Role


def test_conversation_add_user():
    conv = Conversation()
    msg = conv.add_user("Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"
    assert len(conv) == 1


def test_conversation_add_assistant():
    conv = Conversation()
    conv.add_user("Hi")
    msg = conv.add_assistant("Hello!")
    assert msg.role == Role.ASSISTANT
    assert len(conv) == 2


def test_conversation_clear():
    conv = Conversation()
    conv.add_user("Test")
    conv.clear()
    assert len(conv) == 0


def test_conversation_trim():
    conv = Conversation(max_history=3)
    for i in range(10):
        conv.add_user(f"msg {i}")
    assert len(conv) == 3
    assert conv.messages[0].content == "msg 7"


def test_conversation_export():
    conv = Conversation(system_prompt="Test")
    conv.add_user("Hi")
    exported = conv.export_json()
    assert "system_prompt" in exported
    assert "messages" in exported


def test_conversation_get_messages():
    conv = Conversation(system_prompt="System")
    conv.add_user("User msg")
    msgs = conv.get_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
