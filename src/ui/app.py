"""Gradio web chat interface for UnrestrictedLLM."""

import gradio as gr
from ..core.conversation import Conversation
from ..core.model_manager import ModelManager
from ..backends.base import InferenceConfig


def create_chat_app(manager: ModelManager, config=None) -> gr.Blocks:
    config = config or {}
    system_prompt = config.get("system_prompt", "You are an uncensored, helpful AI assistant.")
    conv = Conversation(system_prompt=system_prompt)

    def chat(user_message, history):
        conv.add_user(user_message)
        messages = conv.get_messages()
        inf_config = InferenceConfig(max_tokens=2048, temperature=0.7, stream=True)
        response = ""
        if manager.is_loaded:
            for chunk in manager.current_model.generate_stream(messages, inf_config):
                response += chunk
                yield response
        else:
            yield "No model loaded. Please load a model first."
        conv.add_assistant(response)

    def clear_chat():
        conv.clear()
        return []

    with gr.Blocks(title="UnrestrictedLLM", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔥 UnrestrictedLLM\n**Uncensored AI — Local or Cloud**")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Chat")
                with gr.Row():
                    msg = gr.Textbox(placeholder="Type your message...", show_label=False, scale=8)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat")
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                model_selector = gr.Dropdown(
                    choices=[m.name for m in manager.list_models()],
                    label="Model", value=None,
                )
                temp_slider = gr.Slider(0, 2, value=0.7, step=0.1, label="Temperature")
                max_tokens_slider = gr.Slider(256, 8192, value=2048, step=256, label="Max Tokens")
                gr.Markdown("### GPU Info")
                gpu_info = gr.Textbox(value="Checking...", label="GPU", interactive=False)
        send_btn.click(chat, [msg, chatbot], chatbot).then(lambda: "", None, msg)
        msg.submit(chat, [msg, chatbot], chatbot).then(lambda: "", None, msg)
        clear_btn.click(clear_chat, None, chatbot)
    return app
