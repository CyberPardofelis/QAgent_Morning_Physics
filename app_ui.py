# app_ui.py
import gradio as gr
import markdown
import time
import threading
from agent.controller import run_agent

# ------------------------------
# 模拟 agent 响应函数（替换为你的 run_agent）
# ------------------------------
stop_flag = False

def agent_response(message, history, model_mode):
    global stop_flag
    stop_flag = False
    response_text = run_agent(message, mode=model_mode)
    history.append({
        "user": message,
        "agent": response_text
    })
    return render_chat(history), history, ""

def stop_agent():
    global stop_flag
    stop_flag = True
    return None

def render_chat(history):
    html_messages = ""
    for msg in history:
        user_html = markdown.markdown(msg["user"])
        agent_html = markdown.markdown(msg["agent"])
        html_messages += f"""
        <div style="text-align:right; margin:5px;">
            <div style="display:inline-block; background-color:#DCF8C6; padding:8px 12px; border-radius:12px; max-width:70%;">
                {user_html}
            </div>
        </div>
        <div style="text-align:left; margin:5px;">
            <div style="display:inline-block; background-color:#F1F0F0; padding:8px 12px; border-radius:12px; max-width:70%;">
                {agent_html}
            </div>
        </div>
        """
    # 自动滚动到底部
    html_messages += "<script>var chatWindow=document.getElementById('chat_window'); chatWindow.scrollTop = chatWindow.scrollHeight;</script>"
    # 触发 MathJax 渲染
    html_messages += "<script>if(window.MathJax){MathJax.typesetPromise();}</script>"
    return html_messages

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks(title="Quantum Research Agent") as demo:
    state_history = gr.State([])

    gr.HTML("""
    <style>
    body { margin:0; font-family: Arial; }
    /* 上半部分：对话记录 */
     #chat_window {
        height: calc(100vh - 80px);
        overflow-y: auto;
        padding:5px;
        border:1px solid #ccc;
        border-radius:6px;
        background-color:#fff;
    }
    /* 下半部分：输入条 + 控件 */
    #input_bar {
        display: flex;
        align-items:flex-start;
        padding:4px;
        gap:4px;
        height:50px;
        box-sizing:border-box;
    }
    #input_box {
        flex:3; /* 输入框占3/4 */
        max-height:120px;
        overflow-y:auto;
    }
    #side_controls {
        flex:1; /* 左侧控件占1/4 */
        display:flex;
        gap:4px;
        justify-content:flex-start;
        align-items:center;
        min-width:120px; /* 控件总宽度固定 */
    }
    </style>
    """)

    # 聊天显示区
    chat_display = gr.HTML("<div id='chat_window'></div>", elem_id="chat_window")

    # 下半部分：输入 + 控件
    with gr.Row(elem_id="input_bar"):
        with gr.Column(elem_id="side_controls"):
            model_selector = gr.Dropdown(
                ["openai", "deepseek", "compare"],
                value="deepseek",
                label="模型",
                interactive=True
            )
            stop_btn = gr.Button("停止")
        txt = gr.Textbox(
            placeholder="输入问题然后按回车发送",
            lines=1,
            max_lines=6,
            label="",
            elem_id="input_box"
        )

    # 回车发送
    txt.submit(
        fn=agent_response,
        inputs=[txt, state_history, model_selector],
        outputs=[chat_display, state_history, txt],
        queue=True
    )

    stop_btn.click(fn=stop_agent, inputs=[], outputs=[])

demo.launch()