"""
Streamlit对话前端界面
"""
import streamlit as st
from agent import build_agent, chat_with_agent
from conversation_manager import ConversationManager
from langchain_core.messages import HumanMessage, AIMessage
import json

# 页面配置
st.set_page_config(
    page_title="AMAnalyze 对话助手",
    page_icon="🤖",
    layout="wide"
)

# 初始化对话管理器
@st.cache_resource
def get_conversation_manager():
    return ConversationManager()

@st.cache_resource
def get_agent():
    return build_agent()

conversation_manager = get_conversation_manager()
agent = get_agent()

# 侧边栏：会话管理
with st.sidebar:
    st.title("📚 会话管理")
    
    # 创建新会话
    if st.button("➕ 新建会话", use_container_width=True):
        new_conv_id = conversation_manager.create_conversation()
        st.session_state.conversation_id = new_conv_id
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # 会话列表
    st.subheader("历史会话")
    conversations = conversation_manager.list_conversations()
    
    if conversations:
        for conv in conversations:
            conv_id = conv["conversation_id"]
            turn_count = conv.get("turn_count", 0)
            updated_at = conv.get("updated_at", "")[:19] if conv.get("updated_at") else ""
            
            # 显示会话信息
            label = f"{conv_id[:20]}... ({turn_count}轮)"
            if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                st.session_state.conversation_id = conv_id
                # 加载历史消息
                conv_data = conversation_manager.load_conversation(conv_id)
                st.session_state.messages = []
                for turn in conv_data.get("turns", []):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": turn.get("user_input", "")
                    })
                    if turn.get("final_response"):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": turn.get("final_response", "")
                        })
                st.rerun()
    else:
        st.info("暂无历史会话")
    
    st.divider()
    
    # 当前会话信息
    if "conversation_id" in st.session_state:
        st.subheader("当前会话")
        st.text(f"ID: {st.session_state.conversation_id}")
        if st.button("🗑️ 删除当前会话", use_container_width=True):
            import os
            from pathlib import Path
            conv_file = Path("conversations") / f"{st.session_state.conversation_id}.json"
            if conv_file.exists():
                os.remove(conv_file)
            del st.session_state.conversation_id
            st.session_state.messages = []
            st.rerun()

# 主界面
st.title("🤖 AMAnalyze 对话助手")

# 初始化会话
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = conversation_manager.create_conversation()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到界面
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 获取历史消息用于上下文
    history_messages = []
    conv_data = conversation_manager.load_conversation(st.session_state.conversation_id)
    for turn in conv_data.get("turns", []):
        history_messages.append(HumanMessage(content=turn.get("user_input", "")))
        if turn.get("final_response"):
            history_messages.append(AIMessage(content=turn.get("final_response", "")))
    
    # 调用agent并显示响应
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = chat_with_agent(
                agent=agent,
                user_input=prompt,
                conversation_id=st.session_state.conversation_id,
                conversation_manager=conversation_manager,
                history_messages=history_messages
            )
            st.markdown(response)
    
    # 添加助手消息
    st.session_state.messages.append({"role": "assistant", "content": response})

# 底部：显示对话详情
if st.session_state.messages:
    st.divider()
    
    with st.expander("📋 查看对话详情（JSON格式）"):
        conv_data = conversation_manager.load_conversation(st.session_state.conversation_id)
        st.json(conv_data)
    
    # 显示最近一轮的工具调用
    conv_data = conversation_manager.load_conversation(st.session_state.conversation_id)
    if conv_data.get("turns"):
        last_turn = conv_data["turns"][-1]
        if last_turn.get("tool_calls"):
            with st.expander("🔧 工具调用详情"):
                st.json(last_turn["tool_calls"])

