from skillkit import SkillManager
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import importlib.util
import copy

from llm import get_llm
from conversation_manager import ConversationManager

def load_skills(skill_dir: str):
    """加载技能元数据和实际技能函数"""
    sm = SkillManager(skill_dir)
    sm.discover()  # discover() 返回 None，只是填充内部注册表
    skill_metadatas = sm.list_skills()  # 使用 list_skills() 获取技能元数据列表
    
    # 为每个技能元数据加载实际的技能函数
    skills = []
    for metadata in skill_metadatas:
        # 获取技能文件夹路径（SKILL.md 的父目录）
        skill_folder = metadata.skill_path.parent
        skill_py_path = skill_folder / "skill.py"
        
        if skill_py_path.exists():
            # 动态导入 skill.py 模块
            spec = importlib.util.spec_from_file_location(f"skill_{metadata.name}", skill_py_path)
            skill_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(skill_module)
            
            # 获取 run 函数
            if hasattr(skill_module, 'run'):
                skills.append({
                    'name': metadata.name,
                    'description': metadata.description,
                    'run': skill_module.run
                })
    
    return skills

def skill_to_tool(skill):
    """将 skill 转换为工具函数"""
    def tool_func(expression: str) -> str:
        """执行技能计算
        
        Args:
            expression: 要计算的数学表达式
            
        Returns:
            计算结果字符串
        """
        return skill['run'](expression=expression)
    
    # 设置函数名称和文档字符串
    tool_func.__name__ = skill['name']
    tool_func.__doc__ = skill['description'] or f"执行 {skill['name']} 技能"
    
    # 使用 @tool 装饰器创建工具
    return tool(tool_func)

def build_agent():
    """构建agent实例"""
    llm = get_llm()

    skills = load_skills("./skills")
    tools = [skill_to_tool(s) for s in skills]

    agent = create_agent(
        model=llm,
        tools=tools,
        debug=True
    )
    return agent


def chat_with_agent(
    agent,
    user_input: str,
    conversation_id: str,
    conversation_manager: ConversationManager,
    history_messages: list = None
):
    """
    与agent进行对话并记录历史
    
    Args:
        agent: agent实例
        user_input: 用户输入
        conversation_id: 会话ID
        conversation_manager: 对话管理器
        history_messages: 历史消息列表
        
    Returns:
        最终响应字符串
    """
    # 准备输入消息
    if history_messages:
        messages = copy.deepcopy(history_messages)
        messages.append(HumanMessage(content=user_input))
    else:
        messages = [HumanMessage(content=user_input)]
    
    llm_input = {"messages": messages}
    
    # 调用agent
    try:
        result = agent.invoke(llm_input)
        
        # 提取工具调用信息
        tool_calls = []
        tool_results = {}  # 存储工具执行结果，key为tool_call_id
        
        for msg in result.get("messages", []):
            msg_type = getattr(msg, 'type', '') if hasattr(msg, 'type') else ''
            
            # 检查是否有tool_calls属性（AIMessage中的工具调用请求）
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    # 处理不同的tool_call格式
                    if isinstance(tc, dict):
                        tool_call_info = {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", "")
                        }
                    else:
                        tool_call_info = {
                            "name": getattr(tc, 'name', ''),
                            "args": getattr(tc, 'args', {}),
                            "id": getattr(tc, 'id', '')
                        }
                    tool_calls.append(tool_call_info)
            
            # 检查是否是ToolMessage（工具执行结果）
            if msg_type == 'tool':
                tool_call_id = getattr(msg, 'tool_call_id', '')
                tool_name = getattr(msg, 'name', '')
                tool_content = getattr(msg, 'content', '')
                
                # 将工具执行结果与对应的工具调用关联
                tool_results[tool_call_id] = {
                    "name": tool_name,
                    "output": tool_content
                }
        
        # 合并工具调用和工具执行结果
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            if tc_id in tool_results:
                tc["output"] = tool_results[tc_id].get("output", "")
        
        # 提取最终响应
        final_response = None
        messages_result = result.get("messages", [])
        if messages_result:
            last_message = messages_result[-1]
            if hasattr(last_message, 'content'):
                final_response = str(last_message.content)
            else:
                final_response = str(result)
        else:
            final_response = str(result)
        
        # 记录对话历史
        conversation_manager.add_turn(
            conversation_id=conversation_id,
            user_input=user_input,
            llm_input=llm_input,
            llm_output=result,
            tool_calls=tool_calls if tool_calls else None,
            final_response=final_response
        )
        
        return final_response
        
    except Exception as e:
        error_msg = f"错误: {str(e)}"
        # 记录错误
        conversation_manager.add_turn(
            conversation_id=conversation_id,
            user_input=user_input,
            llm_input=llm_input,
            llm_output={"error": str(e)},
            tool_calls=None,
            final_response=error_msg
        )
        return error_msg

if __name__ == "__main__":
    agent = build_agent()

    query = "帮我计算 12 * (8 + 5)"
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # 提取最后一条 AI 消息的内容
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            print(last_message.content)
        else:
            print(result)
    else:
        print(result)
