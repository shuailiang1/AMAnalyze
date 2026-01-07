from skillkit import SkillManager
from langchain.agents import create_agent
from langchain_core.tools import tool
import importlib.util

from llm import get_llm

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
    llm = get_llm()

    skills = load_skills("./skills")
    tools = [skill_to_tool(s) for s in skills]

    agent = create_agent(
        model=llm,
        tools=tools,
        debug=True
    )
    return agent

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
