# AMAnalyze 对话助手

一个基于 LangChain 和 Streamlit 的多轮对话助手，支持工具调用和对话历史记录。

## 功能特性

- ✅ 多轮对话支持
- ✅ 对话历史记录（保存为JSON文件）
- ✅ 工具调用跟踪
- ✅ Streamlit Web界面
- ✅ 会话管理（创建、切换、删除）

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置环境变量：
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

Windows PowerShell:
```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

## 使用方法

### 启动Web界面

```bash
streamlit run app.py
```

然后在浏览器中打开显示的URL（通常是 http://localhost:8501）

### 命令行使用

```bash
python agent.py
```

## 项目结构

```
AMAnalyze/
├── agent.py                 # Agent核心逻辑
├── llm.py                   # LLM配置
├── conversation_manager.py  # 对话历史管理器
├── app.py                   # Streamlit前端界面
├── requirements.txt         # 依赖包列表
├── conversations/           # 对话历史存储目录（自动创建）
└── skills/                  # 技能目录
    ├── caculator/
    │   ├── SKILL.md
    │   └── skill.py
    └── summarize-skill/
        ├── SKILL.MD
        └── skill.py
```

## 对话历史格式

对话历史保存在 `conversations/` 目录下的JSON文件中，格式如下：

```json
{
  "conversation_id": "20240101_120000_123456",
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:05:00",
  "turns": [
    {
      "turn_number": 1,
      "timestamp": "2024-01-01T12:00:00",
      "user_input": "帮我计算 12 * (8 + 5)",
      "llm_input": [...],
      "llm_output": [...],
      "tool_calls": [
        {
          "name": "calculator",
          "args": {"expression": "12 * (8 + 5)"},
          "id": "..."
        }
      ],
      "final_response": "计算结果为 156"
    }
  ]
}
```

## 开发

### 添加新技能

1. 在 `skills/` 目录下创建新文件夹
2. 创建 `SKILL.md` 文件，定义技能元数据
3. 创建 `skill.py` 文件，实现 `run()` 函数

示例：
```python
# skills/my_skill/skill.py
def run(param: str) -> str:
    """技能描述"""
    # 实现逻辑
    return result
```

## 许可证

MIT License

