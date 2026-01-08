"""
对话历史管理器
用于保存和加载对话轨迹到JSON文件
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class ConversationManager:
    """管理对话历史的类"""
    
    def __init__(self, history_dir: str = "conversations"):
        """
        初始化对话管理器
        
        Args:
            history_dir: 存储对话历史的目录
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
    
    def _get_conversation_file(self, conversation_id: str) -> Path:
        """获取对话文件的路径"""
        return self.history_dir / f"{conversation_id}.json"
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        创建新的对话会话
        
        Args:
            conversation_id: 可选的会话ID，如果不提供则自动生成
            
        Returns:
            会话ID
        """
        if conversation_id is None:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        conversation_file = self._get_conversation_file(conversation_id)
        
        # 如果文件不存在，创建初始结构
        if not conversation_file.exists():
            initial_data = {
                "conversation_id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "turns": []
            }
            self._save_conversation(conversation_id, initial_data)
        
        return conversation_id
    
    def add_turn(
        self,
        conversation_id: str,
        user_input: str,
        llm_input: Dict[str, Any],
        llm_output: Dict[str, Any],
        tool_calls: List[Dict[str, Any]] = None,
        final_response: str = None
    ):
        """
        添加一轮对话记录
        
        Args:
            conversation_id: 会话ID
            user_input: 用户输入的内容
            llm_input: 实际发送给大模型的内容
            llm_output: 大模型返回的内容
            tool_calls: 工具调用列表
            final_response: 最终返回给用户的响应
        """
        conversation = self.load_conversation(conversation_id)
        
        turn = {
            "turn_number": len(conversation["turns"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "llm_input": self._serialize_messages(llm_input),
            "llm_output": self._serialize_messages(llm_output),
            "tool_calls": tool_calls or [],
            "final_response": final_response
        }
        
        conversation["turns"].append(turn)
        conversation["updated_at"] = datetime.now().isoformat()
        
        self._save_conversation(conversation_id, conversation)
    
    def _serialize_messages(self, messages: Any) -> List[Dict[str, Any]]:
        """
        序列化消息对象为字典
        
        Args:
            messages: LangChain消息对象或字典
            
        Returns:
            序列化后的消息列表
        """
        if isinstance(messages, dict):
            if "messages" in messages:
                messages = messages["messages"]
            else:
                return [messages]
        
        serialized = []
        for msg in messages:
            if hasattr(msg, 'content'):
                msg_dict = {
                    "type": type(msg).__name__,
                    "role": getattr(msg, 'type', 'unknown'),
                    "content": msg.content if hasattr(msg.content, '__str__') else str(msg.content)
                }
                # 如果有工具调用信息，也保存
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "name": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "id": tc.get("id", "")
                        }
                        for tc in msg.tool_calls
                    ]
                serialized.append(msg_dict)
            elif isinstance(msg, dict):
                serialized.append(msg)
            else:
                serialized.append({"type": str(type(msg)), "content": str(msg)})
        
        return serialized
    
    def load_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        加载对话历史
        
        Args:
            conversation_id: 会话ID
            
        Returns:
            对话历史字典
        """
        conversation_file = self._get_conversation_file(conversation_id)
        
        if not conversation_file.exists():
            return {
                "conversation_id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "turns": []
            }
        
        with open(conversation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_conversation(self, conversation_id: str, conversation: Dict[str, Any]):
        """保存对话历史到文件"""
        conversation_file = self._get_conversation_file(conversation_id)
        
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        列出所有对话会话
        
        Returns:
            会话列表，每个会话包含ID和基本信息
        """
        conversations = []
        
        for json_file in self.history_dir.glob("*.json"):
            conversation_id = json_file.stem
            try:
                conv = self.load_conversation(conversation_id)
                conversations.append({
                    "conversation_id": conversation_id,
                    "created_at": conv.get("created_at", ""),
                    "updated_at": conv.get("updated_at", ""),
                    "turn_count": len(conv.get("turns", []))
                })
            except Exception as e:
                print(f"加载对话 {conversation_id} 时出错: {e}")
        
        # 按更新时间倒序排列
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations
    
    def get_messages_for_llm(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        从对话历史中提取消息列表，用于LLM的上下文
        
        Args:
            conversation_id: 会话ID
            
        Returns:
            消息列表
        """
        conversation = self.load_conversation(conversation_id)
        messages = []
        
        for turn in conversation.get("turns", []):
            # 添加用户消息
            messages.append({
                "role": "user",
                "content": turn.get("user_input", "")
            })
            
            # 添加助手消息（如果有最终响应）
            if turn.get("final_response"):
                messages.append({
                    "role": "assistant",
                    "content": turn.get("final_response", "")
                })
        
        return messages

