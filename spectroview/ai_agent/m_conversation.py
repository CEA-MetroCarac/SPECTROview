import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

class MConversation:
    """Represents a single conversation stored as JSON."""

    def __init__(self, filepath: Optional[str] = None):
        self.id: str = str(uuid.uuid4())
        self.title: str = "New Conversation"
        self.created_at: str = datetime.now().isoformat()
        self.modified_at: str = self.created_at
        self.messages: List[Dict[str, Any]] = []
        self._filepath: Optional[str] = filepath
        
        if filepath and os.path.exists(filepath):
            self.load(filepath)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    def add_message(self, role: str, content: str, reply_to_index: Optional[int] = None, is_hidden: bool = False, tool_calls: Optional[List[Dict[str, Any]]] = None, tool_call_id: Optional[str] = None) -> None:
        """Add a new message to the conversation."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "reply_to_index": reply_to_index,
            "is_hidden": is_hidden
        }
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
            
        self.messages.append(msg)
        self.modified_at = datetime.now().isoformat()
        
        # Auto-title on first user message
        if role == "user" and self.title == "New Conversation":
            # Extract first line, up to 60 chars
            first_line = content.split('\n')[0].strip()
            if len(first_line) > 60:
                first_line = first_line[:57] + "..."
            if first_line:
                self.title = first_line

    def rename(self, new_title: str) -> None:
        self.title = new_title
        self.modified_at = datetime.now().isoformat()

    def duplicate(self) -> 'MConversation':
        """Create a deep copy with a new ID and timestamps."""
        new_conv = MConversation()
        new_conv.title = f"{self.title} (Copy)"
        # Deep copy messages
        import copy
        new_conv.messages = copy.deepcopy(self.messages)
        # Update timestamps to now, but keep message timestamps as is
        new_conv.modified_at = datetime.now().isoformat()
        return new_conv

    def save(self, folder: Optional[str] = None) -> None:
        """Write the conversation to a JSON file."""
        if not self.messages:
            return # Don't save empty conversations
            
        import re
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', self.title)
        safe_title = safe_title[:40].strip('_')
        if not safe_title:
            safe_title = "Untitled"
        new_filename = f"{safe_title}_{self.id}.json"
            
        # If folder is provided, update filepath
        if folder:
            new_filepath = os.path.join(folder, new_filename)
        elif self._filepath:
            folder_path = os.path.dirname(self._filepath)
            new_filepath = os.path.join(folder_path, new_filename)
        else:
            return # Can't save without a path

        # Remove old file if filepath changed
        if self._filepath and self._filepath != new_filepath and os.path.exists(self._filepath):
            try:
                os.remove(self._filepath)
            except Exception:
                pass
                
        self._filepath = new_filepath

        # Create dir if not exists
        folder_path = os.path.dirname(self._filepath)
        if folder_path and not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
            except Exception:
                pass

        data = {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "messages": self.messages
        }
        
        try:
            with open(self._filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load(self, filepath: str) -> None:
        """Deserialize from JSON."""
        self._filepath = filepath
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.id = data.get("id", self.id)
            self.title = data.get("title", self.title)
            self.created_at = data.get("created_at", self.created_at)
            self.modified_at = data.get("modified_at", self.modified_at)
            self.messages = data.get("messages", [])
        except Exception as e:
            print(f"Error loading conversation {filepath}: {e}")

    def to_llm_messages(self, max_context_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Convert to [{"role": ..., "content": ...}] format for the LLM API, including reply context."""
        llm_messages = []
        
        msgs_to_process = self.messages
        if max_context_messages is not None and max_context_messages > 0:
            # We want the last N messages
            msgs_to_process = self.messages[-max_context_messages:]
            
        for i, msg in enumerate(msgs_to_process):
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Skip error messages from history context
            if role == "error":
                continue
                
            reply_idx = msg.get("reply_to_index")
            if role == "user" and reply_idx is not None and 0 <= reply_idx < len(self.messages):
                # Prepend reply context
                replied_msg = self.messages[reply_idx]
                replied_content = replied_msg.get("content", "")
                
                # Truncate replied content if too long
                if len(replied_content) > 300:
                    replied_content = replied_content[:297] + "..."
                    
                context_prefix = f"[Replying to AI message: \"{replied_content}\"]\n\n"
                content = context_prefix + content
                
            msg_dict = {"role": role, "content": content}
            if "tool_calls" in msg:
                msg_dict["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                msg_dict["tool_call_id"] = msg["tool_call_id"]
                
            llm_messages.append(msg_dict)
            
        return llm_messages
