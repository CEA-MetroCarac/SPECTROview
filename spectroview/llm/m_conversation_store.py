import os
import glob
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from spectroview.llm.m_conversation import MConversation

class ConversationSummary:
    """Lightweight representation of a conversation for lists."""
    def __init__(self, id: str, title: str, created_at: str, modified_at: str, message_count: int):
        self.id = id
        self.title = title
        self.created_at = created_at
        self.modified_at = modified_at
        self.message_count = message_count

class MConversationStore:
    """Manages all conversations in the history folder."""

    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self._index: Dict[str, ConversationSummary] = {}
        
        if self.folder_path and not os.path.exists(self.folder_path):
            try:
                os.makedirs(self.folder_path)
            except Exception as e:
                print(f"Error creating history folder {self.folder_path}: {e}")
                
        self.scan_folder()

    def scan_folder(self) -> None:
        """Refresh index from disk."""
        self._index.clear()
        
        if not self.folder_path or not os.path.exists(self.folder_path):
            return
            
        json_files = glob.glob(os.path.join(self.folder_path, "*.json"))
        
        for filepath in json_files:
            try:
                # We only need to read the header, but since JSON is small enough,
                # we'll read the whole thing for simplicity. For massive files,
                # we could stream or use ijson, but conversations are usually < 1MB.
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                conv_id = data.get("id")
                if not conv_id:
                    continue
                    
                summary = ConversationSummary(
                    id=conv_id,
                    title=data.get("title", "Unknown Conversation"),
                    created_at=data.get("created_at", ""),
                    modified_at=data.get("modified_at", ""),
                    message_count=len(data.get("messages", []))
                )
                self._index[conv_id] = summary
            except Exception as e:
                print(f"Error scanning {filepath}: {e}")

    def list_conversations(self) -> List[ConversationSummary]:
        """Returns summaries sorted by modified_at descending."""
        summaries = list(self._index.values())
        # Sort by modified_at desc, then created_at desc
        summaries.sort(key=lambda s: (s.modified_at, s.created_at), reverse=True)
        return summaries

    def load_conversation(self, conv_id: str) -> Optional[MConversation]:
        """Load full conversation from disk."""
        filepath = os.path.join(self.folder_path, f"{conv_id}.json")
        if not os.path.exists(filepath):
            return None
            
        conv = MConversation(filepath)
        return conv

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete conversation JSON file and remove from index."""
        filepath = os.path.join(self.folder_path, f"{conv_id}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            if conv_id in self._index:
                del self._index[conv_id]
            return True
        except Exception as e:
            print(f"Error deleting conversation {conv_id}: {e}")
            return False

    def create_conversation(self) -> MConversation:
        """Create a new blank conversation."""
        conv = MConversation()
        # Save it immediately to generate the file
        conv.save(self.folder_path)
        
        # Add to index
        self._index[conv.id] = ConversationSummary(
            id=conv.id,
            title=conv.title,
            created_at=conv.created_at,
            modified_at=conv.modified_at,
            message_count=0
        )
        return conv

    def import_legacy_md(self, filepath: str) -> Optional[MConversation]:
        """Parse an existing .md file into JSON format."""
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            conv = MConversation()
            
            # Extract timestamp from filename or content
            filename = os.path.basename(filepath)
            # Format: 26-07-12_12-13-44.md
            try:
                date_str = filename.replace(".md", "")
                dt = datetime.strptime(date_str, "%y-%m-%d_%H-%M-%S")
                iso_time = dt.isoformat()
                conv.created_at = iso_time
                conv.modified_at = iso_time
            except ValueError:
                pass # Use current time as fallback
                
            # Naive parsing of markdown chunks
            # Example: ### User\nadd scatter plot\n\n### AI\n{...}
            import re
            parts = re.split(r'### (User|AI)\n', content)
            
            if len(parts) > 1:
                # First part is header "# AI Chat Session - ..."
                header = parts[0]
                title_match = re.search(r'# AI Chat Session - (.*)', header)
                if title_match:
                    conv.title = f"Legacy: {title_match.group(1).strip()}"
                else:
                    conv.title = f"Legacy: {filename}"
                    
                # Pair the roles and contents
                # parts looks like: [header, 'User', 'content...', 'AI', 'content...', ...]
                for i in range(1, len(parts) - 1, 2):
                    role_str = parts[i]
                    msg_content = parts[i+1].strip()
                    
                    role = "user" if role_str == "User" else "assistant"
                    # Add without triggering auto-title
                    conv.messages.append({
                        "role": role,
                        "content": msg_content,
                        "timestamp": conv.created_at, # Use conversation time since we don't have message times
                        "reply_to_index": None
                    })
                    
            if len(conv.messages) > 0:
                conv.save(self.folder_path)
                
                self._index[conv.id] = ConversationSummary(
                    id=conv.id,
                    title=conv.title,
                    created_at=conv.created_at,
                    modified_at=conv.modified_at,
                    message_count=conv.message_count
                )
                return conv
                
        except Exception as e:
            print(f"Error importing legacy {filepath}: {e}")
            
        return None
