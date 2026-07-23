import os
import glob
import json
from typing import List, Dict, Optional

from spectroview.ai_agent.m_conversation import MConversation

class ConversationSummary:
    """Lightweight representation of a conversation for lists."""
    def __init__(self, id: str, title: str, created_at: str, modified_at: str, message_count: int, filepath: str = ""):
        self.id = id
        self.title = title
        self.created_at = created_at
        self.modified_at = modified_at
        self.message_count = message_count
        self.filepath = filepath

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
                    message_count=len(data.get("messages", [])),
                    filepath=filepath
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

    def get_summary(self, conv_id: str) -> Optional[ConversationSummary]:
        """Return the cached summary for *conv_id*, or None if unknown."""
        return self._index.get(conv_id)

    def load_conversation(self, conv_id: str) -> Optional[MConversation]:
        """Load full conversation from disk."""
        summary = self._index.get(conv_id)
        if summary and summary.filepath and os.path.exists(summary.filepath):
            return MConversation(summary.filepath)
            
        filepath = os.path.join(self.folder_path, f"{conv_id}.json")
        if not os.path.exists(filepath):
            matches = glob.glob(os.path.join(self.folder_path, f"*_{conv_id}.json"))
            if matches:
                filepath = matches[0]
            else:
                return None
            
        conv = MConversation(filepath)
        return conv

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete conversation JSON file and remove from index."""
        summary = self._index.get(conv_id)
        if summary and summary.filepath and os.path.exists(summary.filepath):
            filepath = summary.filepath
        else:
            filepath = os.path.join(self.folder_path, f"{conv_id}.json")
            if not os.path.exists(filepath):
                matches = glob.glob(os.path.join(self.folder_path, f"*_{conv_id}.json"))
                if matches:
                    filepath = matches[0]
                    
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
            message_count=0,
            filepath=conv._filepath or ""
        )
        return conv

