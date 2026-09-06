"""
Conversation history management
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from core.models import ConversationTurn


class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation manager
        
        Args:
            max_history: Maximum number of turns to keep
        """
        self.history: List[ConversationTurn] = []
        self.max_history = max_history
    
    def add_turn(
        self,
        user_query: str,
        clarified_query: str = None,
        retrieved_context: str = None,
        assistant_response: str = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Add a conversation turn
        
        Args:
            user_query: User's original query
            clarified_query: Clarified version of query
            retrieved_context: Retrieved context
            assistant_response: Assistant's response
            metadata: Additional metadata
        """
        turn = ConversationTurn(
            user_query=user_query,
            clarified_query=clarified_query,
            retrieved_context=retrieved_context,
            assistant_response=assistant_response,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.history.append(turn)
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_recent_context(self, num_turns: int = 3) -> str:
        """
        Get recent conversation context as a formatted string
        
        Args:
            num_turns: Number of recent turns to include
        
        Returns:
            Formatted conversation context
        """
        if not self.history:
            return ""
        
        recent_turns = self.history[-num_turns:]
        context_parts = []
        
        for i, turn in enumerate(recent_turns, 1):
            turn_text = f"Turn {i}:\n"
            turn_text += f"User: {turn.user_query}\n"
            if turn.clarified_query and turn.clarified_query != turn.user_query:
                turn_text += f"Clarified: {turn.clarified_query}\n"
            if turn.assistant_response:
                # Include full assistant response without truncation
                turn_text += f"Assistant: {turn.assistant_response}\n"
            # Intentionally exclude retrieved RAG context from history to save tokens
            context_parts.append(turn_text)
        
        return "\n".join(context_parts)
    
    def get_last_entities(self) -> List[str]:
        """
        Extract entities mentioned in recent conversation
        
        Returns:
            List of unique entities
        """
        entities = []
        for turn in self.history[-3:]:
            if turn.metadata and 'entities' in turn.metadata:
                entities.extend(turn.metadata['entities'])
        return list(set(entities))  # Remove duplicates
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def get_summary(self) -> str:
        """
        Get a summary of the conversation
        
        Returns:
            Formatted conversation summary
        """
        if not self.history:
            return "No conversation history"
        
        summary = f"Conversation History ({len(self.history)} turns):\n"
        summary += "=" * 60 + "\n"
        
        for i, turn in enumerate(self.history, 1):
            summary += f"\nTurn {i} ({turn.timestamp}):\n"
            summary += f"  User: {turn.user_query}\n"
            if turn.clarified_query and turn.clarified_query != turn.user_query:
                summary += f"  Clarified: {turn.clarified_query}\n"
            if turn.assistant_response:
                response_preview = turn.assistant_response[:100]
                if len(turn.assistant_response) > 100:
                    response_preview += "..."
                summary += f"  Assistant: {response_preview}\n"
        
        return summary

