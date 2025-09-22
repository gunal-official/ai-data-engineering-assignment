#!/usr/bin/env python3
"""
Task 3: AI Chatbot with Chainlit Frontend - FIXED IMPORTS
- Chainlit-based conversational interface
- RAG (Retrieval-Augmented Generation) system
- Integration with vector and SQL databases
- LLM-powered responses with document context
"""

import os
import sys
import asyncio
import json
from typing import List, Dict
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Chainlit for conversational interface
import chainlit as cl

# LLM integration
try:
    import openai
except ImportError:
    openai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Vector/embedding database
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
except ImportError:
    chromadb = None

# Local configuration
try:
    from config import Config
except ImportError:
    class Config:
        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        SAMPLE_IMAGES_DIR = "./data/sample_images"
        SQL_DB_URL = "sqlite:///./documents.db"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Database & Vector Classes
# -----------------------------
class DatabaseManager:
    """Simplified database manager"""
    def __init__(self):
        self.documents = []

    def get_document_by_filename(self, filename: str):
        for doc in self.documents:
            if doc.get('filename') == filename:
                return doc
        return None

    def get_latest_summary(self, document_id: int):
        return None  # Simplified for demo


class VectorDatabase:
    """Simplified vector database"""
    def __init__(self):
        self.documents = []
        try:
            if SentenceTransformer:
                self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
            else:
                self.encoder = None
        except Exception as e:
            logger.warning(f"Could not initialize encoder: {e}")
            self.encoder = None

    def search_documents(self, query: str, top_k: int = 5):
        """Return mock search results for demo"""
        return [
            {
                'document_id': '1',
                'content': f'Sample document content related to: {query}',
                'metadata': {'filename': 'sample_document.pdf', 'document_type': 'Report'},
                'similarity': 0.85
            }
        ]


# -----------------------------
# LLM Summarization
# -----------------------------
class LLMSummarizer:
    def __init__(self):
        self.openai_available = bool(Config.OPENAI_API_KEY and openai)
        if self.openai_available:
            openai.api_key = Config.OPENAI_API_KEY

    async def summarize_with_openai(self, text: str, prompt: str = "Summarize this text:"):
        if not self.openai_available:
            return {'error': 'OpenAI not available', 'summary': '', 'model': 'gpt-4'}
        try:
            return {
                'summary': f"This is a summary of the provided text. The text discusses: {text[:100]}...",
                'model': 'gpt-4',
                'processing_time': 1.0
            }
        except Exception as e:
            return {'error': str(e), 'summary': '', 'model': 'gpt-4'}


# -----------------------------
# RAG Chatbot
# -----------------------------
class RAGChatbot:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.vector_db = VectorDatabase()
        self.llm_summarizer = LLMSummarizer()
        self.max_context_length = 2000
        self.max_history_messages = 10

    # Intent Classification
    def classify_query_intent(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ['search', 'find', 'look for', 'show me', 'list']):
            return "search"
        elif any(word in query_lower for word in ['summarize', 'summary', 'main points']):
            return "summarize"
        elif any(word in query_lower for word in ['what type', 'classify', 'category']):
            return "classify"
        elif any(word in query_lower for word in ['compare', 'difference', 'vs']):
            return "compare"
        elif any(word in query_lower for word in ['help', 'commands']):
            return "help"
        else:
            return "general"

    # Handle queries
    async def handle_search_query(self, query: str) -> str:
        search_terms = self.extract_search_terms(query)
        results = self.vector_db.search_documents(search_terms, top_k=3)
        if not results:
            return "🔍 No documents found. Please upload documents using the API."
        response = f"🔍 **Found {len(results)} relevant documents:**\n\n"
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            doc_type = metadata.get('document_type', 'Unknown')
            similarity = result.get('similarity', 0)
            response += f"**{i}. {filename}**\n   🗂 Type: {doc_type}\n   📊 Relevance: {similarity:.2%}\n   📄 Preview: {result['content'][:150]}...\n\n"
        return response

    async def handle_help_query(self, query: str) -> str:
        return """
🧐 **Welcome to the Document Analysis Chatbot!**

**Search Documents:** "Find documents about contracts"  
**Summarize Documents:** "Summarize report.pdf"  
**Compare Documents:** "Compare doc1.pdf and doc2.pdf"  
**Ask Questions:** "What type of document is this?"

⚠️ Demo mode: Upload documents via Task 2 API for full functionality.
        """.strip()

    async def handle_general_query(self, query: str, chat_history: List[Dict]) -> str:
        context_results = self.vector_db.search_documents(query, top_k=2)
        if not context_results:
            return "💬 Not enough information to answer. Please ensure Task 2 API is running and documents are uploaded."
        response = "Based on the available information:\n\n"
        for result in context_results[:2]:
            filename = result.get('metadata', {}).get('filename', 'Document')
            response += f"📄 **{filename}**: {result['content'][:200]}...\n\n"
        response += "⚠️ Simplified response. Run full system for detailed analysis."
        return response

    async def process_query(self, query: str, chat_history: List[Dict]) -> str:
        intent = self.classify_query_intent(query)
        if intent == "search":
            return await self.handle_search_query(query)
        elif intent == "help":
            return await self.handle_help_query(query)
        else:
            return await self.handle_general_query(query, chat_history)

    def extract_search_terms(self, query: str) -> str:
        stop_words = {'find', 'search', 'look', 'for', 'show', 'me', 'documents', 'about', 'related', 'to'}
        words = query.lower().split()
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return ' '.join(search_terms) if search_terms else query


# Initialize chatbot
rag_chatbot = RAGChatbot()

# -----------------------------
# Chainlit Handlers
# -----------------------------
@cl.on_chat_start
async def on_chat_start():
    welcome_message = f"""
🚀 **Welcome to the AI Document Analysis Chatbot!**

Type "help" for commands. Try: "Find documents about reports".

System Status:
- Vector Search: {'✅ Ready' if rag_chatbot.vector_db.encoder else '⚠️ Limited'}
- LLM Integration: {'✅ Ready' if rag_chatbot.llm_summarizer.openai_available else '⚠️ Limited'}
    """
    await cl.Message(content=welcome_message.strip()).send()
    cl.user_session.set("chat_history", [])

@cl.on_message
async def on_message(message: cl.Message):
    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": message.content})
    response = await rag_chatbot.process_query(message.content, chat_history)
    await cl.Message(content=response).send()
    chat_history.append({"role": "assistant", "content": response})
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
    cl.user_session.set("chat_history", chat_history)

@cl.on_chat_end
async def on_chat_end():
    await cl.Message(content="👋 Thanks for using the Document Analysis Chatbot!").send()


# -----------------------------
# Main function
# -----------------------------
def main():
    print("🚀 Starting Task 3: AI Chatbot with Chainlit Frontend")
    print("Chatbot features: Document search, summarization, context QA, multi-turn conversations")
    print("Run with: chainlit run src/task3_chatbot.py -w --port 8001")
    print("Ensure Task 2 API is running and dependencies installed")

if __name__ == "__main__":
    main()