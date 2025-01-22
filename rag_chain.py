from typing import List, Dict, Optional, Generator, AsyncGenerator
import logging
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from config import get_chroma_client, get_or_create_collection
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanAssistant:
    def __init__(self, model_name: str = "deepseek-chat"):
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client)
        
        # Initialize LLM based on model name
        self.model_name = model_name
        self.llm = self._initialize_llm(model_name)
        logger.info(f"✓ LLM initialized: {model_name}")

        # Define prompts
        self.system_prompt = """You are a loan document analysis assistant. Provide clear, concise answers based on the provided context.
        Keep responses focused and brief, highlighting only the most relevant information.
        If you can't find a specific answer in the context, provide a general, accurate response based on your knowledge,
        but clearly indicate that this is a general answer not from the specific documents."""

        self.general_system_prompt = """You are a knowledgeable loan and financial expert. When providing general information about loans:
        1. Be clear and concise
        2. Focus on widely accepted practices and standards
        3. Mention that this is general information and specific lenders may have different requirements
        4. Stick to factual, practical information
        5. Keep responses brief and to the point"""

        # Backup response for connection issues
        self.backup_response = """Based on general lending practices (system is experiencing connection issues, providing backup response):

Typical loan requirements usually include:
1. Proof of Identity and Income
2. Credit Score and History
3. Employment Verification
4. Bank Statements
5. Purpose of Loan
6. Collateral (for secured loans)

Note: Specific requirements vary by lender and loan type. Please check with your specific lender for exact requirements."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _initialize_llm(self, model_name: str) -> BaseLLM:
        """Initialize the appropriate LLM based on model name with retry logic"""
        try:
            if model_name == "deepseek-chat":
                return ChatOpenAI(
                    model='deepseek-chat',
                    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
                    openai_api_base='https://api.deepseek.com',
                    max_tokens=1024,
                    streaming=True,
                    request_timeout=30
                )
            elif model_name.startswith("gpt"):
                return ChatOpenAI(
                    model=model_name,
                    openai_api_key=os.getenv('OPENAI_API_KEY'),
                    max_tokens=1024,
                    streaming=True,
                    request_timeout=30
                )
            elif model_name.startswith("claude"):
                return ChatAnthropic(
                    model=model_name,
                    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                    max_tokens=1024,
                    streaming=True
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get LLM response with retry logic"""
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise

    async def get_general_answer(self, question: str) -> str:
        """Get a general answer when no relevant documents are found"""
        try:
            messages = [
                {"role": "system", "content": self.general_system_prompt},
                {"role": "user", "content": f"""Question: {question}

Please provide a general answer based on standard industry practices.
Start your response with: "Based on general industry standards (not from specific documents):"
Keep the response concise and focused."""}
            ]
            
            return await self._get_llm_response(messages)
                
        except Exception as e:
            logger.error(f"Error getting general answer: {str(e)}")
            return self.backup_response

    async def answer_question(
        self, 
        question: str,
        streaming: bool = True
    ) -> Dict:
        """Answer a question using RAG with formatted response"""
        try:
            # Search for relevant context
            results = self.collection.query(
                query_texts=[question],
                n_results=3
            )
            
            if not results["documents"][0]:
                # Get general answer if no relevant documents found
                general_answer = await self.get_general_answer(question)
                return {
                    "answer": general_answer,
                    "references": [],
                    "is_general_answer": True
                }
            
            # Format context and metadata
            contexts = []
            references = []
            
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                source = metadata.get("source", "Unknown")
                doc_type = "PDF" if source.endswith(".pdf") else "Data"
                page_or_row = metadata.get("page", metadata.get("row", "Unknown"))
                chunk_id = results["ids"][0][i]
                
                # Format reference
                ref = {
                    "source": f"[{doc_type}: {Path(source).name}]",
                    "location": f"{'Page' if doc_type == 'PDF' else 'Row'} {page_or_row}",
                    "content": doc[:150] + "..." if len(doc) > 150 else doc,
                    "id": chunk_id
                }
                references.append(ref)
                contexts.append(doc)
            
            try:
                # Try to get response from LLM
                context_text = "\n\n".join(contexts)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"""Context:\n{context_text}\n\nQuestion: {question}\n\nProvide a brief, focused answer based only on the above context."""}
                ]
                
                response_content = await self._get_llm_response(messages)
                
            except Exception as e:
                # If LLM fails, fall back to general answer
                logger.error(f"LLM response failed, using backup: {str(e)}")
                return {
                    "answer": self.backup_response,
                    "references": references,
                    "is_general_answer": True,
                    "is_backup_response": True
                }
            
            return {
                "answer": response_content,
                "references": references,
                "is_general_answer": False
            }
                
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return {
                "answer": self.backup_response,
                "references": [],
                "is_general_answer": True,
                "is_backup_response": True
            }

    async def analyze_loan_application(self, application_id: str) -> Dict:
        """Analyze a specific loan application"""
        try:
            # Get application documents
            results = self.collection.get(
                where={"metadata.application_id": application_id}
            )
            
            if not results["documents"]:
                return {
                    "status": "error",
                    "message": f"No documents found for application {application_id}"
                }
            
            # Format application data
            application_data = "\n\n".join(results["documents"])
            
            # Create analysis prompt
            analysis_prompt = f"""Please analyze this loan application:
            
            {application_data}
            
            Provide a detailed analysis including:
            1. Document completeness
            2. Risk assessment
            3. Compliance check
            4. Recommendations"""
            
            try:
                # Get analysis from LLM
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ]
                
                response = await self._get_llm_response(messages)
                
                return {
                    "status": "success",
                    "analysis": response,
                    "documents_analyzed": len(results["documents"])
                }
            except Exception as e:
                logger.error(f"LLM analysis failed: {str(e)}")
                return {
                    "status": "error",
                    "error": "Unable to generate analysis due to connection issues. Please try again later."
                }
            
        except Exception as e:
            logger.error(f"Error analyzing application: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }