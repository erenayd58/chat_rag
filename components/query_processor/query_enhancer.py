# /Users/murseltasgin/projects/chat_rag/components/query_processor/query_enhancer.py
"""
Query enhancement and understanding
"""
import json
from typing import List, Dict, Any
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from components.llm import BaseLLM
from core.models import QueryClarification, SearchStrategy, SearchQuery
from core.exceptions import LLMException, RESOURCE_CONTROL_EXCEPTIONS
from utils.logger import get_logger, RAGLogger
import re

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class QueryEnhancer:
    """Enhances queries for better retrieval with conversation awareness"""
    
    def __init__(self, llm_model: BaseLLM):
        """
        Initialize query enhancer
        
        Args:
            llm_model: LLM model instance
        """
        self.llm_model = llm_model
        self.stop_words = set(stopwords.words('english'))
        self.logger = get_logger("QueryEnhancer")
    
    def _extract_json(self, content: str) -> str:
        """
        Extract a JSON object from an LLM response robustly and sanitize it.
        - Removes markdown code fences
        - Extracts substring between first '{' and last '}'
        - Removes invalid control characters that break json.loads
        - Trims trailing commas and attempts minimal repair
        """
        if not content:
            return "{}"
        text = content.strip()
        # Strip code fences
        if '```json' in text:
            try:
                text = text.split('```json', 1)[1].split('```', 1)[0].strip()
            except Exception:
                pass
        elif '```' in text:
            try:
                text = text.split('```', 1)[1].split('```', 1)[0].strip()
            except Exception:
                pass
        # Extract between first '{' and last '}', else fallback to list between '[' and ']'
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}')
            text = text[start:end+1]
        elif '[' in text and ']' in text:
            start = text.find('[')
            end = text.rfind(']')
            list_text = text[start:end+1]
            text = '{"queries": ' + list_text + '}'
        else:
            # No JSON structure detected
            return "{}"
        # Remove invalid control chars except \t, \n, \r
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
        # Trim trailing commas before closing braces/brackets
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text
    
    def clarify_query_with_context(
        self,
        current_query: str,
        conversation_context: str
    ) -> QueryClarification:
        """
        Clarify and resolve ambiguous queries using conversation history
        
        Args:
            current_query: Current user query
            conversation_context: Recent conversation context
        
        Returns:
            QueryClarification object
        """
        if not conversation_context:
            return QueryClarification(
                clarified_query=current_query,
                needs_clarification=False,
                entities=[],
                resolution_notes="No conversation history available",
                confidence="high"
            )
        
        prompt = f"""Given the conversation history and the current user query, determine if the query needs clarification and resolve any ambiguous references.

Conversation History:
{conversation_context}

Current Query: "{current_query}"

Analyze:
1. Does the query reference previous context (pronouns like "he", "she", "it", "his", "her", "that")?
2. Does it reference entities mentioned earlier?
3. Is the query incomplete without context?
4. If ambiguous, produce a complete, standalone version that replaces pronouns with the correct entities from the history.
5. Ensure the clarified query preserves the user's original intent and specificity (names, dates, roles, etc.).

Provide a JSON response. For any natural-language values, use the user's input language.
{{
    "needs_clarification": true/false,
    "clarified_query": "Complete standalone version of the query",
    "entities": ["entity1", "entity2"],
    "resolution_notes": "Brief explanation of how query was resolved",
    "confidence": "high/medium/low"
}}

Response:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert at understanding conversational context and resolving ambiguous references. Return only valid JSON. Use the user's input language for any natural-language content in values; keep JSON keys in English."},
                {"role": "user", "content": prompt}
            ]
            
            # Log LLM input
            RAGLogger.log_llm_request(self.logger, messages, 0.2, 400)
            response = self.llm_model.generate(messages, temperature=0.2, max_tokens=400, format='json')
            # Log LLM response
            RAGLogger.log_llm_response(self.logger, response, success=True)
            
            # Validate response
            if not response or not response.strip():
                self.logger.warning("Empty clarification response from the model")
                return QueryClarification(
                    clarified_query=current_query,
                    needs_clarification=False,
                    entities=[],
                    resolution_notes="LLM returned empty response",
                    confidence="low"
                )
            
            # Parse JSON response
            content = self._extract_json(response)
            if not content or content == "{}":
                # Graceful default when extraction fails
                self.logger.warning("Clarification JSON extraction failed; using original query")
                return QueryClarification(
                    clarified_query=current_query,
                    needs_clarification=False,
                    entities=[],
                    resolution_notes="Extraction failed",
                    confidence="low"
                )
            
            # Try to fix incomplete JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to repair incomplete JSON
                if not content.endswith('}'):
                    content = content.rstrip().rstrip(',')
                    # Add missing fields with defaults if needed
                    if '"confidence"' not in content:
                        content += ',\n    "confidence": "medium"'
                    content += '\n}'
                result = json.loads(content)
            
            # Log full clarification JSON
            try:
                self.logger.debug(f"Clarification JSON: {json.dumps(result, ensure_ascii=False)}")
            except Exception:
                self.logger.debug("Clarification JSON: <unserializable>")
            return QueryClarification(
                clarified_query=result.get('clarified_query', current_query),
                needs_clarification=result.get('needs_clarification', False),
                entities=result.get('entities', []),
                resolution_notes=result.get('resolution_notes', ''),
                confidence=result.get('confidence', 'medium')
            )
            
        except RESOURCE_CONTROL_EXCEPTIONS:
            # The query is over -- its deadline passed, or capacity refused
            # it. Falling back to the original question here would hide that
            # and let the query go on spending time and slots it does not
            # have (core/exceptions.py).
            raise
        except json.JSONDecodeError as e:
            self.logger.warning(f"Clarification JSON decode failed: {e}")
            return QueryClarification(
                clarified_query=current_query,
                needs_clarification=False,
                entities=[],
                resolution_notes=f"JSON error: {str(e)}",
                confidence="low"
            )
        except Exception as e:
            self.logger.error(f"Clarification error: {e}")
            return QueryClarification(
                clarified_query=current_query,
                needs_clarification=False,
                entities=[],
                resolution_notes=f"Error: {str(e)}",
                confidence="low"
            )
    
    def determine_search_strategy(
        self,
        query: str,
        clarified_query: str = None
    ) -> SearchStrategy:
        """
        Determine the best search strategy for the query
        
        Args:
            query: Original query
            clarified_query: Clarified query (if available)
        
        Returns:
            SearchStrategy object
        """
        analysis_query = clarified_query or query
        
        prompt = f"""Analyze this search query and recommend the best retrieval strategy.

Query: "{analysis_query}"

Consider:
1. Is it a specific factual question (better for keyword/BM25)?
2. Is it conceptual or semantic (better for vector search)?
3. Does it benefit from both approaches (hybrid)?
4. How many results are likely needed?

Provide JSON response:
{{
    "recommended_strategy": "vector" or "bm25" or "hybrid",
    "reasoning": "Brief explanation",
    "query_type": "factual/conceptual/comparison/definition/procedural",
    "expected_answer_type": "specific_fact/explanation/list/comparison",
    "suggested_top_k": 3-10,
    "use_query_expansion": true/false,
    "use_reranking": true/false
}}

Response:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert in information retrieval strategy. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            RAGLogger.log_llm_request(self.logger, messages, 0.3, 250)
            response = self.llm_model.generate(messages, temperature=0.3, max_tokens=250, format='json')
            RAGLogger.log_llm_response(self.logger, response, success=True)
            
            # Validate response
            if not response or not response.strip():
                self.logger.warning("Empty strategy response from the model")
                return SearchStrategy(
                    recommended_strategy="hybrid",
                    reasoning="LLM returned empty response, using hybrid",
                    query_type="unknown",
                    expected_answer_type="unknown",
                    suggested_top_k=5,
                    use_query_expansion=True,
                    use_reranking=True
                )
            
            # Parse JSON response
            content = self._extract_json(response)
            if not content or content == "{}":
                self.logger.warning("Strategy JSON extraction failed; using hybrid defaults")
                return SearchStrategy(
                    recommended_strategy="hybrid",
                    reasoning="Extraction failed",
                    query_type="unknown",
                    expected_answer_type="unknown",
                    suggested_top_k=5,
                    use_query_expansion=True,
                    use_reranking=True
                )
            
            # Try to fix incomplete JSON by adding default values
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to repair incomplete JSON
                if not content.endswith('}'):
                    # Add missing fields and close JSON
                    content = content.rstrip().rstrip(',')
                    if '"use_query_expansion"' not in content:
                        content += ',\n    "use_query_expansion": true'
                    if '"use_reranking"' not in content:
                        content += ',\n    "use_reranking": true'
                    content += '\n}'
                result = json.loads(content)
            
            # Log full strategy JSON
            try:
                self.logger.debug(f"Strategy JSON: {json.dumps(result, ensure_ascii=False)}")
            except Exception:
                self.logger.debug("Strategy JSON: <unserializable>")
            return SearchStrategy(
                recommended_strategy=result.get('recommended_strategy', 'hybrid'),
                reasoning=result.get('reasoning', ''),
                query_type=result.get('query_type', 'unknown'),
                expected_answer_type=result.get('expected_answer_type', 'unknown'),
                suggested_top_k=result.get('suggested_top_k', 5),
                use_query_expansion=result.get('use_query_expansion', True),
                use_reranking=result.get('use_reranking', True)
            )
            
        except RESOURCE_CONTROL_EXCEPTIONS:
            raise
        except json.JSONDecodeError as e:
            self.logger.warning(f"Strategy JSON decode failed: {e}")
            return SearchStrategy(
                recommended_strategy="hybrid",
                reasoning=f"JSON error, using hybrid",
                query_type="unknown",
                expected_answer_type="unknown",
                suggested_top_k=5,
                use_query_expansion=True,
                use_reranking=True
            )
        except Exception as e:
            self.logger.error(f"Strategy error: {e}")
            return SearchStrategy(
                recommended_strategy="hybrid",
                reasoning=f"Default to hybrid due to error: {str(e)}",
                query_type="unknown",
                expected_answer_type="unknown",
                suggested_top_k=5,
                use_query_expansion=True,
                use_reranking=True
            )
    
    def generate_search_queries(
        self,
        original_query: str,
        clarified_query: str,
        strategy: SearchStrategy
    ) -> List[SearchQuery]:
        """
        Generate optimized search queries for different retrieval methods
        
        Args:
            original_query: Original query
            clarified_query: Clarified query
            strategy: Search strategy
        
        Returns:
            List of SearchQuery objects
        """
        prompt = f"""Generate optimized search queries for a retrieval system.

Original Query: "{original_query}"
Clarified Query: "{clarified_query}"
Search Strategy: {strategy.recommended_strategy}
Query Type: {strategy.query_type}

Generate 3-5 search query variations optimized for:
1. Vector search (semantic, natural language)
2. Keyword search (important terms, entities)
3. Alternative phrasings

Format as JSON:
{{
    "queries": [
        {{"text": "query text", "type": "semantic/keyword/alternative", "purpose": "brief description"}},
        ...
    ]
}}

Response:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert at query optimization for search engines. Return only valid JSON. Ensure any natural-language fields (like 'text' and 'purpose') are in the user's input language."},
                {"role": "user", "content": prompt}
            ]
            
            RAGLogger.log_llm_request(self.logger, messages, 0.6, 500)
            response = self.llm_model.generate(messages, temperature=0.6, max_tokens=500, format='json')
            RAGLogger.log_llm_response(self.logger, response, success=True)
            
            # Validate response
            if not response or not response.strip():
                self.logger.warning("Empty query-generation response from the model")
                return [
                    SearchQuery(text=clarified_query, type='original', purpose='primary'),
                    SearchQuery(text=original_query, type='original', purpose='fallback')
                ]
            
            # Parse JSON response
            content = self._extract_json(response)
            if not content or content == "{}":
                self.logger.warning("Query generation JSON extraction failed; using fallbacks")
                return [
                    SearchQuery(text=clarified_query, type='original', purpose='primary'),
                    SearchQuery(text=original_query, type='original', purpose='fallback')
                ]
            
            # Try to fix incomplete JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to repair incomplete JSON by closing arrays/objects
                if '"queries"' in content and not content.rstrip().endswith('}'):
                    # Try to close the JSON structure
                    content = content.rstrip().rstrip(',')
                    # Count unclosed braces and brackets
                    open_braces = content.count('{') - content.count('}')
                    open_brackets = content.count('[') - content.count(']')
                    for _ in range(open_brackets):
                        content += '\n]'
                    for _ in range(open_braces):
                        content += '\n}'
                result = json.loads(content)
            
            queries_list = result.get('queries', [])
            # Log full search queries JSON
            try:
                self.logger.debug(f"Search Queries JSON: {json.dumps({'queries': queries_list}, ensure_ascii=False)}")
            except Exception:
                self.logger.debug("Search Queries JSON: <unserializable>")
            
            search_queries = []
            for q in queries_list:
                search_queries.append(SearchQuery(
                    text=q.get('text', clarified_query),
                    type=q.get('type', 'original'),
                    purpose=q.get('purpose', '')
                ))
            
            return search_queries if search_queries else [
                SearchQuery(text=clarified_query, type='original', purpose='primary')
            ]
            
        except RESOURCE_CONTROL_EXCEPTIONS:
            raise
        except json.JSONDecodeError as e:
            self.logger.warning(f"Query generation JSON decode failed: {e}")
            return [
                SearchQuery(text=clarified_query, type='original', purpose='primary'),
                SearchQuery(text=original_query, type='original', purpose='fallback')
            ]
        except Exception as e:
            self.logger.error(f"Query generation error: {e}")
            return [
                SearchQuery(text=clarified_query, type='original', purpose='primary'),
                SearchQuery(text=original_query, type='original', purpose='fallback')
            ]
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms and reformulations
        
        Args:
            query: Query to expand
        
        Returns:
            List of query variations
        """
        prompt = f"""Given the user query: "{query}"

Generate 3 alternative ways to phrase this query or related search terms that would help find relevant information. Focus on:
1. Synonyms and related concepts
2. More specific formulations
3. Broader related topics

Format as a JSON list of strings.

Example:
["alternative query 1", "alternative query 2", "alternative query 3"]

Response:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a query expansion expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            RAGLogger.log_llm_request(self.logger, messages, 0.7, 200)
            response = self.llm_model.generate(messages, temperature=0.7, max_tokens=200, format='json')
            RAGLogger.log_llm_response(self.logger, response, success=True)
            
            # Parse JSON response
            content = response.strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            expanded = json.loads(content)
            return [query] + expanded[:3]
        except RESOURCE_CONTROL_EXCEPTIONS:
            raise
        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
            return [query]
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Args:
            query: Query text
        
        Returns:
            List of keywords
        """
        words = word_tokenize(query.lower())
        keywords = [
            word for word in words 
            if word.isalnum() and word not in self.stop_words and len(word) > 2
        ]
        return keywords
    
    def understand_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query intent and structure
        
        Args:
            query: Query text
        
        Returns:
            Dictionary with intent information
        """
        prompt = f"""Analyze this user query and extract:
1. Main topic/subject
2. Query type (question, instruction, keyword search, etc.)
3. Specific entities mentioned
4. Desired information type

Query: "{query}"

Format as JSON:
{{
    "main_topic": "...",
    "query_type": "...",
    "entities": [...],
    "info_type": "..."
}}

Response:"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a query understanding expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            RAGLogger.log_llm_request(self.logger, messages, 0.3, 150)
            response = self.llm_model.generate(messages, temperature=0.3, max_tokens=150, format='json')
            RAGLogger.log_llm_response(self.logger, response, success=True)
            
            # Parse JSON response
            content = response.strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            return json.loads(content)
        except RESOURCE_CONTROL_EXCEPTIONS:
            raise
        except Exception as e:
            self.logger.warning(f"Intent analysis failed: {e}")
            return {
                "main_topic": query,
                "query_type": "general",
                "entities": [],
                "info_type": "general"
            }

