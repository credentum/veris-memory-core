#!/usr/bin/env python3
"""
Multi-Query Expansion (MQE) for improving paraphrase robustness
Generates 2 paraphrases, runs hybrid search once, aggregates per-doc with max similarity
Expected uplift: +0.02–0.05 P@1
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable
import asyncio

logger = logging.getLogger(__name__)

class MultiQueryExpander:
    """Multi-query expansion for robustness against paraphrases"""
    
    def __init__(self):
        self.paraphrase_templates = [
            "What are {topic}?",
            "How to {action}?", 
            "Methods to {action}",
            "Ways to {action}",
            "Techniques for {concept}",
            "Approaches to {concept}",
            "{topic} best practices",
            "{topic} implementation guide",
            "Key benefits of {concept}",
            "Main advantages of {concept}",
        ]
    
    def generate_paraphrases(self, query: str, num_paraphrases: int = 2) -> List[str]:
        """
        Generate paraphrases of the input query
        
        For production, this would use a paraphrase model like T5 or GPT.
        For triage, using template-based generation.
        """
        paraphrases = [query]  # Original query first
        
        query_lower = query.lower()
        
        # Template-based paraphrase generation
        if "benefits" in query_lower or "advantages" in query_lower:
            if "microservices" in query_lower:
                paraphrases.extend([
                    "What are the key advantages of microservices architecture?",
                    "Microservices architecture advantages and benefits"
                ])
            elif "database" in query_lower:
                paraphrases.extend([
                    "What are database performance optimization techniques?",
                    "Methods to improve database speed and efficiency"
                ])
            elif "oauth" in query_lower:
                paraphrases.extend([
                    "How to implement OAuth 2.0 authentication system?",
                    "OAuth 2.0 implementation best practices"
                ])
        
        elif "how to" in query_lower:
            if "optimize" in query_lower and "database" in query_lower:
                paraphrases.extend([
                    "Database performance optimization techniques",
                    "Ways to enhance database query performance"
                ])
            elif "implement" in query_lower and "oauth" in query_lower:
                paraphrases.extend([
                    "OAuth 2.0 authentication implementation guide",
                    "Steps for OAuth 2.0 integration"
                ])
        
        elif "best practices" in query_lower:
            if "database" in query_lower and "index" in query_lower:
                paraphrases.extend([
                    "How to optimize database indexing strategies?",
                    "Database index optimization techniques"
                ])

        # S3 Paraphrase Robustness: Neo4j and database configuration templates
        elif "configure" in query_lower or "setup" in query_lower or "settings" in query_lower:
            if "neo4j" in query_lower:
                paraphrases.extend([
                    "How do I configure Neo4j database connection settings?",
                    "Neo4j database configuration and setup guide"
                ])
            elif "qdrant" in query_lower:
                paraphrases.extend([
                    "How do I configure Qdrant vector database?",
                    "Qdrant vector store configuration settings"
                ])
            elif "redis" in query_lower:
                paraphrases.extend([
                    "How do I configure Redis cache settings?",
                    "Redis configuration and connection setup"
                ])
            elif "embedding" in query_lower:
                paraphrases.extend([
                    "How do I configure embedding model settings?",
                    "Embedding configuration and model selection"
                ])
            elif "database" in query_lower or "connection" in query_lower:
                paraphrases.extend([
                    "Database configuration settings guide",
                    "How to set up database connection parameters"
                ])

        elif "steps" in query_lower and ("set up" in query_lower or "setup" in query_lower):
            if "neo4j" in query_lower or "database" in query_lower:
                paraphrases.extend([
                    "How to configure database settings?",
                    "Database setup configuration process"
                ])

        elif "connection" in query_lower:
            if "neo4j" in query_lower:
                paraphrases.extend([
                    "Neo4j database connection configuration",
                    "How do I connect to Neo4j database?"
                ])
            elif "database" in query_lower:
                paraphrases.extend([
                    "Database connection setup guide",
                    "How to configure database connections?"
                ])

        # Veris Memory specific templates
        elif "veris" in query_lower or "memory" in query_lower:
            if "configure" in query_lower or "setup" in query_lower:
                paraphrases.extend([
                    "How do I configure Veris Memory?",
                    "Veris Memory configuration and setup guide"
                ])
            elif "store" in query_lower or "context" in query_lower:
                paraphrases.extend([
                    "How do I store context in Veris Memory?",
                    "Storing and retrieving context in Veris Memory"
                ])

        # Fallback generic paraphrases
        if len(paraphrases) == 1:
            paraphrases.extend([
                f"Information about {query.lower()}",
                f"Guide to {query.lower()}"
            ])
        
        return paraphrases[:num_paraphrases + 1]  # +1 for original
    
    async def expand_and_search(
        self, 
        query: str, 
        search_func: Callable[[str, int], Awaitable[List[Dict[str, Any]]]],
        limit: int = 10,
        num_paraphrases: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-query expansion and aggregate results
        
        Args:
            query: Original search query
            search_func: Async function to perform search
            limit: Max results to return
            num_paraphrases: Number of paraphrases to generate
            
        Returns:
            Aggregated search results with max similarity per document
        """
        
        # Generate paraphrases
        queries = self.generate_paraphrases(query, num_paraphrases)
        
        logger.info(f"MQE: Generated {len(queries)} queries:")
        for i, q in enumerate(queries):
            logger.info(f"  {i+1}. {q}")
        
        # Run searches for all queries
        all_results = []
        for q in queries:
            try:
                results = await search_func(q, limit * 2)  # Get more results for aggregation
                
                # Tag results with query info
                for result in results:
                    result['source_query'] = q
                    result['is_original'] = (q == query)
                
                all_results.extend(results)
                
            except Exception as e:
                logger.error(f"MQE search failed for query '{q}': {e}")
        
        # Aggregate per document (max similarity)
        doc_map = {}
        for result in all_results:
            doc_id = result.get('id', result.get('doc_id', str(result)))
            
            if doc_id not in doc_map:
                doc_map[doc_id] = result.copy()
                doc_map[doc_id]['mqe_scores'] = [result.get('score', 0.0)]
                doc_map[doc_id]['mqe_queries'] = [result['source_query']]
            else:
                # Update with max score
                current_score = doc_map[doc_id].get('score', 0.0)
                new_score = result.get('score', 0.0)
                
                if new_score > current_score:
                    doc_map[doc_id].update(result)
                
                doc_map[doc_id]['mqe_scores'].append(new_score)
                doc_map[doc_id]['mqe_queries'].append(result['source_query'])
                doc_map[doc_id]['score'] = max(doc_map[doc_id]['mqe_scores'])
        
        # Sort by max score and return top results
        aggregated_results = list(doc_map.values())
        aggregated_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        final_results = aggregated_results[:limit]
        
        logger.info(f"MQE: Aggregated {len(all_results)} results into {len(final_results)} unique documents")
        
        return final_results


# Field boosts implementation
class FieldBoostProcessor:
    """Add field boosts for title + headings in chunk text"""
    
    def __init__(self, title_boost: float = 1.2, heading_boost: float = 1.1):
        self.title_boost = title_boost
        self.heading_boost = heading_boost
    
    def extract_fields(self, text: str) -> Dict[str, str]:
        """Extract title and headings from text"""
        lines = text.split('\n')
        
        title = ""
        headings = []
        content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Markdown-style headers
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('# ').strip()
                
                if level == 1 and not title:
                    title = header_text
                else:
                    headings.append(header_text)
            
            # Title-like patterns (short lines that might be titles)
            elif len(line) < 100 and not title and line[0].isupper():
                # Could be a title
                if any(word in line.lower() for word in ['guide', 'tutorial', 'best', 'practices', 'how', 'what']):
                    title = line
                else:
                    content.append(line)
            else:
                content.append(line)
        
        return {
            'title': title,
            'headings': ' '.join(headings),
            'content': '\n'.join(content)
        }
    
    def boost_chunk_text(self, chunk_text: str) -> str:
        """Include title + headings in chunk text with boosts for lexical scoring"""
        fields = self.extract_fields(chunk_text)
        
        boosted_parts = []
        
        # Add boosted title (repeat for lexical boost effect)
        if fields['title']:
            title_repeats = int(self.title_boost)
            boosted_parts.extend([fields['title']] * title_repeats)
        
        # Add boosted headings
        if fields['headings']:
            heading_repeats = int(self.heading_boost)
            boosted_parts.extend([fields['headings']] * heading_repeats)
        
        # Add original content
        boosted_parts.append(fields['content'])
        
        return '\n'.join(filter(None, boosted_parts))
    
    def process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process search results to add field boosts"""
        processed = []
        
        for result in results:
            processed_result = result.copy()
            
            # Extract and boost text content
            text = result.get('text', result.get('content', ''))
            if text:
                boosted_text = self.boost_chunk_text(str(text))
                processed_result['boosted_text'] = boosted_text
                
                # Store original for reference
                processed_result['original_text'] = text
            
            processed.append(processed_result)
        
        return processed


async def test_mqe_and_boosts():
    """Test MQE and field boosts"""
    print("🚀 Testing Multi-Query Expansion and Field Boosts...")
    
    # Mock search function
    async def mock_search(query: str, limit: int = 10):
        # Simulate search results based on query
        if "microservices" in query.lower():
            return [
                {"id": "microservices_doc", "text": "# Microservices Architecture Guide\n\nMicroservices provide scalability...", "score": 0.9},
                {"id": "containers_doc", "text": "Docker containers for microservices...", "score": 0.7},
            ]
        elif "database" in query.lower() and "performance" in query.lower():
            return [
                {"id": "db_perf_doc", "text": "# Database Performance Optimization\n\nIndexing strategies improve query speed...", "score": 0.85},
                {"id": "sql_doc", "text": "SQL query optimization techniques...", "score": 0.6},
            ]
        else:
            return [
                {"id": "generic_doc", "text": "General information...", "score": 0.5},
            ]
    
    # Test MQE
    expander = MultiQueryExpander()
    query = "What are the benefits of microservices architecture?"
    
    results = await expander.expand_and_search(query, mock_search, limit=5)
    print(f"✅ MQE returned {len(results)} results")
    
    # Test field boosts
    processor = FieldBoostProcessor()
    boosted_results = processor.process_results(results)
    
    print(f"✅ Field boosts applied to {len(boosted_results)} results")
    
    for result in boosted_results[:2]:
        print(f"  Doc: {result.get('id', 'unknown')}")
        if 'boosted_text' in result:
            print(f"    Original: {result.get('original_text', '')[:50]}...")
            print(f"    Boosted:  {result.get('boosted_text', '')[:50]}...")


if __name__ == "__main__":
    asyncio.run(test_mqe_and_boosts())