#!/usr/bin/env python3
"""
Search Enhancement Module for Veris Memory

This module implements search improvements to address issues identified by workspace 001:
- Exact match boosting for filenames and keywords
- Context type weighting for better relevance
- Recency balancing to prevent new content from overshadowing older relevant content
- Technical term recognition for code-specific searches

Author: Workspace 002
Date: 2025-08-19
"""

import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Context type weights for different content types
# Higher weights = higher priority in search results
CONTEXT_TYPE_WEIGHTS = {
    'code': 2.0,           # Actual code files (highest priority for technical searches)
    'log': 1.5,            # Code and technical logs
    'documentation': 1.3,  # Technical documentation
    'infrastructure_improvement': 1.0,  # Infrastructure work
    'accomplishment': 0.8,  # Achievement logs
    'conversation': 0.5,    # Conversational content (lowest priority)
    'unknown': 1.0         # Default weight
}

# Technical vocabulary for boosting technical searches
TECHNICAL_TERMS = {
    # Programming languages
    'python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp', 'c++',
    
    # Programming concepts
    'function', 'class', 'method', 'variable', 'constant', 'parameter',
    'argument', 'return', 'import', 'export', 'module', 'package',
    
    # Data structures
    'array', 'list', 'dict', 'dictionary', 'set', 'tuple', 'map', 'hash',
    'tree', 'graph', 'node', 'edge', 'queue', 'stack',
    
    # Web/API terms
    'api', 'endpoint', 'rest', 'graphql', 'http', 'https', 'request',
    'response', 'header', 'payload', 'json', 'xml', 'webhook',
    
    # Database terms
    'database', 'query', 'schema', 'model', 'table', 'column', 'index',
    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
    
    # Infrastructure terms
    'docker', 'kubernetes', 'deployment', 'container', 'pod', 'service',
    'server', 'client', 'proxy', 'load balancer', 'nginx', 'apache',
    
    # Veris Memory specific terms
    'vector', 'embedding', 'graph', 'neo4j', 'qdrant', 'redis',
    'mcp', 'context', 'storage', 'retrieval', 'search', 'ranking',
    
    # File extensions (without dots)
    'py', 'js', 'ts', 'jsx', 'tsx', 'json', 'yaml', 'yml', 'md',
    'dockerfile', 'sh', 'bash', 'sql', 'css', 'html'
}


def calculate_exact_match_boost(
    query: str, 
    content: Union[str, Dict], 
    metadata: Optional[Dict] = None
) -> float:
    """
    Calculate boost factor for exact matches in filenames and keywords.
    
    This addresses the issue where searching for "server.py" returns conversation
    contexts instead of the actual server.py file.
    
    Args:
        query: The search query string
        content: The content to check (can be string or dict)
        metadata: Optional metadata dict containing file_path, title, etc.
    
    Returns:
        Float boost factor (1.0 = no boost, >1.0 = boosted)
    """
    boost = 1.0
    query_lower = query.lower().strip()
    
    # Handle dict content
    if isinstance(content, dict):
        content_str = str(content.get('text', '')) + ' ' + str(content.get('content', ''))
    else:
        content_str = str(content)
    
    content_lower = content_str.lower()
    
    # Check metadata for exact matches
    if metadata:
        # Check for exact filename match (highest boost)
        if 'file_path' in metadata:
            file_path = metadata['file_path']
            filename = os.path.basename(file_path).lower()
            
            # Exact filename match
            if filename == query_lower or query_lower == filename:
                boost *= 5.0  # 5x boost for exact filename match
                logger.debug(f"Exact filename match: {filename} == {query_lower}, boost: 5.0x")
            # Filename contains query or query contains filename
            elif filename in query_lower or query_lower in filename:
                boost *= 3.0  # 3x boost for partial filename match
                logger.debug(f"Partial filename match: {filename} in {query_lower}, boost: 3.0x")
        
        # Check title for exact match
        if 'title' in metadata:
            title_lower = metadata['title'].lower()
            if query_lower in title_lower:
                boost *= 2.0  # 2x boost for title match
                logger.debug(f"Title match: {query_lower} in {title_lower}, boost: 2.0x")
    
    # Check for exact phrase match in content (only if no filename match)
    if boost == 1.0 and query_lower in content_lower:
        boost *= 1.5  # 1.5x boost for exact phrase match
        logger.debug(f"Exact phrase match in content, boost: 1.5x")
    
    # Check for all keywords present (only if no other boost applied yet)
    if boost == 1.0:
        keywords = query_lower.split()
        if len(keywords) > 1:
            all_present = all(keyword in content_lower for keyword in keywords)
            if all_present:
                boost *= 1.3  # 1.3x boost if all keywords present
                logger.debug(f"All keywords present, boost: 1.3x")
    
    return boost


def apply_context_type_weight(result: Dict[str, Any]) -> float:
    """
    Apply weight based on context type to prioritize code/documentation over conversations.
    
    This addresses the issue where technical searches return conversational contexts
    instead of code contexts.
    
    Args:
        result: Search result dict with payload containing type and category
    
    Returns:
        Float weight factor based on context type
    """
    # Extract type information from result
    payload = result.get('payload', {})
    
    # Check multiple fields for type information
    type_value = payload.get('type', 'unknown')
    category = payload.get('category', 'unknown')
    content_type = payload.get('content_type', 'unknown')
    
    # Determine effective type with priority
    effective_type = 'unknown'
    
    # Highest priority: explicit category
    if category in ['code', 'python_code', 'javascript_code']:
        effective_type = 'code'
    elif category == 'documentation':
        effective_type = 'documentation'
    elif category == 'configuration':
        effective_type = 'documentation'  # Treat config as documentation
    # Second priority: type field
    elif type_value in CONTEXT_TYPE_WEIGHTS:
        effective_type = type_value
    # Third priority: content_type field
    elif content_type in CONTEXT_TYPE_WEIGHTS:
        effective_type = content_type
    
    weight = CONTEXT_TYPE_WEIGHTS.get(effective_type, 1.0)
    
    logger.debug(f"Context type: {effective_type}, weight: {weight}")
    return weight


def calculate_recency_decay(
    created_at: Optional[Union[str, datetime]], 
    base_score: float,
    decay_rate: float = 7.0
) -> float:
    """
    Apply exponential decay based on content age to balance recent vs older content.
    
    This addresses the issue where recent content completely overshadows older
    relevant content.
    
    Args:
        created_at: Creation timestamp (ISO string or datetime)
        base_score: Original relevance score
        decay_rate: Days for 50% decay (default: 7 days)
    
    Returns:
        Adjusted score with recency decay applied
    """
    if not created_at:
        return base_score
    
    try:
        # Parse timestamp
        if isinstance(created_at, str):
            # Handle ISO format with or without timezone
            created_dt = datetime.fromisoformat(
                created_at.replace('Z', '+00:00').replace('T', ' ')
            )
        else:
            created_dt = created_at
        
        # Calculate age in days
        now = datetime.now()
        if created_dt.tzinfo:
            # Make now timezone-aware if created_dt is
            from datetime import timezone
            now = datetime.now(timezone.utc)
        
        age_days = (now - created_dt).days
        
        # Apply exponential decay: score * exp(-age_days / decay_rate)
        # decay_rate=7 gives 50% weight after 1 week, 25% after 2 weeks
        decay_factor = math.exp(-age_days / decay_rate)
        
        # Don't let decay go below 10% to maintain some relevance for old content
        decay_factor = max(decay_factor, 0.1)
        
        logger.debug(f"Age: {age_days} days, decay factor: {decay_factor:.2f}")
        return base_score * decay_factor
        
    except Exception as e:
        logger.warning(f"Failed to calculate recency decay: {e}")
        return base_score


def calculate_technical_boost(query: str, content: Union[str, Dict]) -> float:
    """
    Boost results containing technical terms when query is technical.
    
    This helps technical searches find code and documentation instead of
    conversational content.
    
    Args:
        query: The search query string
        content: The content to analyze
    
    Returns:
        Float boost factor for technical content
    """
    boost = 1.0
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    
    # Handle dict content
    if isinstance(content, dict):
        content_str = str(content.get('text', '')) + ' ' + str(content.get('content', ''))
    else:
        content_str = str(content)
    
    content_lower = content_str.lower()
    
    # Check if query contains technical terms
    technical_query = bool(query_terms & TECHNICAL_TERMS)
    
    # Also check for file extensions in query (e.g., "server.py")
    has_extension = any(f'.{ext}' in query_lower for ext in ['py', 'js', 'ts', 'md', 'json', 'yaml'])
    technical_query = technical_query or has_extension
    
    if technical_query:
        # Count technical terms in content
        tech_count = sum(1 for term in TECHNICAL_TERMS if term in content_lower)
        
        if tech_count > 0:
            # Logarithmic boost to avoid over-weighting
            # This gives diminishing returns for many technical terms
            boost *= (1 + math.log(1 + tech_count) * 0.3)
            logger.debug(f"Technical terms found: {tech_count}, boost: {boost:.2f}x")
    
    return boost


def apply_search_enhancements(
    results: List[Dict[str, Any]], 
    query: str,
    enable_exact_match: bool = True,
    enable_type_weighting: bool = True,
    enable_recency_decay: bool = True,
    enable_technical_boost: bool = True
) -> List[Dict[str, Any]]:
    """
    Apply all search enhancements to a list of search results.
    
    This is the main function that orchestrates all improvements.
    
    Args:
        results: List of search results to enhance
        query: The original search query
        enable_exact_match: Enable exact match boosting
        enable_type_weighting: Enable context type weighting
        enable_recency_decay: Enable recency decay
        enable_technical_boost: Enable technical term boosting
    
    Returns:
        Enhanced and re-sorted list of results
    """
    enhanced_results = []
    
    for result in results:
        # Start with original score
        original_score = result.get('score', 0.0)
        enhanced_score = original_score
        
        # Extract content and metadata
        payload = result.get('payload', {})
        content = payload.get('content', payload)
        metadata = payload.get('metadata', payload)
        
        # Track individual boosts for debugging
        boosts = {
            'original': original_score,
            'exact_match': 1.0,
            'type_weight': 1.0,
            'recency': 1.0,
            'technical': 1.0
        }
        
        # Apply exact match boost
        if enable_exact_match:
            exact_boost = calculate_exact_match_boost(query, content, metadata)
            enhanced_score *= exact_boost
            boosts['exact_match'] = exact_boost
        
        # Apply context type weight
        if enable_type_weighting:
            type_weight = apply_context_type_weight(result)
            enhanced_score *= type_weight
            boosts['type_weight'] = type_weight
        
        # Apply recency decay (before other boosts to maintain proportions)
        if enable_recency_decay:
            created_at = metadata.get('created_at') or payload.get('created_at')
            if created_at:
                # Apply decay to the enhanced score
                decayed_score = calculate_recency_decay(created_at, enhanced_score)
                recency_factor = decayed_score / enhanced_score if enhanced_score > 0 else 1.0
                enhanced_score = decayed_score
                boosts['recency'] = recency_factor
        
        # Apply technical boost
        if enable_technical_boost:
            tech_boost = calculate_technical_boost(query, content)
            enhanced_score *= tech_boost
            boosts['technical'] = tech_boost
        
        # Create enhanced result
        enhanced_result = result.copy()
        enhanced_result['enhanced_score'] = enhanced_score
        enhanced_result['original_score'] = original_score
        enhanced_result['score_boosts'] = boosts
        enhanced_result['score'] = enhanced_score  # Update main score field
        
        enhanced_results.append(enhanced_result)
    
    # Sort by enhanced score (highest first)
    enhanced_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
    
    # Log top results for debugging
    if enhanced_results:
        logger.info(f"Top result after enhancement: score={enhanced_results[0]['enhanced_score']:.3f}, "
                   f"boosts={enhanced_results[0]['score_boosts']}")
    
    return enhanced_results


def is_technical_query(query: str) -> bool:
    """
    Determine if a query is technical in nature.
    
    Args:
        query: The search query string
    
    Returns:
        True if query appears to be technical
    """
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    
    # Check for technical terms
    has_technical_terms = bool(query_terms & TECHNICAL_TERMS)
    
    # Check for file extensions
    has_extension = any(f'.{ext}' in query_lower for ext in ['py', 'js', 'ts', 'md', 'json'])
    
    # Check for code-like patterns
    has_code_patterns = any(pattern in query_lower for pattern in ['()', '[]', '{}', '->', '=>', 'function', 'class', 'def'])
    
    return has_technical_terms or has_extension or has_code_patterns