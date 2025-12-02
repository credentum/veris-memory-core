"""
Deterministic fact storage using Redis for fast fact lookup.

This module provides the foundation for reliable fact retrieval by storing
user facts in a structured key-value format with lineage tracking.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, asdict
import logging
from redis import Redis

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """Structured fact with metadata and lineage."""
    value: Any
    confidence: float
    source_turn_id: str
    updated_at: str
    provenance: str
    attribute: str
    user_id: str
    namespace: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        return cls(**data)


class FactStore:
    """
    Redis-backed deterministic fact storage with lineage tracking.
    
    Key pattern: facts:{namespace}:{user_id}:{attribute}
    Supports fact updates with last-write-wins and historical tracking.
    """

    def __init__(self, redis_client: Redis, ttl_seconds: int = 86400 * 30) -> None:  # 30 days default
        self.redis = redis_client
        self.ttl = ttl_seconds
        self._fact_key_prefix = "facts"
        self._history_key_prefix = "fact_history"

    def _make_fact_key(self, namespace: str, user_id: str, attribute: str) -> str:
        """Generate Redis key for a fact."""
        return f"{self._fact_key_prefix}:{namespace}:{user_id}:{attribute}"

    def _make_history_key(self, namespace: str, user_id: str, attribute: str) -> str:
        """Generate Redis key for fact history."""
        return f"{self._history_key_prefix}:{namespace}:{user_id}:{attribute}"

    def _make_user_facts_pattern(self, namespace: str, user_id: str) -> str:
        """Generate Redis pattern for all user facts."""
        return f"{self._fact_key_prefix}:{namespace}:{user_id}:*"

    def store_fact(
        self,
        namespace: str,
        user_id: str,
        attribute: str,
        value: Any,
        confidence: float = 1.0,
        source_turn_id: str = "",
        provenance: str = "user_input"
    ) -> None:
        """
        Store a fact with lineage tracking.
        
        Args:
            namespace: Isolation namespace (agent_id)
            user_id: User identifier
            attribute: Fact attribute (name, email, preferences.food, etc.)
            value: Fact value
            confidence: Confidence score [0.0, 1.0]
            source_turn_id: Source conversation turn
            provenance: Source of the fact
        """
        if not namespace or not user_id or not attribute:
            raise ValueError("namespace, user_id, and attribute are required")

        fact = Fact(
            value=value,
            confidence=confidence,
            source_turn_id=source_turn_id,
            updated_at=datetime.utcnow().isoformat(),
            provenance=provenance,
            attribute=attribute,
            user_id=user_id,
            namespace=namespace
        )

        fact_key = self._make_fact_key(namespace, user_id, attribute)
        history_key = self._make_history_key(namespace, user_id, attribute)

        # Store previous fact in history before updating
        existing_fact = self.redis.get(fact_key)
        if existing_fact:
            history_entry = {
                "fact": existing_fact.decode('utf-8'),
                "replaced_at": datetime.utcnow().isoformat(),
                "replaced_by": fact.source_turn_id
            }
            self.redis.lpush(history_key, json.dumps(history_entry))
            self.redis.expire(history_key, self.ttl)

        # Store new fact
        fact_json = json.dumps(fact.to_dict())
        self.redis.setex(fact_key, self.ttl, fact_json)

        logger.info(f"Stored fact: {namespace}:{user_id}:{attribute} = {value}")

    def get_fact(self, namespace: str, user_id: str, attribute: str) -> Optional[Fact]:
        """
        Retrieve a fact by namespace, user, and attribute.
        
        Returns None if fact doesn't exist.
        """
        if not namespace or not user_id or not attribute:
            return None

        fact_key = self._make_fact_key(namespace, user_id, attribute)
        fact_data = self.redis.get(fact_key)

        if not fact_data:
            return None

        try:
            fact_dict = json.loads(fact_data.decode('utf-8'))
            return Fact.from_dict(fact_dict)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse fact {fact_key}: {e}")
            return None

    def get_user_facts(self, namespace: str, user_id: str) -> Dict[str, Fact]:
        """
        Retrieve all facts for a user.
        
        Returns dict mapping attribute -> Fact.
        """
        if not namespace or not user_id:
            return {}

        pattern = self._make_user_facts_pattern(namespace, user_id)
        fact_keys = self.redis.keys(pattern)

        facts = {}
        for key in fact_keys:
            fact_data = self.redis.get(key)
            if fact_data:
                try:
                    fact_dict = json.loads(fact_data.decode('utf-8'))
                    fact = Fact.from_dict(fact_dict)
                    facts[fact.attribute] = fact
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Failed to parse fact {key}: {e}")

        return facts

    def delete_fact(self, namespace: str, user_id: str, attribute: str) -> bool:
        """
        Delete a specific fact.
        
        Returns True if fact was deleted, False if it didn't exist.
        """
        if not namespace or not user_id or not attribute:
            return False

        fact_key = self._make_fact_key(namespace, user_id, attribute)
        history_key = self._make_history_key(namespace, user_id, attribute)

        deleted_count = self.redis.delete(fact_key, history_key)
        return deleted_count > 0

    def delete_user_facts(self, namespace: str, user_id: str) -> int:
        """
        Delete all facts for a user (forget-me functionality).
        
        Returns number of facts deleted.
        """
        if not namespace or not user_id:
            return 0

        # Get all fact keys
        fact_pattern = self._make_user_facts_pattern(namespace, user_id)
        fact_keys = self.redis.keys(fact_pattern)

        # Get all history keys
        history_pattern = f"{self._history_key_prefix}:{namespace}:{user_id}:*"
        history_keys = self.redis.keys(history_pattern)

        all_keys = fact_keys + history_keys
        if not all_keys:
            return 0

        deleted_count = self.redis.delete(*all_keys)
        logger.info(f"Deleted {deleted_count} fact entries for {namespace}:{user_id}")
        return deleted_count

    def get_fact_history(self, namespace: str, user_id: str, attribute: str) -> List[Dict[str, Any]]:
        """
        Get the update history for a fact.
        
        Returns list of historical fact versions, newest first.
        """
        if not namespace or not user_id or not attribute:
            return []

        history_key = self._make_history_key(namespace, user_id, attribute)
        history_data = self.redis.lrange(history_key, 0, -1)

        history = []
        for entry_data in history_data:
            try:
                entry = json.loads(entry_data.decode('utf-8'))
                history.append(entry)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse history entry: {e}")

        return history

    def search_facts_by_value(self, namespace: str, user_id: str, search_value: str) -> List[Tuple[str, Fact]]:
        """
        Search facts by value content (for debugging/admin).
        
        Returns list of (attribute, fact) Tuples.
        """
        facts = self.get_user_facts(namespace, user_id)
        matches = []

        search_lower = search_value.lower()
        for attribute, fact in facts.items():
            if isinstance(fact.value, str) and search_lower in fact.value.lower():
                matches.append((attribute, fact))

        return matches

    def get_stats(self, namespace: str) -> Dict[str, Union[int, float]]:
        """
        Get storage statistics for a namespace.
        
        Returns dict with counts and storage info.
        """
        fact_pattern = f"{self._fact_key_prefix}:{namespace}:*"
        fact_keys = self.redis.keys(fact_pattern)

        user_counts = {}
        total_facts = len(fact_keys)

        for key in fact_keys:
            # Extract user_id from key pattern
            key_parts = key.decode('utf-8').split(':')
            if len(key_parts) >= 3:
                user_id = key_parts[2]
                user_counts[user_id] = user_counts.get(user_id, 0) + 1

        return {
            "total_facts": total_facts,
            "unique_users": len(user_counts),
            "avg_facts_per_user": total_facts / len(user_counts) if user_counts else 0,
            "max_facts_per_user": max(user_counts.values()) if user_counts else 0
        }