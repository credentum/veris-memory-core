"""
Unit tests for Redis TTL Manager
Sprint 13 - Phase 3: Memory Management
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from src.storage.redis_manager import RedisTTLManager, RedisEventLog


class TestRedisTTLManager:
    """Test Redis TTL Manager functionality"""

    def test_init_without_client(self):
        """Test initialization without Redis client"""
        manager = RedisTTLManager()
        assert manager.redis_client is None
        assert manager.cleanup_stats["total_cleaned"] == 0

    def test_init_with_client(self):
        """Test initialization with Redis client"""
        mock_client = Mock()
        manager = RedisTTLManager(mock_client)
        assert manager.redis_client is mock_client

    def test_get_default_ttl(self):
        """Test getting default TTL values"""
        manager = RedisTTLManager()

        assert manager.get_default_ttl("scratchpad") == 3600  # 1 hour
        assert manager.get_default_ttl("session") == 604800  # 7 days
        assert manager.get_default_ttl("cache") == 300  # 5 minutes
        assert manager.get_default_ttl("temporary") == 60  # 1 minute
        assert manager.get_default_ttl("persistent") == 2592000  # 30 days

    def test_get_default_ttl_unknown_type(self):
        """Test getting default TTL for unknown type returns temporary"""
        manager = RedisTTLManager()
        assert manager.get_default_ttl("unknown") == 60  # Falls back to temporary

    def test_set_with_ttl_success(self):
        """Test setting a key with TTL"""
        mock_client = Mock()
        mock_client.setex.return_value = True

        manager = RedisTTLManager(mock_client)
        result = manager.set_with_ttl("test:key", "test_value", ttl=3600)

        assert result is True
        mock_client.setex.assert_called_once_with("test:key", 3600, "test_value")

    def test_set_with_ttl_uses_default(self):
        """Test setting a key with default TTL"""
        mock_client = Mock()
        mock_client.setex.return_value = True

        manager = RedisTTLManager(mock_client)
        result = manager.set_with_ttl("test:key", "test_value", key_type="scratchpad")

        assert result is True
        mock_client.setex.assert_called_once_with("test:key", 3600, "test_value")

    def test_set_with_ttl_no_client(self):
        """Test setting key without Redis client"""
        manager = RedisTTLManager()
        result = manager.set_with_ttl("test:key", "test_value")

        assert result is False

    def test_set_with_ttl_error(self):
        """Test handling error during set"""
        mock_client = Mock()
        mock_client.setex.side_effect = Exception("Connection error")

        manager = RedisTTLManager(mock_client)
        result = manager.set_with_ttl("test:key", "test_value")

        assert result is False

    def test_update_ttl_success(self):
        """Test updating TTL for existing key"""
        mock_client = Mock()
        mock_client.expire.return_value = True

        manager = RedisTTLManager(mock_client)
        result = manager.update_ttl("test:key", 7200)

        assert result is True
        mock_client.expire.assert_called_once_with("test:key", 7200)

    def test_update_ttl_no_client(self):
        """Test updating TTL without Redis client"""
        manager = RedisTTLManager()
        result = manager.update_ttl("test:key", 7200)

        assert result is False

    def test_get_ttl_success(self):
        """Test getting TTL for a key"""
        mock_client = Mock()
        mock_client.ttl.return_value = 3600

        manager = RedisTTLManager(mock_client)
        ttl = manager.get_ttl("test:key")

        assert ttl == 3600
        mock_client.ttl.assert_called_once_with("test:key")

    def test_get_ttl_no_expiry(self):
        """Test getting TTL for key with no expiry"""
        mock_client = Mock()
        mock_client.ttl.return_value = -1  # No expiry

        manager = RedisTTLManager(mock_client)
        ttl = manager.get_ttl("test:key")

        assert ttl == -1

    def test_get_ttl_key_not_exists(self):
        """Test getting TTL for non-existent key"""
        mock_client = Mock()
        mock_client.ttl.return_value = -2  # Key doesn't exist

        manager = RedisTTLManager(mock_client)
        ttl = manager.get_ttl("test:key")

        assert ttl == -2

    def test_get_ttl_no_client(self):
        """Test getting TTL without Redis client"""
        manager = RedisTTLManager()
        ttl = manager.get_ttl("test:key")

        assert ttl == -2

    def test_cleanup_expired_keys(self):
        """Test cleanup scan operation"""
        mock_client = Mock()

        # Mock scan to return keys
        mock_client.scan.side_effect = [
            (0, [b"key1", b"key2", b"key3"])  # cursor=0 means done
        ]

        # Mock TTL checks
        mock_client.ttl.side_effect = [3600, -1, 7200]  # key2 has no TTL

        manager = RedisTTLManager(mock_client)
        count = manager.cleanup_expired_keys(pattern="test:*")

        assert count == 3
        assert manager.cleanup_stats["total_cleaned"] == 3
        assert manager.cleanup_stats["last_cleanup"] is not None

    def test_cleanup_expired_keys_multiple_scans(self):
        """Test cleanup with multiple scan iterations"""
        mock_client = Mock()

        # Mock scan to return multiple batches
        mock_client.scan.side_effect = [
            (1, [b"key1", b"key2"]),  # First batch, cursor=1
            (2, [b"key3", b"key4"]),  # Second batch, cursor=2
            (0, [b"key5"])  # Final batch, cursor=0
        ]

        mock_client.ttl.return_value = 3600

        manager = RedisTTLManager(mock_client)
        count = manager.cleanup_expired_keys()

        assert count == 5
        assert mock_client.scan.call_count == 3

    def test_cleanup_expired_keys_no_client(self):
        """Test cleanup without Redis client"""
        manager = RedisTTLManager()
        count = manager.cleanup_expired_keys()

        assert count == 0

    def test_get_keys_by_ttl_range(self):
        """Test getting keys within TTL range"""
        mock_client = Mock()

        mock_client.scan.side_effect = [
            (0, [b"key1", b"key2", b"key3", b"key4"])
        ]

        # key1: 100, key2: 500, key3: 1000, key4: 5000
        mock_client.ttl.side_effect = [100, 500, 1000, 5000]

        manager = RedisTTLManager(mock_client)
        keys = manager.get_keys_by_ttl_range(200, 2000, pattern="test:*")

        # Should include key2 (500) and key3 (1000)
        assert len(keys) == 2
        assert "key2" in keys
        assert "key3" in keys

    def test_get_keys_by_ttl_range_no_client(self):
        """Test TTL range query without Redis client"""
        manager = RedisTTLManager()
        keys = manager.get_keys_by_ttl_range(0, 1000)

        assert keys == []

    def test_get_cleanup_stats(self):
        """Test getting cleanup statistics"""
        manager = RedisTTLManager()
        manager.cleanup_stats["total_cleaned"] = 100
        manager.cleanup_stats["errors"] = 2

        stats = manager.get_cleanup_stats()

        assert stats["total_cleaned"] == 100
        assert stats["errors"] == 2
        assert "last_cleanup" in stats


class TestRedisEventLog:
    """Test Redis Event Log functionality"""

    def test_init(self):
        """Test event log initialization"""
        mock_client = Mock()
        event_log = RedisEventLog(mock_client)

        assert event_log.redis_client is mock_client
        assert event_log.log_key_prefix == "event_log"
        assert event_log.max_log_size == 10000

    def test_log_event_success(self):
        """Test logging an event"""
        mock_client = Mock()
        mock_client.lpush.return_value = 1
        mock_client.ltrim.return_value = True
        mock_client.expire.return_value = True

        event_log = RedisEventLog(mock_client)
        result = event_log.log_event(
            event_type="set",
            key="test:key",
            operation="create",
            metadata={"user": "test"}
        )

        assert result is True
        assert mock_client.lpush.called
        assert mock_client.ltrim.called
        assert mock_client.expire.called

    def test_log_event_no_client(self):
        """Test logging event without Redis client"""
        event_log = RedisEventLog()
        result = event_log.log_event(
            event_type="set",
            key="test:key",
            operation="create"
        )

        assert result is False

    def test_log_event_with_metadata(self):
        """Test logging event with metadata"""
        mock_client = Mock()
        mock_client.lpush.return_value = 1
        mock_client.ltrim.return_value = True
        mock_client.expire.return_value = True

        event_log = RedisEventLog(mock_client)
        result = event_log.log_event(
            event_type="delete",
            key="test:key",
            operation="hard_delete",
            metadata={"reason": "test", "user": "admin"}
        )

        assert result is True

    def test_log_event_error(self):
        """Test handling error during event logging"""
        mock_client = Mock()
        mock_client.lpush.side_effect = Exception("Connection error")

        event_log = RedisEventLog(mock_client)
        result = event_log.log_event(
            event_type="set",
            key="test:key",
            operation="create"
        )

        assert result is False

    def test_get_recent_events(self):
        """Test retrieving recent events"""
        mock_client = Mock()

        # Mock events as JSON strings
        mock_events = [
            b'{"timestamp": "2025-10-18T12:00:00", "event_type": "set", "key": "key1"}',
            b'{"timestamp": "2025-10-18T11:59:00", "event_type": "delete", "key": "key2"}'
        ]
        mock_client.lrange.return_value = mock_events

        event_log = RedisEventLog(mock_client)
        events = event_log.get_recent_events(limit=10)

        assert len(events) == 2
        assert events[0]["event_type"] == "set"
        assert events[1]["event_type"] == "delete"
        mock_client.lrange.assert_called_once_with("event_log:events", 0, 9)

    def test_get_recent_events_no_client(self):
        """Test getting events without Redis client"""
        event_log = RedisEventLog()
        events = event_log.get_recent_events()

        assert events == []

    def test_get_recent_events_invalid_json(self):
        """Test handling invalid JSON in events"""
        mock_client = Mock()
        mock_events = [
            b'{"valid": "json"}',
            b'invalid json',
            b'{"another": "valid"}'
        ]
        mock_client.lrange.return_value = mock_events

        event_log = RedisEventLog(mock_client)
        events = event_log.get_recent_events()

        # Should skip invalid JSON
        assert len(events) == 2

    @patch('src.storage.redis_manager.datetime')
    def test_clear_old_events(self, mock_datetime):
        """Test clearing old events"""
        # Mock current time
        mock_now = datetime(2025, 10, 18, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        mock_datetime.fromisoformat.side_effect = lambda x: datetime.fromisoformat(x)

        mock_client = Mock()

        # Mock existing events
        old_event = {"timestamp": "2025-10-17T10:00:00", "key": "old"}
        recent_event = {"timestamp": "2025-10-18T11:00:00", "key": "recent"}

        with patch.object(RedisEventLog, 'get_recent_events', return_value=[recent_event, old_event]):
            event_log = RedisEventLog(mock_client)
            removed = event_log.clear_old_events(older_than_hours=2)

            # Should remove old event (more than 2 hours old)
            assert removed >= 1

    def test_clear_old_events_no_client(self):
        """Test clearing events without Redis client"""
        event_log = RedisEventLog()
        removed = event_log.clear_old_events()

        assert removed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
