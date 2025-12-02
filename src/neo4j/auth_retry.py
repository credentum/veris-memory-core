#!/usr/bin/env python3
"""
Neo4j authentication with exponential backoff retry logic.
Handles transient failures and connection issues gracefully.
"""

import time
import random
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import (
    ServiceUnavailable,
    AuthError,
    TransientError,
    SessionExpired,
    Neo4jError
)


logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_auth_error: bool = False
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay = delay * (0.5 + random.random())
        
        return delay


class Neo4jRetryClient:
    """Neo4j client with automatic retry and reconnection logic."""
    
    def __init__(
        self,
        uri: str,
        auth: tuple,
        retry_config: Optional[RetryConfig] = None,
        **driver_config
    ):
        """
        Initialize Neo4j client with retry capabilities.
        
        Args:
            uri: Neo4j connection URI
            auth: Authentication tuple (username, password)
            retry_config: Retry configuration
            **driver_config: Additional driver configuration
        """
        self.uri = uri
        self.auth = auth
        self.retry_config = retry_config or RetryConfig()
        self.driver_config = driver_config
        self._driver: Optional[Driver] = None
        self._connection_attempts = 0
        self._last_connection_error: Optional[Exception] = None
        
        # Set reasonable defaults for driver config
        self.driver_config.setdefault('max_connection_lifetime', 3600)
        self.driver_config.setdefault('max_connection_pool_size', 50)
        self.driver_config.setdefault('connection_acquisition_timeout', 60)
        self.driver_config.setdefault('keep_alive', True)
        
    def _create_driver(self) -> Driver:
        """Create a new driver instance."""
        return GraphDatabase.driver(
            self.uri,
            auth=self.auth,
            **self.driver_config
        )
    
    def _ensure_driver(self) -> Driver:
        """Ensure driver exists and is connected."""
        if self._driver is None:
            self._driver = self._create_driver()
        
        # Verify connectivity
        try:
            self._driver.verify_connectivity()
        except (ServiceUnavailable, SessionExpired) as e:
            logger.warning(f"Driver connectivity check failed: {e}")
            # Recreate driver
            if self._driver:
                self._driver.close()
            self._driver = self._create_driver()
            self._driver.verify_connectivity()
        
        return self._driver
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.retry_config.max_retries:
            return False
        
        # Transient errors should always be retried
        if isinstance(error, (TransientError, ServiceUnavailable, SessionExpired)):
            return True
        
        # Auth errors only if configured
        if isinstance(error, AuthError):
            return self.retry_config.retry_on_auth_error
        
        # Generic Neo4j errors might be retriable
        if isinstance(error, Neo4jError):
            # Check error code for known transient conditions
            code = getattr(error, 'code', '')
            transient_codes = [
                'Neo.TransientError',
                'Neo.ClientError.Transaction.Terminated',
                'Neo.ClientError.Transaction.LockClientStopped'
            ]
            return any(code.startswith(tc) for tc in transient_codes)
        
        return False
    
    def execute_with_retry(
        self,
        work_fn: Callable[[Session], Any],
        database: Optional[str] = None
    ) -> Any:
        """
        Execute work function with automatic retry on failure.
        
        Args:
            work_fn: Function that takes a session and performs work
            database: Optional database name
            
        Returns:
            Result from work function
            
        Raises:
            Exception: After all retries exhausted
        """
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                driver = self._ensure_driver()
                
                with driver.session(database=database) as session:
                    result = work_fn(session)
                    
                    # Reset connection attempts on success
                    self._connection_attempts = 0
                    self._last_connection_error = None
                    
                    return result
                    
            except Exception as e:
                last_error = e
                self._last_connection_error = e
                self._connection_attempts += 1
                
                if not self._should_retry(e, attempt):
                    logger.error(
                        f"Neo4j operation failed (non-retriable): {e}",
                        extra={
                            'attempt': attempt + 1,
                            'error_type': type(e).__name__
                        }
                    )
                    raise
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.calculate_delay(attempt)
                    logger.warning(
                        f"Neo4j operation failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}",
                        extra={
                            'attempt': attempt + 1,
                            'delay': delay,
                            'error_type': type(e).__name__
                        }
                    )
                    time.sleep(delay)
                    
                    # Reset driver on certain errors
                    if isinstance(e, (ServiceUnavailable, SessionExpired)):
                        if self._driver:
                            try:
                                self._driver.close()
                            except:
                                pass
                        self._driver = None
        
        logger.error(
            f"Neo4j operation failed after {self.retry_config.max_retries + 1} attempts",
            extra={
                'last_error': str(last_error),
                'error_type': type(last_error).__name__ if last_error else None
            }
        )
        
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Operation failed with no error captured")
    
    def read_with_retry(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> list:
        """
        Execute read query with retry logic.
        
        Args:
            query: Cypher query to execute
            parameters: Query parameters
            database: Optional database name
            
        Returns:
            List of records
        """
        def work(session: Session):
            result = session.run(query, parameters or {})
            return list(result)
        
        return self.execute_with_retry(work, database)
    
    def write_with_retry(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> Any:
        """
        Execute write query with retry logic.
        
        Args:
            query: Cypher query to execute
            parameters: Query parameters
            database: Optional database name
            
        Returns:
            Query result summary
        """
        def work(session: Session):
            with session.begin_transaction() as tx:
                result = tx.run(query, parameters or {})
                records = list(result)  # Consume result
                tx.commit()
                return result.consume()
        
        return self.execute_with_retry(work, database)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check with retry.
        
        Returns:
            Health status dictionary
        """
        start_time = time.time()
        
        try:
            # Simple query to check connectivity
            result = self.read_with_retry("RETURN 1 as health")
            
            return {
                'status': 'healthy',
                'latency_ms': (time.time() - start_time) * 1000,
                'connection_attempts': self._connection_attempts,
                'result': result[0]['health'] if result else None
            }
            
        except AuthError as e:
            return {
                'status': 'auth_error',
                'latency_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'connection_attempts': self._connection_attempts
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'latency_ms': (time.time() - start_time) * 1000,
                'error': str(e),
                'error_type': type(e).__name__,
                'connection_attempts': self._connection_attempts
            }
    
    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class Neo4jConnectionPool:
    """Connection pool manager with health monitoring."""
    
    def __init__(
        self,
        uri: str,
        auth: tuple,
        pool_size: int = 5,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize connection pool.
        
        Args:
            uri: Neo4j connection URI
            auth: Authentication tuple
            pool_size: Number of clients in pool
            retry_config: Retry configuration
        """
        self.uri = uri
        self.auth = auth
        self.pool_size = pool_size
        self.retry_config = retry_config or RetryConfig()
        
        # Create pool of clients
        self.clients = [
            Neo4jRetryClient(uri, auth, retry_config)
            for _ in range(pool_size)
        ]
        
        self._current_index = 0
        self._healthy_clients = set(range(pool_size))
        
    def get_client(self) -> Neo4jRetryClient:
        """
        Get next available healthy client.
        
        Returns:
            Neo4jRetryClient instance
            
        Raises:
            RuntimeError: If no healthy clients available
        """
        if not self._healthy_clients:
            raise RuntimeError("No healthy Neo4j clients available")
        
        # Round-robin through healthy clients
        healthy_list = sorted(self._healthy_clients)
        index = healthy_list[self._current_index % len(healthy_list)]
        self._current_index += 1
        
        return self.clients[index]
    
    def mark_unhealthy(self, client: Neo4jRetryClient):
        """Mark a client as unhealthy."""
        try:
            index = self.clients.index(client)
            self._healthy_clients.discard(index)
            logger.warning(f"Marked Neo4j client {index} as unhealthy")
        except ValueError:
            pass
    
    def health_check_all(self) -> Dict[str, Any]:
        """
        Check health of all clients in pool.
        
        Returns:
            Pool health status
        """
        results = []
        newly_healthy = set()
        
        for i, client in enumerate(self.clients):
            health = client.health_check()
            results.append({
                'client_id': i,
                **health
            })
            
            if health['status'] == 'healthy':
                newly_healthy.add(i)
        
        # Update healthy clients set
        self._healthy_clients = newly_healthy
        
        healthy_count = len(self._healthy_clients)
        total_count = len(self.clients)
        
        return {
            'pool_status': 'healthy' if healthy_count > 0 else 'unhealthy',
            'healthy_clients': healthy_count,
            'total_clients': total_count,
            'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0,
            'clients': results
        }
    
    def close_all(self):
        """Close all client connections."""
        for client in self.clients:
            client.close()


# Example usage and testing
def main():
    """Example usage of Neo4j retry client."""
    import os
    
    # Configuration from environment
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')
    
    # Create retry configuration
    retry_config = RetryConfig(
        max_retries=5,
        initial_delay=1.0,
        max_delay=30.0,
        jitter=True
    )
    
    # Create client
    with Neo4jRetryClient(uri, (user, password), retry_config) as client:
        # Health check
        print("Performing health check...")
        health = client.health_check()
        print(f"Health status: {health}")
        
        # Example read query
        try:
            result = client.read_with_retry(
                "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            )
            print(f"Node count: {result[0]['node_count'] if result else 0}")
        except Exception as e:
            print(f"Query failed: {e}")
        
        # Example write query with transaction
        try:
            client.write_with_retry(
                "CREATE (n:TestNode {id: $id, created: timestamp()}) RETURN n",
                {'id': 'test-123'}
            )
            print("Test node created successfully")
        except Exception as e:
            print(f"Write failed: {e}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()