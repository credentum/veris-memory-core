#!/usr/bin/env python3
"""
neo4j_readonly.py: Read-only Neo4j client for secure graph queries

This module provides a secure, read-only interface to Neo4j for the query_graph tool.
It uses separate credentials with minimal privileges to reduce security risks.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import AuthError, ServiceUnavailable

logger = logging.getLogger(__name__)

class Neo4jReadOnlyClient:
    """Read-only Neo4j client with restricted permissions"""
    
    def __init__(self, 
                 uri: str = None,
                 username: str = "veris_ro", 
                 password: str = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize read-only Neo4j client
        
        Args:
            uri: Neo4j connection URI (defaults to env/config)
            username: Read-only username (defaults to veris_ro)
            password: Read-only password (defaults to env/config)
            config: Configuration dictionary for fallback values
        """
        self.uri = uri or self._get_uri_from_config(config)
        self.username = username
        self.password = password or self._get_password_from_config(config)
        self.driver: Optional[Driver] = None
        self.is_connected = False
        
    def _get_uri_from_config(self, config: Optional[Dict[str, Any]]) -> str:
        """Get Neo4j URI from environment or config"""
        # Try environment first
        if os.getenv("NEO4J_URI"):
            return os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_URL"):
            return os.getenv("NEO4J_URL")
        
        # Fallback to config
        if config and "neo4j" in config:
            return config["neo4j"].get("uri", "bolt://neo4j:7687")
            
        # Default for Docker environment
        return "bolt://neo4j:7687"
    
    def _get_password_from_config(self, config: Optional[Dict[str, Any]]) -> str:
        """Get read-only password from environment or config"""
        # Try environment first  
        ro_password = os.getenv("NEO4J_RO_PASSWORD")
        if ro_password:
            return ro_password
            
        # Fallback to default secure password
        return "readonly_secure_2024!"
    
    def connect(self) -> bool:
        """
        Connect to Neo4j with read-only credentials
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.is_connected and self.driver:
            return True
            
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Test connection with simple read query
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()
                if test_value and test_value["test"] == 1:
                    self.is_connected = True
                    logger.info(f"✅ Connected to Neo4j as read-only user: {self.username}")
                    return True
                else:
                    raise Exception("Connection test failed")
                    
        except AuthError as e:
            logger.error(f"❌ Authentication failed for {self.username}: {e}")
            self.is_connected = False
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable: {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            self.is_connected = False
            
        return False
    
    def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute read-only Cypher query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            Exception: If not connected or query fails
        """
        if not self.is_connected or not self.driver:
            if not self.connect():
                raise Exception("Not connected to Neo4j and failed to reconnect")
        
        parameters = parameters or {}
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                
                # Convert Neo4j records to dictionaries
                records = []
                for record in result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j objects by converting to basic types
                        if hasattr(value, '__dict__'):
                            # Neo4j Node or Relationship object
                            if hasattr(value, '_properties'):
                                record_dict[key] = dict(value._properties)
                                if hasattr(value, '_labels'):
                                    record_dict[key]['_labels'] = list(value._labels)
                                if hasattr(value, '_element_id'):
                                    record_dict[key]['_id'] = value._element_id
                            else:
                                record_dict[key] = str(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                logger.info(f"Query executed successfully, returned {len(records)} records")
                return records
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def test_readonly_restrictions(self) -> Dict[str, Any]:
        """
        Test that write operations are properly blocked
        
        Returns:
            Dict with test results
        """
        if not self.is_connected:
            if not self.connect():
                return {"success": False, "error": "Not connected"}
        
        test_results = {
            "basic_read": False,
            "write_blocked": False,
            "traversal_works": False
        }
        
        try:
            # Test 1: Basic read
            result = self.query("RETURN 1 as test")
            test_results["basic_read"] = len(result) == 1 and result[0]["test"] == 1
            
            # Test 2: Graph traversal
            result = self.query("MATCH (n) RETURN COUNT(n) as count LIMIT 1")
            test_results["traversal_works"] = len(result) == 1
            
            # Test 3: Write should be blocked
            try:
                self.query("CREATE (n:TestNode {test: true})")
                test_results["write_blocked"] = False  # Should not reach here
            except Exception:
                test_results["write_blocked"] = True  # Expected behavior
            
            return {
                "success": True,
                "tests": test_results,
                "all_passed": all(test_results.values())
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            logger.info("Neo4j read-only connection closed")

# Global read-only client instance (lazy initialization)
_readonly_client: Optional[Neo4jReadOnlyClient] = None

def get_readonly_client(config: Optional[Dict[str, Any]] = None) -> Neo4jReadOnlyClient:
    """
    Get singleton read-only Neo4j client
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Neo4jReadOnlyClient instance
    """
    global _readonly_client
    
    if _readonly_client is None:
        _readonly_client = Neo4jReadOnlyClient(config=config)
    
    return _readonly_client

def test_readonly_setup():
    """Test the read-only setup"""
    client = get_readonly_client()
    
    print("🔐 Testing Neo4j Read-Only Client")
    print("=" * 40)
    
    # Test connection
    if not client.connect():
        print("❌ Failed to connect as read-only user")
        return False
    
    # Test restrictions
    test_result = client.test_readonly_restrictions()
    
    if not test_result["success"]:
        print(f"❌ Test failed: {test_result['error']}")
        return False
    
    tests = test_result["tests"]
    print(f"✅ Basic read: {'PASS' if tests['basic_read'] else 'FAIL'}")
    print(f"✅ Traversal: {'PASS' if tests['traversal_works'] else 'FAIL'}")  
    print(f"✅ Write blocked: {'PASS' if tests['write_blocked'] else 'FAIL'}")
    
    if test_result["all_passed"]:
        print("\n🎉 All read-only tests passed!")
        return True
    else:
        print("\n❌ Some tests failed")
        return False

if __name__ == "__main__":
    test_readonly_setup()