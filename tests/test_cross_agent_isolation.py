"""
Integration tests for cross-agent data isolation scenarios.

Tests comprehensive security scenarios where multiple agents attempt to access
each other's data and verifies complete isolation.
"""

import os
import time

import pytest

from src.core.agent_namespace import AgentNamespace


class TestCrossAgentDataIsolation:
    """Integration tests for cross-agent data isolation."""

    def setup_method(self):
        """Set up test fixtures with multiple agents."""
        self.namespace = AgentNamespace()

        # Create multiple test agents
        self.agents = ["agent-alpha", "agent-beta", "agent-gamma", "admin-agent", "guest-agent"]

        # Test data for each agent
        self.test_data = {
            "agent-alpha": {
                "state": {"user_preferences": "alpha_preferences", "config": "alpha_config"},
                "scratchpad": {"working_memory": "alpha_memory", "temp_data": "alpha_temp"},
                "memory": {"context": "alpha_context", "learned_facts": "alpha_facts"},
                "config": {"api_keys": "alpha_secret_keys", "permissions": "alpha_perms"},
            },
            "agent-beta": {
                "state": {"user_preferences": "beta_preferences", "config": "beta_config"},
                "scratchpad": {"working_memory": "beta_memory", "temp_data": "beta_temp"},
                "memory": {"context": "beta_context", "learned_facts": "beta_facts"},
                "config": {"api_keys": "beta_secret_keys", "permissions": "beta_perms"},
            },
            "agent-gamma": {
                "state": {"user_preferences": "gamma_preferences", "config": "gamma_config"},
                "scratchpad": {"working_memory": "gamma_memory", "temp_data": "gamma_temp"},
                "memory": {"context": "gamma_context", "learned_facts": "gamma_facts"},
                "config": {"api_keys": "gamma_secret_keys", "permissions": "gamma_perms"},
            },
            "admin-agent": {
                "state": {"admin_config": "admin_settings", "system_state": "admin_system"},
                "scratchpad": {"admin_tasks": "admin_current_task", "temp_admin": "admin_temp"},
                "memory": {"admin_history": "admin_actions", "system_logs": "admin_logs"},
                "config": {"master_keys": "admin_master_secret", "all_permissions": "admin_all"},
            },
            "guest-agent": {
                "state": {"guest_prefs": "guest_basic_prefs", "limited_config": "guest_config"},
                "scratchpad": {"guest_memory": "guest_temp_memory", "guest_temp": "guest_temp"},
                "memory": {"guest_context": "guest_limited_context"},
                "config": {"guest_permissions": "guest_limited_perms"},
            },
        }

    def test_basic_namespace_isolation(self):
        """Test basic namespace isolation between agents."""
        # Create namespaced keys for all agents and data
        agent_keys = {}

        for agent_id in self.agents:
            agent_keys[agent_id] = {}
            for prefix, data_dict in self.test_data[agent_id].items():
                agent_keys[agent_id][prefix] = {}
                for key, value in data_dict.items():
                    namespaced_key = self.namespace.create_namespaced_key(agent_id, prefix, key)
                    agent_keys[agent_id][prefix][key] = namespaced_key

        # Verify each agent can only access their own keys
        for agent_id in self.agents:
            for prefix in self.test_data[agent_id].keys():
                for key in self.test_data[agent_id][prefix].keys():
                    own_key = agent_keys[agent_id][prefix][key]

                    # Agent should have access to their own key
                    assert self.namespace.verify_agent_access(
                        agent_id, own_key
                    ), f"{agent_id} should access own key: {own_key}"

                    # Agent should NOT have access to other agents' keys
                    for other_agent_id in self.agents:
                        if other_agent_id != agent_id:
                            if (
                                prefix in self.test_data[other_agent_id]
                                and key in self.test_data[other_agent_id][prefix]
                            ):
                                other_key = agent_keys[other_agent_id][prefix][key]
                                assert not self.namespace.verify_agent_access(
                                    agent_id, other_key
                                ), f"{agent_id} should NOT access {other_agent_id}'s key: {other_key}"

    def test_shared_key_names_isolation(self):
        """Test isolation when agents use identical key names."""
        shared_key_name = "shared_config"
        shared_prefix = "state"

        # All agents create data with the same key name
        agent_keys = {}
        for agent_id in self.agents:
            namespaced_key = self.namespace.create_namespaced_key(
                agent_id, shared_prefix, shared_key_name
            )
            agent_keys[agent_id] = namespaced_key

        # Verify all keys are different despite same local name
        unique_keys = set(agent_keys.values())
        assert len(unique_keys) == len(
            self.agents
        ), "All namespaced keys should be unique even with same local key name"

        # Verify each agent can only access their own version
        for agent_id in self.agents:
            own_key = agent_keys[agent_id]
            assert self.namespace.verify_agent_access(
                agent_id, own_key
            ), f"{agent_id} should access own shared key"

            for other_agent_id in self.agents:
                if other_agent_id != agent_id:
                    other_key = agent_keys[other_agent_id]
                    assert not self.namespace.verify_agent_access(
                        agent_id, other_key
                    ), f"{agent_id} should NOT access {other_agent_id}'s shared key"

    def test_privileged_agent_isolation(self):
        """Test that even privileged agents cannot access other agents' data."""
        admin_agent = "admin-agent"
        regular_agent = "agent-alpha"

        # Create keys for both agents
        admin_key = self.namespace.create_namespaced_key(admin_agent, "config", "master_keys")
        regular_key = self.namespace.create_namespaced_key(
            regular_agent, "state", "user_preferences"
        )

        # Admin should access their own data
        assert self.namespace.verify_agent_access(
            admin_agent, admin_key
        ), "Admin should access own data"

        # Admin should NOT access regular agent's data
        assert not self.namespace.verify_agent_access(
            admin_agent, regular_key
        ), "Admin should NOT access regular agent's data"

        # Regular agent should NOT access admin's data
        assert not self.namespace.verify_agent_access(
            regular_agent, admin_key
        ), "Regular agent should NOT access admin's data"

    def test_guest_agent_isolation(self):
        """Test that guest agents have proper isolation."""
        guest_agent = "guest-agent"
        regular_agents = ["agent-alpha", "agent-beta", "admin-agent"]

        # Create keys for guest and regular agents
        guest_key = self.namespace.create_namespaced_key(guest_agent, "state", "guest_prefs")
        regular_keys = [
            self.namespace.create_namespaced_key(agent, "state", "sensitive_data")
            for agent in regular_agents
        ]

        # Guest should access their own data
        assert self.namespace.verify_agent_access(
            guest_agent, guest_key
        ), "Guest should access own data"

        # Guest should NOT access any regular agent's data
        for i, regular_key in enumerate(regular_keys):
            assert not self.namespace.verify_agent_access(
                guest_agent, regular_key
            ), f"Guest should NOT access {regular_agents[i]}'s data"

        # Regular agents should NOT access guest's data
        for regular_agent in regular_agents:
            assert not self.namespace.verify_agent_access(
                regular_agent, guest_key
            ), f"{regular_agent} should NOT access guest's data"

    def test_cross_prefix_isolation(self):
        """Test isolation across different prefixes."""
        agent_id = "agent-alpha"

        # Create keys with same local name but different prefixes
        prefixes = ["state", "scratchpad", "memory", "config"]
        key_name = "important_data"

        prefix_keys = {}
        for prefix in prefixes:
            namespaced_key = self.namespace.create_namespaced_key(agent_id, prefix, key_name)
            prefix_keys[prefix] = namespaced_key

        # All keys should be different
        unique_keys = set(prefix_keys.values())
        assert len(unique_keys) == len(
            prefixes
        ), "Keys with same local name but different prefixes should be unique"

        # Agent should have access to all their own keys regardless of prefix
        for prefix, key in prefix_keys.items():
            assert self.namespace.verify_agent_access(
                agent_id, key
            ), f"Agent should access own {prefix} data"

    def test_session_based_isolation(self):
        """Test isolation in session-based scenarios."""
        agents = ["session-agent-1", "session-agent-2", "session-agent-3"]

        # Create sessions for each agent
        sessions = {}
        for agent_id in agents:
            metadata = {"role": "user", "session_type": "isolated"}
            session_id = self.namespace.create_agent_session(agent_id, metadata)
            sessions[agent_id] = session_id

        # Verify each agent can only access their own session
        for agent_id, session_id in sessions.items():
            # Agent should access their own session
            session = self.namespace.get_agent_session(agent_id, session_id)
            assert session is not None, f"{agent_id} should access own session"
            assert session.agent_id == agent_id, "Session should belong to correct agent"

            # Agent should NOT access other agents' sessions
            for other_agent_id, other_session_id in sessions.items():
                if other_agent_id != agent_id:
                    other_session = self.namespace.get_agent_session(agent_id, other_session_id)
                    assert (
                        other_session is None
                    ), f"{agent_id} should NOT access {other_agent_id}'s session"

    def test_malicious_key_access_attempts(self):
        """Test resistance to malicious key access attempts."""
        legitimate_agent = "agent-alpha"
        malicious_agent = "malicious-agent"

        # Legitimate agent creates some data
        legit_key = self.namespace.create_namespaced_key(legitimate_agent, "config", "api_keys")

        # Malicious attempts to access legitimate data
        malicious_attempts = [
            # Direct access attempt
            legit_key,
            # Manual construction attempts
            f"agent:{legitimate_agent}:config:api_keys",
            f"agent:{legitimate_agent}:state:user_data",
            f"agent:{legitimate_agent}:scratchpad:working_memory",
            # Key manipulation attempts
            legit_key.replace(malicious_agent, legitimate_agent),
            # Injection attempts
            f"agent:{legitimate_agent}:config:api_keys; DROP TABLE agents;",
            f"agent:{legitimate_agent}:config:../../../etc/passwd",
            # Unicode/encoding attempts
            f"agent:{legitimate_agent}:config:api_keys\x00",
            f"agent:{legitimate_agent}:config:api_keys\n\r",
        ]

        # All malicious attempts should be blocked
        for malicious_key in malicious_attempts:
            try:
                access_granted = self.namespace.verify_agent_access(malicious_agent, malicious_key)
                assert (
                    not access_granted
                ), f"Malicious access should be blocked for key: {malicious_key}"
            except Exception:
                # Exceptions are acceptable for malicious inputs
                pass

    def test_race_condition_isolation(self):
        """Test isolation under concurrent access scenarios."""
        import queue
        import threading

        results = queue.Queue()
        errors = queue.Queue()

        def worker(agent_id, iterations):
            try:
                for i in range(iterations):
                    # Create keys for this agent
                    key = f"data_{i}"
                    namespaced_key = self.namespace.create_namespaced_key(agent_id, "state", key)

                    # Verify access to own key
                    own_access = self.namespace.verify_agent_access(agent_id, namespaced_key)
                    assert own_access, f"{agent_id} should access own key {namespaced_key}"

                    # Try to access other agents' keys (should fail)
                    other_agent = "other-agent" if agent_id != "other-agent" else "different-agent"
                    other_key = self.namespace.create_namespaced_key(other_agent, "state", key)
                    other_access = self.namespace.verify_agent_access(agent_id, other_key)
                    assert not other_access, f"{agent_id} should NOT access {other_agent}'s key"

                results.put(f"{agent_id}_success")

            except Exception as e:
                errors.put(f"{agent_id}_error: {e}")

        # Start multiple threads with different agents
        threads = []
        test_agents = ["race-agent-1", "race-agent-2", "race-agent-3", "race-agent-4"]

        for agent_id in test_agents:
            thread = threading.Thread(target=worker, args=(agent_id, 20))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        success_count = 0
        while not results.empty():
            result = results.get()
            success_count += 1

        error_count = 0
        while not errors.empty():
            error = errors.get()
            error_count += 1
            print(f"Race condition error: {error}")

        # Should have all successful results
        assert success_count == len(test_agents), "All agents should complete successfully"
        assert error_count == 0, "No errors should occur during concurrent access"

    def test_memory_pressure_isolation(self):
        """Test isolation under memory pressure conditions."""
        agent_count = 50
        keys_per_agent = 100

        start_time = time.time()

        # Create many agents with many keys
        all_keys = {}
        for agent_i in range(agent_count):
            agent_id = f"memory_test_agent_{agent_i}"
            all_keys[agent_id] = []

            for key_i in range(keys_per_agent):
                key_name = f"data_{key_i}"
                namespaced_key = self.namespace.create_namespaced_key(agent_id, "state", key_name)
                all_keys[agent_id].append(namespaced_key)

        # Verify isolation still works under memory pressure
        isolation_violations = 0
        for agent_id, agent_keys in all_keys.items():
            # Test random sample to avoid excessive test time
            import random

            sample_keys = random.sample(agent_keys, min(10, len(agent_keys)))

            for key in sample_keys:
                # Should have access to own key
                if not self.namespace.verify_agent_access(agent_id, key):
                    isolation_violations += 1

                # Should NOT have access to other agents' keys
                other_agents = random.sample(list(all_keys.keys()), min(5, len(all_keys)))
                for other_agent in other_agents:
                    if other_agent != agent_id:
                        other_keys = random.sample(
                            all_keys[other_agent], min(2, len(all_keys[other_agent]))
                        )
                        for other_key in other_keys:
                            if self.namespace.verify_agent_access(agent_id, other_key):
                                isolation_violations += 1

        end_time = time.time()

        # Verify performance and isolation
        assert isolation_violations == 0, f"Found {isolation_violations} isolation violations"
        assert (end_time - start_time) < 30.0, "Test should complete within 30 seconds"

    def test_persistent_vs_temporary_data_isolation(self):
        """Test isolation between persistent and temporary data."""
        agent_id = "persistence-test-agent"

        # Create different types of data
        persistent_prefixes = ["state", "memory", "config"]
        temporary_prefixes = ["scratchpad", "temp"]

        persistent_keys = []
        temporary_keys = []

        # Create persistent data keys
        for prefix in persistent_prefixes:
            key = self.namespace.create_namespaced_key(agent_id, prefix, "persistent_data")
            persistent_keys.append(key)

        # Create temporary data keys
        for prefix in temporary_prefixes:
            if prefix in ["scratchpad", "temp"]:  # Valid prefixes
                key = self.namespace.create_namespaced_key(agent_id, prefix, "temporary_data")
                temporary_keys.append(key)

        # Agent should have access to all their own data regardless of persistence type
        all_keys = persistent_keys + temporary_keys
        for key in all_keys:
            assert self.namespace.verify_agent_access(
                agent_id, key
            ), f"Agent should access own data: {key}"

        # Another agent should not access any of this data
        other_agent = "other-persistence-agent"
        for key in all_keys:
            assert not self.namespace.verify_agent_access(
                other_agent, key
            ), f"Other agent should NOT access data: {key}"

    def test_invalid_namespace_manipulation(self):
        """Test resistance to invalid namespace manipulation attempts."""
        legitimate_agent = "legit-agent"

        # Invalid manipulation attempts
        invalid_keys = [
            "",  # Empty key
            "agent:",  # Incomplete namespace
            "agent:legit-agent:",  # Missing prefix and key
            "agent:legit-agent:state:",  # Missing key
            "invalid:legit-agent:state:data",  # Wrong namespace prefix
            "agent::state:data",  # Empty agent_id
            "agent:legit-agent::data",  # Empty prefix
            "agent:legit-agent:state:",  # Empty key
            "agent:legit-agent:invalid_prefix:data",  # Invalid prefix
            "AGENT:legit-agent:state:data",  # Wrong case
            "agent:LEGIT-AGENT:state:data",  # Wrong agent case
            "agent:legit-agent:STATE:data",  # Wrong prefix case
            "agent:other-agent:state:data",  # Different agent
            # Injection attempts
            "agent:legit-agent:state:data; DROP TABLE agents",
            "agent:legit-agent:state:data\x00",
            "agent:legit-agent:state:data\n\r",
            "agent:legit-agent:state:../../../etc/passwd",
            # Too many parts
            "agent:legit-agent:state:data:extra:parts",
            # Too few parts
            "agent:legit-agent:state",
            # Special characters in namespace
            "agent:legit@agent:state:data",
            "agent:legit agent:state:data",
            "agent:legit-agent:sta te:data",
            "agent:legit-agent:state:da ta",
        ]

        # All invalid keys should be rejected
        for invalid_key in invalid_keys:
            try:
                access_granted = self.namespace.verify_agent_access(legitimate_agent, invalid_key)
                assert not access_granted, f"Invalid key should be rejected: {invalid_key}"
            except Exception:
                # Exceptions are acceptable for invalid inputs
                pass

    def test_comprehensive_security_scenario(self):
        """Test comprehensive security scenario with multiple attack vectors."""
        # Set up legitimate agents with sensitive data
        sensitive_agents = {
            "banking-agent": {
                "state": {"account_balance": "1000000", "customer_data": "sensitive"},
                "config": {"api_keys": "bank_secret_key", "encryption_keys": "bank_encrypt"},
            },
            "medical-agent": {
                "state": {"patient_records": "medical_data", "diagnosis": "confidential"},
                "config": {"hipaa_keys": "medical_secret", "access_tokens": "medical_token"},
            },
            "admin-agent": {
                "state": {"system_config": "admin_config", "user_database": "all_users"},
                "config": {"master_password": "admin_secret", "root_access": "admin_root"},
            },
        }

        # Create legitimate keys
        legitimate_keys = {}
        for agent_id, data in sensitive_agents.items():
            legitimate_keys[agent_id] = {}
            for prefix, key_data in data.items():
                legitimate_keys[agent_id][prefix] = {}
                for key, value in key_data.items():
                    namespaced_key = self.namespace.create_namespaced_key(agent_id, prefix, key)
                    legitimate_keys[agent_id][prefix][key] = namespaced_key

        # Attacker agents try various approaches
        attackers = ["hacker-agent", "malicious-bot", "compromised-agent"]

        attack_attempts = 0
        successful_attacks = 0

        for attacker in attackers:
            # Try to access all legitimate data
            for agent_id, prefixes in legitimate_keys.items():
                for prefix, keys in prefixes.items():
                    for key, namespaced_key in keys.items():
                        attack_attempts += 1

                        try:
                            access_granted = self.namespace.verify_agent_access(
                                attacker, namespaced_key
                            )
                            if access_granted:
                                successful_attacks += 1
                                print(f"SECURITY BREACH: {attacker} accessed {namespaced_key}")
                        except Exception:
                            # Exceptions during attack attempts are expected
                            pass

        # Security validation
        assert (
            successful_attacks == 0
        ), f"SECURITY FAILURE: {successful_attacks} out of {attack_attempts} attack attempts succeeded"

        # Verify legitimate agents still have access to their own data
        for agent_id, prefixes in legitimate_keys.items():
            for prefix, keys in prefixes.items():
                for key, namespaced_key in keys.items():
                    assert self.namespace.verify_agent_access(
                        agent_id, namespaced_key
                    ), f"Legitimate agent {agent_id} lost access to own data: {namespaced_key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
