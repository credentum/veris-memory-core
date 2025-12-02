"""
Complex nested query tests for CypherValidator.

Tests complex nested MATCH patterns, complex WHERE clauses, and edge cases
to achieve comprehensive coverage of the CypherValidator.
"""

import pytest

from src.security.cypher_validator import CypherValidator, ValidationResult


class TestCypherValidatorComplex:
    """Test complex nested queries and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CypherValidator()
        self.strict_validator = CypherValidator(max_complexity=50, max_length=1000)

    def test_complex_nested_match_patterns(self):
        """Test complex nested MATCH patterns."""
        # Valid complex nested queries
        valid_complex_queries = [
            # Multiple MATCH with nested patterns
            """
            MATCH (a:Context)-[:RELATES_TO]->(b:Context)
            MATCH (b)-[:CONTAINS]->(c:Data)
            WHERE a.type = 'document' AND b.status = 'active'
            RETURN a.id, b.title, c.content
            """,
            # Nested relationship patterns
            """
            MATCH (user:User)-[:HAS_CONTEXT]->(ctx:Context)-[:CONTAINS]->(data:Data)
            WHERE user.agent_id = $agent_id AND ctx.active = true
            RETURN user.name, ctx.title, data.value
            ORDER BY ctx.created_at DESC
            LIMIT 50
            """,
            # Complex path patterns
            """
            MATCH path = (start:Context)-[:RELATES_TO*1..3]->(end:Context)
            WHERE start.type = 'root' AND end.type = 'leaf'
            RETURN path, LENGTH(path) as path_length
            ORDER BY path_length
            """,
            # Multiple relationship directions
            """
            MATCH (center:Context)<-[:BELONGS_TO]-(left:Data)-[:CONNECTS_TO]->(right:Data)-[:BELONGS_TO]->(center)  # noqa: E501
            WHERE center.category = $category
            RETURN center.id, left.value, right.value
            """,
            # Nested aggregation with complex grouping
            """
            MATCH (ctx:Context)-[:HAS_TAG]->(tag:Tag)
            WITH ctx, COLLECT(tag.name) as tags
            WHERE SIZE(tags) >= 3
            RETURN ctx.id, tags, SIZE(tags) as tag_count
            ORDER BY tag_count DESC
            """,
            # Complex CASE expressions
            """
            MATCH (n:Context)
            RETURN n.id,
                   CASE
                     WHEN n.priority = 'high' THEN 3
                     WHEN n.priority = 'medium' THEN 2
                     ELSE 1
                   END as priority_score,
                   CASE
                     WHEN n.status = 'active' AND n.priority = 'high' THEN 'urgent'
                     WHEN n.status = 'active' THEN 'normal'
                     ELSE 'inactive'
                   END as urgency
            """,
            # Nested EXISTS with complex conditions
            """
            MATCH (ctx:Context)
            WHERE EXISTS {
                MATCH (ctx)-[:RELATES_TO]->(related:Context)
                WHERE related.type = 'reference' AND related.active = true
            }
            AND EXISTS {
                MATCH (ctx)-[:HAS_DATA]->(data:Data)
                WHERE data.validated = true
            }
            RETURN ctx.id, ctx.title
            """,
            # Complex UNION patterns
            """
            MATCH (a:Context) WHERE a.type = 'primary'
            RETURN a.id, a.title, 'primary' as source
            UNION
            MATCH (b:Context) WHERE b.type = 'secondary' AND b.priority = 'high'
            RETURN b.id, b.title, 'secondary' as source
            ORDER BY source, title
            """,
            # Nested UNWIND with complex processing
            """
            MATCH (ctx:Context)
            WITH ctx, ctx.tags as tag_list
            UNWIND tag_list as tag
            WITH ctx, tag, SIZE(tag_list) as total_tags
            WHERE total_tags > 2 AND tag IS NOT NULL
            RETURN ctx.id, COLLECT(DISTINCT tag) as processed_tags
            """,
            # Complex pattern with multiple WITH clauses
            """
            MATCH (user:User)-[:HAS_SESSION]->(session:Session)
            WITH user, session, session.created_at as session_start
            MATCH (session)-[:CONTAINS]->(ctx:Context)
            WITH user, session, ctx, COUNT(ctx) as context_count
            WHERE context_count > 5
            MATCH (ctx)-[:HAS_DATA]->(data:Data)
            RETURN user.id, session.id, context_count, COUNT(data) as data_count
            """,
        ]

        for query in valid_complex_queries:
            result = self.validator.validate_query(query.strip())
            assert result.is_valid, f"Complex query should be valid: {query[:100]}..."
            # Complex queries should have higher complexity scores
            assert result.complexity_score >= 10, "Complex query should have high complexity score"

    def test_complex_where_clauses(self):
        """Test complex WHERE clauses with nested conditions."""
        complex_where_queries = [
            # Complex boolean logic
            """
            MATCH (n:Context)
            WHERE (n.type = 'document' AND n.status = 'active')
                  OR (n.type = 'reference' AND n.priority = 'high' AND n.validated = true)
                  OR (n.type = 'temporary' AND n.created_at > $recent_date)
            RETURN n.id, n.type, n.status
            """,
            # Nested function calls in WHERE
            """
            MATCH (ctx:Context)-[:HAS_DATA]->(data:Data)
            WHERE SIZE(ctx.tags) > 2
                  AND LENGTH(data.content) > 100
                  AND data.content =~ '.*important.*'
                  AND ctx.created_at > datetime() - duration('P7D')
            RETURN ctx.id, data.id
            """,
            # Complex pattern matching in WHERE
            """
            MATCH (n:Context)
            WHERE n.title =~ '(?i).*analysis.*'
                  AND ANY(tag IN n.tags WHERE tag STARTS WITH 'data_')
                  AND ALL(field IN n.required_fields WHERE field IS NOT NULL)
                  AND NONE(error IN n.validation_errors WHERE error.severity = 'critical')
            RETURN n.id, n.title
            """,
            # Nested subqueries in WHERE
            """
            MATCH (ctx:Context)
            WHERE ctx.id IN [id IN $context_ids WHERE id IS NOT NULL]
                  AND ctx.type IN ['document', 'reference', 'analysis']
                  AND NOT ctx.archived = true
            RETURN ctx.id, ctx.type
            """,
            # Complex range and comparison operations
            """
            MATCH (n:Context)
            WHERE n.priority IN ['high', 'critical']
                  AND n.score >= 0.8
                  AND n.confidence BETWEEN 0.7 AND 1.0
                  AND n.last_updated > datetime() - duration('P1D')
                  AND SIZE(n.dependencies) BETWEEN 1 AND 10
            RETURN n.id, n.priority, n.score
            """,
            # Complex pattern with IS NULL and EXISTS
            """
            MATCH (ctx:Context)
            WHERE ctx.deleted_at IS NULL
                  AND ctx.parent_id IS NOT NULL
                  AND EXISTS(ctx.metadata.version)
                  AND NOT EXISTS(ctx.errors)
            RETURN ctx.id, ctx.metadata
            """,
            # Nested CASE in WHERE
            """
            MATCH (n:Context)
            WHERE CASE n.type
                    WHEN 'critical' THEN n.priority = 'high'
                    WHEN 'important' THEN n.priority IN ['high', 'medium']
                    ELSE n.priority IS NOT NULL
                  END
                  AND n.status = 'active'
            RETURN n.id, n.type, n.priority
            """,
        ]

        for query in complex_where_queries:
            result = self.validator.validate_query(query.strip())
            assert result.is_valid, f"Complex WHERE query should be valid: {query[:100]}..."

    def test_complex_aggregation_patterns(self):
        """Test complex aggregation and grouping patterns."""
        aggregation_queries = [
            # Complex aggregation with multiple GROUP BY
            """
            MATCH (ctx:Context)-[:HAS_TAG]->(tag:Tag)
            RETURN ctx.type, tag.category,
                   COUNT(DISTINCT ctx) as context_count,
                   COUNT(DISTINCT tag) as tag_count,
                   COLLECT(DISTINCT ctx.id) as context_ids,
                   AVG(ctx.score) as avg_score,
                   MAX(ctx.created_at) as latest_created
            """,
            # Nested aggregation with complex expressions
            """
            MATCH (user:User)-[:HAS_CONTEXT]->(ctx:Context)
            WITH user,
                 COUNT(ctx) as total_contexts,
                 SUM(ctx.size) as total_size,
                 COLLECT(ctx.type) as context_types
            WHERE total_contexts > 5
            RETURN user.id,
                   total_contexts,
                   total_size,
                   SIZE(COLLECT(DISTINCT context_types)) as unique_types
            """,
            # Complex aggregation with percentiles and statistics
            """
            MATCH (ctx:Context)
            WHERE ctx.created_at > $start_date
            RETURN ctx.type,
                   COUNT(*) as total,
                   MIN(ctx.score) as min_score,
                   MAX(ctx.score) as max_score,
                   AVG(ctx.score) as avg_score,
                   percentileDisc(ctx.score, 0.5) as median_score,
                   percentileDisc(ctx.score, 0.95) as p95_score,
                   stDev(ctx.score) as score_std_dev
            """,
            # Aggregation with complex COLLECT operations
            """
            MATCH (ctx:Context)-[:RELATES_TO]->(related:Context)
            WITH ctx,
                 COLLECT({
                   id: related.id,
                   type: related.type,
                   score: related.score
                 }) as related_contexts
            WHERE SIZE(related_contexts) > 2
            RETURN ctx.id,
                   SIZE(related_contexts) as relation_count,
                   [r IN related_contexts WHERE r.score > 0.8] as high_score_relations
            """,
            # Complex aggregation with string operations
            """
            MATCH (ctx:Context)
            RETURN ctx.category,
                   COUNT(*) as total,
                   COLLECT(DISTINCT ctx.type) as types,
                   reduce(s = '', tag IN COLLECT(ctx.primary_tag) | s + tag + ', ') as all_tags,
                   SIZE(COLLECT(DISTINCT substring(ctx.title, 0, 10))) as title_prefixes
            """,
        ]

        for query in aggregation_queries:
            result = self.validator.validate_query(query.strip())
            assert result.is_valid, f"Complex aggregation query should be valid: {query[:100]}..."

    def test_forbidden_operations_in_complex_queries(self):
        """Test detection of forbidden operations in complex nested queries."""
        forbidden_complex_queries = [
            # Hidden CREATE in complex query
            """
            MATCH (ctx:Context)
            WHERE ctx.type = 'document'
            WITH ctx
            CREATE (new:Context {title: 'malicious'})
            RETURN ctx.id
            """,
            # Hidden DELETE in subquery
            """
            MATCH (ctx:Context)
            WHERE EXISTS {
                MATCH (temp:TempData)
                DELETE temp
                RETURN temp
            }
            RETURN ctx.id
            """,
            # Hidden SET in complex expression
            """
            MATCH (ctx:Context)
            WITH ctx,
                 CASE WHEN ctx.type = 'test'
                      THEN SET ctx.modified = true
                      ELSE ctx.id
                 END as result
            RETURN result
            """,
            # Hidden MERGE in nested pattern
            """
            MATCH (a:Context)-[:RELATES_TO]->(b:Context)
            WHERE a.type = 'source'
            FOREACH (item IN b.items |
                MERGE (new:Item {id: item.id})
            )
            RETURN a.id
            """,
            # Hidden procedure CALL
            """
            MATCH (ctx:Context)
            WHERE ctx.active = true
            CALL db.clearQueryCaches()
            RETURN ctx.id
            """,
            # Nested REMOVE operation
            """
            MATCH (ctx:Context)
            WHERE SIZE(ctx.tags) > 5
            WITH ctx
            REMOVE ctx.sensitive_data
            RETURN ctx.id
            """,
        ]

        for query in forbidden_complex_queries:
            result = self.validator.validate_query(query.strip())
            assert (
                not result.is_valid
            ), f"Query with forbidden operation should be invalid: {query[:100]}..."
            assert result.error_type == "forbidden_operation"

    def test_complexity_score_calculation(self):
        """Test complexity score calculation for various query patterns."""
        complexity_test_cases = [
            # Simple query - low complexity
            ("MATCH (n:Context) RETURN n.id", 10),  # 1 MATCH = 10 points
            # Query with WHERE - medium complexity
            ("MATCH (n:Context) WHERE n.type = 'doc' RETURN n.id", 15),  # MATCH(10) + WHERE(5)
            # Query with relationship - higher complexity
            (
                "MATCH (a:Context)-[:RELATES_TO]->(b:Context) RETURN a.id, b.id",
                15,
            ),  # MATCH(10) + relationship(5)
            # Complex query with multiple clauses
            (
                """
            MATCH (a:Context)-[:RELATES_TO]->(b:Context)
            WHERE a.type = 'source' AND b.type = 'target'
            WITH a, b, COUNT(*) as count
            MATCH (b)-[:CONTAINS]->(c:Data)
            RETURN a.id, b.id, c.value
            ORDER BY count DESC
            """,
                45,
            ),  # Should be high complexity
            # Complex query with moderate nesting - should be valid
            (
                """
            MATCH (user:User)-[:HAS_SESSION]->(session:Session)
            WHERE user.active = true
            WITH user, session
            MATCH (session)-[:CONTAINS]->(ctx:Context)
            RETURN user.id, session.id, ctx.id
            """,
                40,
            ),  # Moderate complexity
        ]

        for query, expected_min_complexity in complexity_test_cases:
            result = self.validator.validate_query(query.strip())
            assert result.is_valid, f"Query should be valid: {query[:50]}..."
            assert (
                result.complexity_score >= expected_min_complexity
            ), f"Complexity score {result.complexity_score} should be >= {expected_min_complexity}"

    def test_complexity_limit_enforcement(self):
        """Test that complexity limits are properly enforced."""
        # Create a very complex query that exceeds limits
        very_complex_query = """
        MATCH (a:Context)-[:RELATES_TO]->(b:Context)-[:CONTAINS]->(c:Data)
        WHERE a.type = 'source' AND b.type = 'intermediate' AND c.validated = true
        WITH a, b, c, COUNT(*) as count1
        MATCH (c)-[:CONNECTS_TO]->(d:Data)-[:BELONGS_TO]->(e:Context)
        WHERE d.active = true AND e.category = 'target'
        WITH a, b, c, d, e, count1, COUNT(*) as count2
        MATCH (e)-[:HAS_METADATA]->(meta:Metadata)-[:REFERENCES]->(ref:Reference)
        WHERE meta.version > 1 AND ref.status = 'current'
        WITH a, b, c, d, e, meta, ref, count1, count2, COUNT(*) as count3
        MATCH (ref)-[:LINKS_TO]->(final:Context)
        WHERE final.priority = 'high'
        RETURN a.id, b.id, c.id, d.id, e.id, meta.id, ref.id, final.id, count1, count2, count3
        ORDER BY count1 DESC, count2 DESC, count3 DESC
        UNION
        MATCH (x:Context) WHERE x.type = 'backup'
        RETURN x.id, null, null, null, null, null, null, null, 0, 0, 0
        """

        # Test with strict validator (low complexity limit)
        strict_result = self.strict_validator.validate_query(very_complex_query)
        assert not strict_result.is_valid
        assert strict_result.error_type == "complexity_too_high"
        assert strict_result.complexity_score > self.strict_validator.max_complexity

        # Test with regular validator (higher limit)
        self.validator.validate_query(very_complex_query)  # Just validate, result may vary
        # This might pass or fail depending on actual complexity

    def test_parameter_injection_in_complex_queries(self):
        """Test parameter injection detection in complex queries."""
        # Test legitimate parameters
        safe_query = """
        MATCH (ctx:Context)-[:RELATES_TO]->(related:Context)
        WHERE ctx.agent_id = $agent_id
              AND ctx.type = $context_type
              AND related.category IN $allowed_categories
              AND ctx.created_at > $start_date
        RETURN ctx.id, related.id, ctx.metadata
        ORDER BY ctx.created_at DESC
        LIMIT $max_results
        """

        safe_params = {
            "agent_id": "agent-123",
            "context_type": "document",
            "allowed_categories": ["public", "shared"],
            "start_date": "2024-01-01T00:00:00Z",
            "max_results": 50,
        }

        result = self.validator.validate_query(safe_query, safe_params)
        assert result.is_valid

        # Test malicious parameters
        malicious_params = {
            "agent_id": "agent-123'; DROP TABLE contexts; --",
            "context_type": "document",
            "allowed_categories": ["public"],
        }

        result = self.validator.validate_query(safe_query, malicious_params)
        assert not result.is_valid
        assert result.error_type == "suspicious_parameter"

        # Test parameters with CREATE injection
        create_injection_params = {
            "agent_id": "agent-123",
            "context_type": "CREATE (malicious:Hack)",
            "allowed_categories": ["public"],
        }

        result = self.validator.validate_query(safe_query, create_injection_params)
        assert not result.is_valid
        assert result.error_type == "suspicious_parameter"

    def test_nested_pattern_validation(self):
        """Test validation of deeply nested patterns."""
        # Test valid nested patterns
        valid_nested_queries = [
            # Triple nested relationships
            """
            MATCH (a:User)-[:HAS_SESSION]->(s:Session)-[:CONTAINS]->(c:Context)-[:HAS_DATA]->(d:Data)  # noqa: E501
            WHERE a.agent_id = $agent_id
            RETURN a.id, s.id, c.id, d.value
            """,
            # Nested patterns with optional matches
            """
            MATCH (ctx:Context)
            OPTIONAL MATCH (ctx)-[:HAS_METADATA]->(meta:Metadata)
            OPTIONAL MATCH (meta)-[:REFERENCES]->(ref:Reference)
            WHERE ctx.type = $type
            RETURN ctx.id, meta.version, ref.url
            """,
            # Complex path patterns with variable length
            """
            MATCH path = (start:Context)-[:RELATES_TO*2..4]->(end:Context)
            WHERE start.type = 'root' AND end.type = 'leaf'
            AND ALL(node IN nodes(path) WHERE node.active = true)
            RETURN path, LENGTH(path)
            """,
        ]

        for query in valid_nested_queries:
            result = self.validator.validate_query(query.strip())
            assert result.is_valid, f"Nested pattern query should be valid: {query[:100]}..."

    def test_security_edge_cases(self):
        """Test security validation edge cases."""
        # Test query length limits
        very_long_query = (
            "MATCH (n:Context) WHERE "
            + " AND ".join([f"n.prop{i} = 'value{i}'" for i in range(500)])
            + " RETURN n.id"
        )

        result = self.strict_validator.validate_query(very_long_query)
        assert not result.is_valid
        assert result.error_type == "query_too_long"

        # Test large LIMIT values
        large_limit_query = "MATCH (n:Context) RETURN n.id LIMIT 50000"
        result = self.validator.validate_query(large_limit_query)
        assert not result.is_valid
        assert result.error_type == "limit_too_large"

        # Test potential injection patterns
        injection_queries = [
            "MATCH (n:Context); DROP TABLE users; RETURN n.id",
            "MATCH (n:Context) /* malicious comment */ RETURN n.id",
            "MATCH (n:Context) -- comment RETURN n.id",
            "MATCH (n:Context) RETURN n.id; EXEC('malicious')",
        ]

        for query in injection_queries:
            result = self.validator.validate_query(query)
            assert not result.is_valid
            assert result.error_type in ["potential_injection", "forbidden_operation"]

    def test_performance_with_complex_queries(self):
        """Test validator performance with complex queries."""
        import time

        complex_queries = [
            """
            MATCH (a:Context)-[:RELATES_TO*1..3]->(b:Context)
            WHERE a.type = 'source' AND b.type = 'target'
            WITH a, b, COUNT(*) as relation_count
            MATCH (b)-[:CONTAINS]->(data:Data)
            WHERE data.validated = true
            RETURN a.id, b.id, relation_count, COUNT(data) as data_count
            ORDER BY relation_count DESC, data_count DESC
            LIMIT 100
            """
            for _ in range(50)  # 50 complex queries
        ]

        start_time = time.time()
        for query in complex_queries:
            result = self.validator.validate_query(query)
            assert isinstance(result, ValidationResult)
        end_time = time.time()

        # Should validate 50 complex queries in under 5 seconds
        total_time = end_time - start_time
        assert total_time < 5.0, f"Validation took too long: {total_time} seconds"

        # Average should be under 100ms per query
        avg_time = total_time / len(complex_queries)
        assert avg_time < 0.1, f"Average validation time too slow: {avg_time} seconds"

    def test_comprehensive_query_patterns(self):
        """Test comprehensive validation of various query patterns."""
        # Test all major Cypher constructs that should be allowed
        comprehensive_queries = [
            # Advanced WHERE patterns
            """
            MATCH (ctx:Context)
            WHERE ctx.type IN ['doc', 'ref']
                  AND ctx.score > 0.5
                  AND ctx.title =~ '.*analysis.*'
                  AND SIZE(ctx.tags) BETWEEN 1 AND 10
                  AND ctx.created_at < datetime()
            RETURN ctx.id, ctx.title
            """,
            # Complex CASE and conditional logic
            """
            MATCH (n:Context)
            RETURN n.id,
                   CASE n.priority
                     WHEN 'high' THEN 'urgent'
                     WHEN 'medium' THEN 'normal'
                     ELSE 'low'
                   END as urgency_level,
                   CASE
                     WHEN n.score > 0.8 THEN 'excellent'
                     WHEN n.score > 0.6 THEN 'good'
                     WHEN n.score > 0.4 THEN 'fair'
                     ELSE 'poor'
                   END as quality_rating
            """,
            # Advanced string and list operations
            """
            MATCH (ctx:Context)
            WHERE ANY(tag IN ctx.tags WHERE tag STARTS WITH 'data_')
                  AND ALL(req IN ctx.requirements WHERE req IS NOT NULL)
                  AND NONE(err IN ctx.errors WHERE err.level = 'critical')
                  AND SINGLE(primary IN ctx.flags WHERE primary = 'primary')
            RETURN ctx.id, ctx.tags, ctx.requirements
            """,
            # Complex mathematical operations
            """
            MATCH (ctx:Context)
            RETURN ctx.id,
                   abs(ctx.score - 0.5) as score_deviation,
                   round(ctx.confidence * 100) as confidence_percent,
                   ceil(ctx.processing_time) as time_ceiling,
                   floor(ctx.size / 1024) as size_kb,
                   sqrt(ctx.complexity) as complexity_sqrt
            """,
            # Advanced temporal operations
            """
            MATCH (ctx:Context)
            WHERE ctx.created_at > datetime() - duration('P30D')
                  AND ctx.updated_at < datetime() - duration('PT1H')
            RETURN ctx.id,
                   duration.between(ctx.created_at, ctx.updated_at) as age,
                   date(ctx.created_at) as creation_date
            """,
        ]

        for query in comprehensive_queries:
            result = self.validator.validate_query(query.strip())
            assert result.is_valid, f"Comprehensive query should be valid: {query[:100]}..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
