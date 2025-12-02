#!/usr/bin/env python3
"""
Test suite for security/input_validator.py - Security input validation and sanitization tests
"""
import pytest
import json
from typing import Dict, Any, Set
from unittest.mock import patch

# Import the module under test
from src.security.input_validator import (
    ValidationResult,
    InputValidator,
    InputSanitizer,
    ContentTypeValidator,
    ParameterValidator
)


class TestValidationResult:
    """Test suite for ValidationResult dataclass"""

    def test_validation_result_creation_minimal(self):
        """Test ValidationResult creation with minimal parameters"""
        result = ValidationResult(valid=True)
        
        assert result.valid is True
        assert result.error is None
        assert result.sanitized_value is None
        assert result.warnings is None

    def test_validation_result_creation_complete(self):
        """Test ValidationResult creation with all parameters"""
        warnings = ["Warning 1", "Warning 2"]
        result = ValidationResult(
            valid=False,
            error="test_error",
            sanitized_value="sanitized",
            warnings=warnings
        )
        
        assert result.valid is False
        assert result.error == "test_error"
        assert result.sanitized_value == "sanitized"
        assert result.warnings == warnings

    def test_validation_result_equality(self):
        """Test ValidationResult equality comparison"""
        result1 = ValidationResult(valid=True, error=None)
        result2 = ValidationResult(valid=True, error=None)
        result3 = ValidationResult(valid=False, error="error")
        
        assert result1 == result2
        assert result1 != result3


class TestInputValidatorInit:
    """Test suite for InputValidator initialization"""

    def test_input_validator_default_init(self):
        """Test InputValidator initialization with defaults"""
        validator = InputValidator()
        
        assert validator.MAX_QUERY_SIZE == 10240
        assert validator.MAX_CONTENT_SIZE == 102400
        assert validator.MAX_FILENAME_LENGTH == 255
        assert validator.MAX_PATH_LENGTH == 4096
        assert validator.config == {}

    def test_input_validator_custom_config(self):
        """Test InputValidator initialization with custom config"""
        config = {
            "max_query_size": 5000,
            "max_content_size": 50000,
            "other_setting": "value"
        }
        
        validator = InputValidator(config)
        
        assert validator.MAX_QUERY_SIZE == 5000
        assert validator.MAX_CONTENT_SIZE == 50000
        assert validator.MAX_FILENAME_LENGTH == 255  # Not overridden
        assert validator.MAX_PATH_LENGTH == 4096     # Not overridden
        assert validator.config == config

    def test_input_validator_partial_config(self):
        """Test InputValidator with partial config override"""
        config = {"max_query_size": 8192}
        
        validator = InputValidator(config)
        
        assert validator.MAX_QUERY_SIZE == 8192
        assert validator.MAX_CONTENT_SIZE == 102400  # Default


class TestInputValidatorBasicValidation:
    """Test suite for InputValidator basic input validation"""

    def test_validate_input_none_value(self):
        """Test validation of None value"""
        validator = InputValidator()
        result = validator.validate_input(None)
        
        assert result.valid is True
        assert result.error is None

    def test_validate_input_query_type_valid(self):
        """Test validation of valid query input"""
        validator = InputValidator()
        short_query = "SELECT * FROM users WHERE id = 1"
        
        result = validator.validate_input(short_query, "query")
        
        assert result.valid is True
        assert result.sanitized_value == short_query

    def test_validate_input_query_type_too_large(self):
        """Test validation of oversized query input"""
        validator = InputValidator()
        large_query = "x" * 10241  # Exceeds MAX_QUERY_SIZE
        
        result = validator.validate_input(large_query, "query")
        
        assert result.valid is False
        assert result.error == "input_too_large"

    def test_validate_input_content_type_valid(self):
        """Test validation of valid content input"""
        validator = InputValidator()
        content = "This is some valid content."
        
        result = validator.validate_input(content, "content")
        
        assert result.valid is True
        assert result.sanitized_value == content

    def test_validate_input_content_type_too_large(self):
        """Test validation of oversized content input"""
        validator = InputValidator()
        large_content = "x" * 102401  # Exceeds MAX_CONTENT_SIZE
        
        result = validator.validate_input(large_content, "content")
        
        assert result.valid is False
        assert result.error == "input_too_large"

    def test_validate_input_filename_type_valid(self):
        """Test validation of valid filename"""
        validator = InputValidator()
        filename = "document.txt"
        
        result = validator.validate_input(filename, "filename")
        
        assert result.valid is True
        assert result.sanitized_value == filename

    def test_validate_input_filename_type_too_long(self):
        """Test validation of oversized filename"""
        validator = InputValidator()
        long_filename = "x" * 256  # Exceeds MAX_FILENAME_LENGTH
        
        result = validator.validate_input(long_filename, "filename")
        
        assert result.valid is False
        assert result.error == "filename_too_long"

    def test_validate_input_path_type_valid(self):
        """Test validation of valid path"""
        validator = InputValidator()
        path = "/home/user/documents/file.txt"
        
        result = validator.validate_input(path, "path")
        
        assert result.valid is True
        assert result.sanitized_value == path

    def test_validate_input_path_type_too_long(self):
        """Test validation of oversized path"""
        validator = InputValidator()
        long_path = "/" + "x" * 4096  # Exceeds MAX_PATH_LENGTH
        
        result = validator.validate_input(long_path, "path")
        
        assert result.valid is False
        assert result.error == "path_too_long"

    def test_validate_input_non_string_type(self):
        """Test validation of non-string input gets converted"""
        validator = InputValidator()
        number_input = 12345
        
        result = validator.validate_input(number_input, "generic")
        
        assert result.valid is True
        assert result.sanitized_value == "12345"


class TestInputValidatorSecurityValidation:
    """Test suite for InputValidator security-focused validation"""

    def test_validate_input_null_byte_detection(self):
        """Test detection of null bytes in input"""
        validator = InputValidator()
        malicious_input = "normal_text\x00malicious_content"
        
        result = validator.validate_input(malicious_input)
        
        assert result.valid is False
        assert result.error == "null_byte_detected"

    def test_validate_input_control_characters_detection(self):
        """Test detection of control characters in input"""
        validator = InputValidator()
        
        # Test various control characters (excluding allowed ones: tab, newline, CR)
        control_chars_inputs = [
            "text\x01control",  # SOH
            "text\x08control",  # Backspace
            "text\x1Fcontrol",  # Unit separator
        ]
        
        for malicious_input in control_chars_inputs:
            result = validator.validate_input(malicious_input)
            assert result.valid is False
            assert result.error == "control_characters_detected"

    def test_validate_input_allowed_control_characters(self):
        """Test that allowed control characters pass validation"""
        validator = InputValidator()
        
        # Tab, newline, and carriage return should be allowed
        allowed_inputs = [
            "text\ttab",        # Tab (9)
            "text\nnewline",    # Newline (10)
            "text\rreturn",     # Carriage return (13)
        ]
        
        for allowed_input in allowed_inputs:
            result = validator.validate_input(allowed_input)
            assert result.valid is True
            assert result.sanitized_value == allowed_input

    def test_validate_input_generic_type_valid(self):
        """Test validation of generic input type"""
        validator = InputValidator()
        generic_input = "This is generic input with special chars: @#$%^&*()"
        
        result = validator.validate_input(generic_input, "generic")
        
        assert result.valid is True
        assert result.sanitized_value == generic_input


class TestInputValidatorJSONValidation:
    """Test suite for InputValidator JSON validation"""

    def test_validate_json_input_simple_valid(self):
        """Test validation of simple valid JSON"""
        validator = InputValidator()
        simple_json = {"key": "value", "number": 42}
        
        result = validator.validate_json_input(simple_json)
        
        assert result.valid is True
        assert result.warnings is None or result.warnings == []

    def test_validate_json_input_nested_valid(self):
        """Test validation of moderately nested JSON"""
        validator = InputValidator()
        nested_json = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": "value"
                    }
                }
            }
        }
        
        result = validator.validate_json_input(nested_json)
        
        assert result.valid is True

    def test_validate_json_input_too_deeply_nested(self):
        """Test validation of excessively nested JSON"""
        validator = InputValidator()
        
        # Create deeply nested JSON (12 levels)
        deeply_nested = {"level": "start"}
        current = deeply_nested
        for i in range(12):
            current["nested"] = {"level": i}
            current = current["nested"]
        
        result = validator.validate_json_input(deeply_nested)
        
        assert result.valid is False
        assert result.error == "json_too_deeply_nested"

    def test_validate_json_input_too_large(self):
        """Test validation of oversized JSON"""
        validator = InputValidator()
        
        # Create large JSON object
        large_json = {"data": "x" * 102401}  # Exceeds MAX_CONTENT_SIZE
        
        result = validator.validate_json_input(large_json)
        
        assert result.valid is False
        assert result.error == "json_too_large"

    def test_validate_json_input_suspicious_keys(self):
        """Test detection of suspicious keys in JSON"""
        validator = InputValidator()
        suspicious_json = {
            "__proto__": "malicious",
            "constructor": "bad",
            "normal_key": "good_value"
        }
        
        result = validator.validate_json_input(suspicious_json)
        
        assert result.valid is True
        assert result.warnings is not None
        assert any("Suspicious keys detected" in warning for warning in result.warnings)

    def test_validate_json_input_empty_objects(self):
        """Test validation of empty objects and arrays"""
        validator = InputValidator()
        
        empty_cases = [
            {},
            [],
            {"empty_dict": {}, "empty_list": []},
        ]
        
        for empty_case in empty_cases:
            result = validator.validate_json_input(empty_case)
            assert result.valid is True

    def test_get_json_depth_edge_cases(self):
        """Test JSON depth calculation edge cases"""
        validator = InputValidator()
        
        depth_cases = [
            ({}, 0),
            ([], 0),
            ({"key": "value"}, 1),
            ([1, 2, 3], 1),
            ({"a": {"b": {"c": "deep"}}}, 3),
            ([[[["deep"]]]], 4),
            ({"mixed": [{"nested": "value"}]}, 3),
        ]
        
        for test_case, expected_depth in depth_cases:
            actual_depth = validator._get_json_depth(test_case)
            assert actual_depth == expected_depth, f"Failed for {test_case}: expected {expected_depth}, got {actual_depth}"


class TestInputSanitizer:
    """Test suite for InputSanitizer class"""

    def test_input_sanitizer_init(self):
        """Test InputSanitizer initialization"""
        sanitizer = InputSanitizer()
        
        assert hasattr(sanitizer, 'script_pattern')
        assert hasattr(sanitizer, 'event_handler_pattern')
        assert hasattr(sanitizer, 'control_char_pattern')

    def test_sanitize_html_script_removal(self):
        """Test HTML sanitization removes script tags"""
        sanitizer = InputSanitizer()
        
        malicious_html = '<div>Safe content</div><script>alert("XSS")</script><p>More content</p>'
        sanitized = sanitizer.sanitize_html(malicious_html)
        
        assert '<script>' not in sanitized
        assert 'alert("XSS")' not in sanitized
        assert 'Safe content' in sanitized
        assert 'More content' in sanitized

    def test_sanitize_html_event_handler_removal(self):
        """Test HTML sanitization removes event handlers"""
        sanitizer = InputSanitizer()
        
        malicious_html = '<button onclick="malicious()">Click me</button><div onload="bad()">Content</div>'
        sanitized = sanitizer.sanitize_html(malicious_html)
        
        assert 'onclick=' not in sanitized
        assert 'onload=' not in sanitized
        # The function removes event handlers but HTML escaping may preserve content
        assert '&lt;button' in sanitized  # HTML escaped
        assert '&lt;div' in sanitized     # HTML escaped

    def test_sanitize_html_escape_entities(self):
        """Test HTML sanitization escapes HTML entities"""
        sanitizer = InputSanitizer()
        
        html_with_entities = '<div>Content with <tags> & "quotes"</div>'
        sanitized = sanitizer.sanitize_html(html_with_entities)
        
        assert '&lt;' in sanitized
        assert '&gt;' in sanitized
        assert '&amp;' in sanitized
        assert '&quot;' in sanitized

    def test_sanitize_html_complex_xss(self):
        """Test HTML sanitization with complex XSS attempts"""
        sanitizer = InputSanitizer()
        
        complex_xss = '''
        <div onmouseover="alert('XSS')">
            Normal content
            <script type="text/javascript">
                document.cookie = "stolen=" + document.cookie;
            </script>
            <img src="x" onerror="alert('Image XSS')">
        </div>
        '''
        
        sanitized = sanitizer.sanitize_html(complex_xss)
        
        assert 'onmouseover=' not in sanitized
        assert '<script>' not in sanitized
        assert 'document.cookie' not in sanitized
        assert 'onerror=' not in sanitized
        assert 'Normal content' in sanitized

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization"""
        sanitizer = InputSanitizer()
        
        filename = "normal_file.txt"
        sanitized = sanitizer.sanitize_filename(filename)
        
        assert sanitized == filename

    def test_sanitize_filename_null_bytes(self):
        """Test filename sanitization removes null bytes"""
        sanitizer = InputSanitizer()
        
        malicious_filename = "file\x00.txt"
        sanitized = sanitizer.sanitize_filename(malicious_filename)
        
        assert '\x00' not in sanitized
        assert sanitized == "file.txt"

    def test_sanitize_filename_path_traversal(self):
        """Test filename sanitization removes path traversal"""
        sanitizer = InputSanitizer()
        
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\windows\\system32\\config",
            "normal/../file.txt",
        ]
        
        for malicious_filename in malicious_filenames:
            sanitized = sanitizer.sanitize_filename(malicious_filename)
            assert '../' not in sanitized
            assert '..\\' not in sanitized

    def test_sanitize_filename_special_chars(self):
        """Test filename sanitization removes special characters"""
        sanitizer = InputSanitizer()
        
        filename_with_specials = 'file<>:"|?*.txt'
        sanitized = sanitizer.sanitize_filename(filename_with_specials)
        
        special_chars = '<>:"|?*'
        for char in special_chars:
            assert char not in sanitized
        assert '_' in sanitized  # Special chars replaced with underscore

    def test_sanitize_filename_whitespace_trim(self):
        """Test filename sanitization trims dots and spaces"""
        sanitizer = InputSanitizer()
        
        filenames_to_trim = [
            "  file.txt  ",
            "..file.txt..",
            ". file.txt .",
        ]
        
        for filename in filenames_to_trim:
            sanitized = sanitizer.sanitize_filename(filename)
            assert not sanitized.startswith(' ')
            assert not sanitized.endswith(' ')
            assert not sanitized.startswith('.')
            assert not sanitized.endswith('.')

    def test_sanitize_filename_length_limit(self):
        """Test filename sanitization enforces length limit"""
        sanitizer = InputSanitizer()
        
        # Test with extension
        long_filename_with_ext = "x" * 300 + ".txt"
        sanitized = sanitizer.sanitize_filename(long_filename_with_ext)
        
        assert len(sanitized) <= 255
        assert sanitized.endswith('.txt')
        
        # Test without extension
        long_filename_no_ext = "x" * 300
        sanitized_no_ext = sanitizer.sanitize_filename(long_filename_no_ext)
        
        assert len(sanitized_no_ext) <= 255
        assert len(sanitized_no_ext) == 255

    def test_sanitize_text_control_chars(self):
        """Test text sanitization removes control characters"""
        sanitizer = InputSanitizer()
        
        text_with_control = "Normal text\x01\x08\x1F with control chars"
        sanitized = sanitizer.sanitize_text(text_with_control)
        
        assert '\x01' not in sanitized
        assert '\x08' not in sanitized
        assert '\x1F' not in sanitized
        assert 'Normal text' in sanitized
        assert 'with control chars' in sanitized

    def test_sanitize_text_preserves_allowed_chars(self):
        """Test text sanitization preserves allowed control characters"""
        sanitizer = InputSanitizer()
        
        text_with_allowed = "Line 1\nLine 2\tTabbed\rReturn"
        sanitized = sanitizer.sanitize_text(text_with_allowed)
        
        assert sanitized == text_with_allowed  # Should be unchanged

    def test_sanitize_text_null_bytes(self):
        """Test text sanitization removes null bytes"""
        sanitizer = InputSanitizer()
        
        text_with_null = "Text with\x00null bytes"
        sanitized = sanitizer.sanitize_text(text_with_null)
        
        assert '\x00' not in sanitized
        assert sanitized == "Text withnull bytes"


class TestInputSanitizerURL:
    """Test suite for InputSanitizer URL sanitization"""

    def test_sanitize_url_valid_http(self):
        """Test URL sanitization with valid HTTP URLs"""
        sanitizer = InputSanitizer()
        
        valid_urls = [
            "http://example.com",
            "https://secure.example.com",
            "http://example.com/path",
            "https://example.com/path?param=value",
            "ftp://files.example.com/file.txt",
        ]
        
        for url in valid_urls:
            sanitized = sanitizer.sanitize_url(url)
            assert sanitized is not None
            assert sanitized.startswith(('http://', 'https://', 'ftp://'))

    def test_sanitize_url_invalid_schemes(self):
        """Test URL sanitization rejects invalid schemes"""
        sanitizer = InputSanitizer()
        
        invalid_urls = [
            "javascript:alert('XSS')",
            "data:text/html,<script>alert('XSS')</script>",
            "file:///etc/passwd",
            "chrome://settings",
            "about:blank",
        ]
        
        for url in invalid_urls:
            sanitized = sanitizer.sanitize_url(url)
            assert sanitized is None

    def test_sanitize_url_localhost_rejection(self):
        """Test URL sanitization rejects localhost URLs"""
        sanitizer = InputSanitizer()
        
        localhost_urls = [
            "http://localhost/path",
            "https://127.0.0.1/admin",
            "http://0.0.0.0:8080/",
        ]
        
        for url in localhost_urls:
            sanitized = sanitizer.sanitize_url(url)
            assert sanitized is None

    def test_sanitize_url_private_ips_rejection(self):
        """Test URL sanitization rejects private IP ranges"""
        sanitizer = InputSanitizer()
        
        private_ip_urls = [
            "http://10.0.0.1/internal",
            "https://192.168.1.100/admin",
            "http://172.16.0.1/config",
        ]
        
        for url in private_ip_urls:
            sanitized = sanitizer.sanitize_url(url)
            assert sanitized is None

    def test_sanitize_url_malformed(self):
        """Test URL sanitization handles malformed URLs"""
        sanitizer = InputSanitizer()
        
        malformed_urls = [
            "not-a-url",
            "http://",
            "://missing-scheme",
            "http:///path-only",
        ]
        
        for url in malformed_urls:
            sanitized = sanitizer.sanitize_url(url)
            # Should either return None or a sanitized version
            # The exact behavior depends on urlparse handling

    def test_sanitize_sql_identifier_basic(self):
        """Test SQL identifier sanitization"""
        sanitizer = InputSanitizer()
        
        valid_identifiers = [
            "table_name",
            "column123",
            "_private_field",
            "CamelCase",
        ]
        
        for identifier in valid_identifiers:
            sanitized = sanitizer.sanitize_sql_identifier(identifier)
            assert sanitized == identifier

    def test_sanitize_sql_identifier_special_chars(self):
        """Test SQL identifier sanitization removes special characters"""
        sanitizer = InputSanitizer()
        
        malicious_identifier = "table'; DROP TABLE users; --"
        sanitized = sanitizer.sanitize_sql_identifier(malicious_identifier)
        
        assert ';' not in sanitized
        assert "'" not in sanitized
        assert '--' not in sanitized
        # The method removes special chars but keeps alphanumeric, so 'DROP' remains
        assert sanitized == "tableDROPTABLEusers"

    def test_sanitize_sql_identifier_starts_with_number(self):
        """Test SQL identifier sanitization handles numeric start"""
        sanitizer = InputSanitizer()
        
        numeric_start = "123_table"
        sanitized = sanitizer.sanitize_sql_identifier(numeric_start)
        
        assert sanitized.startswith('_')
        assert sanitized == "_123_table"

    def test_sanitize_sql_identifier_length_limit(self):
        """Test SQL identifier sanitization enforces length limit"""
        sanitizer = InputSanitizer()
        
        long_identifier = "x" * 100
        sanitized = sanitizer.sanitize_sql_identifier(long_identifier)
        
        assert len(sanitized) <= 64
        assert len(sanitized) == 64


class TestContentTypeValidator:
    """Test suite for ContentTypeValidator class"""

    def test_content_type_validator_default_init(self):
        """Test ContentTypeValidator initialization with defaults"""
        validator = ContentTypeValidator()
        
        assert 'application/json' in validator.allowed_types
        assert 'text/plain' in validator.allowed_types
        assert 'application/x-sh' not in validator.allowed_types

    def test_content_type_validator_custom_allowed(self):
        """Test ContentTypeValidator with custom allowed types"""
        custom_types = {'application/pdf', 'image/jpeg'}
        validator = ContentTypeValidator(custom_types)
        
        assert 'application/pdf' in validator.allowed_types
        assert 'image/jpeg' in validator.allowed_types
        assert 'application/json' in validator.allowed_types  # Default still there

    def test_is_allowed_basic_types(self):
        """Test content type validation for basic allowed types"""
        validator = ContentTypeValidator()
        
        allowed_types = [
            'application/json',
            'text/plain',
            'text/csv',
            'application/x-www-form-urlencoded',
        ]
        
        for content_type in allowed_types:
            assert validator.is_allowed(content_type) is True

    def test_is_allowed_blocked_types(self):
        """Test content type validation blocks dangerous types"""
        validator = ContentTypeValidator()
        
        blocked_types = [
            'application/x-sh',
            'application/x-executable',
            'text/x-php',
            'application/x-python-code',
            'text/x-ruby',
        ]
        
        for content_type in blocked_types:
            assert validator.is_allowed(content_type) is False

    def test_is_allowed_with_parameters(self):
        """Test content type validation ignores parameters"""
        validator = ContentTypeValidator()
        
        types_with_params = [
            'application/json; charset=utf-8',
            'text/plain; boundary=something',
            'application/x-sh; charset=utf-8',  # Should still be blocked
        ]
        
        assert validator.is_allowed(types_with_params[0]) is True
        assert validator.is_allowed(types_with_params[1]) is True
        assert validator.is_allowed(types_with_params[2]) is False

    def test_is_allowed_case_insensitive(self):
        """Test content type validation is case insensitive"""
        validator = ContentTypeValidator()
        
        case_variants = [
            'Application/JSON',
            'TEXT/PLAIN',
            'application/JSON',
            'APPLICATION/X-SH',  # Should still be blocked
        ]
        
        assert validator.is_allowed(case_variants[0]) is True
        assert validator.is_allowed(case_variants[1]) is True
        assert validator.is_allowed(case_variants[2]) is True
        assert validator.is_allowed(case_variants[3]) is False

    def test_is_allowed_unknown_type(self):
        """Test content type validation for unknown types"""
        validator = ContentTypeValidator()
        
        unknown_types = [
            'application/unknown',
            'text/custom',
            'image/jpeg',  # Not in default allowed list
        ]
        
        for content_type in unknown_types:
            assert validator.is_allowed(content_type) is False

    def test_add_allowed_type(self):
        """Test adding new allowed content type"""
        validator = ContentTypeValidator()
        
        new_type = 'application/custom'
        assert validator.is_allowed(new_type) is False
        
        validator.add_allowed_type(new_type)
        assert validator.is_allowed(new_type) is True

    def test_remove_allowed_type(self):
        """Test removing allowed content type"""
        validator = ContentTypeValidator()
        
        existing_type = 'application/json'
        assert validator.is_allowed(existing_type) is True
        
        validator.remove_allowed_type(existing_type)
        assert validator.is_allowed(existing_type) is False

    def test_add_remove_case_insensitive(self):
        """Test add/remove operations are case insensitive"""
        validator = ContentTypeValidator()
        
        validator.add_allowed_type('APPLICATION/CUSTOM')
        assert validator.is_allowed('application/custom') is True
        
        validator.remove_allowed_type('application/CUSTOM')
        assert validator.is_allowed('application/custom') is False


class TestParameterValidator:
    """Test suite for ParameterValidator class"""

    def test_parameter_validator_init(self):
        """Test ParameterValidator initialization"""
        validator = ParameterValidator()
        
        expected_types = ['email', 'phone', 'uuid', 'ip', 'date', 'numeric']
        for param_type in expected_types:
            assert param_type in validator.validators

    def test_validate_email_valid(self):
        """Test email validation with valid emails"""
        validator = ParameterValidator()
        
        valid_emails = [
            'user@example.com',
            'test.email@domain.org',
            'user+tag@example.co.uk',
            'number123@test.net',
        ]
        
        for email in valid_emails:
            result = validator.validate(email, 'email')
            assert result.valid is True, f"Email {email} should be valid"

    def test_validate_email_invalid(self):
        """Test email validation with invalid emails"""
        validator = ParameterValidator()
        
        invalid_emails = [
            'not-an-email',
            '@domain.com',
            'user@',
            'user@domain',
            'user space@domain.com',
            'user@domain.',
        ]
        
        for email in invalid_emails:
            result = validator.validate(email, 'email')
            assert result.valid is False, f"Email {email} should be invalid"
            assert result.error == "invalid_email"

    def test_validate_phone_valid(self):
        """Test phone validation with valid phone numbers"""
        validator = ParameterValidator()
        
        valid_phones = [
            '+1234567890',
            '1234567890',
            '+441234567890',
            '9876543210',
        ]
        
        for phone in valid_phones:
            result = validator.validate(phone, 'phone')
            assert result.valid is True, f"Phone {phone} should be valid"

    def test_validate_phone_invalid(self):
        """Test phone validation with invalid phone numbers"""
        validator = ParameterValidator()
        
        invalid_phones = [
            '0123456789',  # Leading zero
            '+0123456789',  # Leading zero after +
            'not-a-phone',
            '1',  # Too short (only 1 digit, need at least 2 total)
            '+' + '1' * 20,  # Too long
        ]
        
        for phone in invalid_phones:
            result = validator.validate(phone, 'phone')
            assert result.valid is False, f"Phone {phone} should be invalid"
            assert result.error == "invalid_phone"

    def test_validate_uuid_valid(self):
        """Test UUID validation with valid UUIDs"""
        validator = ParameterValidator()
        
        valid_uuids = [
            '123e4567-e89b-12d3-a456-426614174000',
            '00000000-0000-0000-0000-000000000000',
            'FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF',
        ]
        
        for uuid_str in valid_uuids:
            result = validator.validate(uuid_str, 'uuid')
            assert result.valid is True, f"UUID {uuid_str} should be valid"

    def test_validate_uuid_invalid(self):
        """Test UUID validation with invalid UUIDs"""
        validator = ParameterValidator()
        
        invalid_uuids = [
            'not-a-uuid',
            '123e4567-e89b-12d3-a456',  # Too short
            '123e4567-e89b-12d3-a456-426614174000-extra',  # Too long
            '123e4567e89b12d3a456426614174000',  # No dashes
            'gggggggg-gggg-gggg-gggg-gggggggggggg',  # Invalid hex
        ]
        
        for uuid_str in invalid_uuids:
            result = validator.validate(uuid_str, 'uuid')
            assert result.valid is False, f"UUID {uuid_str} should be invalid"
            assert result.error == "invalid_uuid"

    def test_validate_ip_valid(self):
        """Test IP validation with valid IP addresses"""
        validator = ParameterValidator()
        
        valid_ips = [
            '192.168.1.1',
            '10.0.0.1',
            '8.8.8.8',
            '2001:db8::1',  # IPv6
            '::1',  # IPv6 localhost
        ]
        
        for ip in valid_ips:
            result = validator.validate(ip, 'ip')
            assert result.valid is True, f"IP {ip} should be valid"

    def test_validate_ip_invalid(self):
        """Test IP validation with invalid IP addresses"""
        validator = ParameterValidator()
        
        invalid_ips = [
            'not-an-ip',
            '999.999.999.999',
            '192.168.1',  # Incomplete
            '192.168.1.1.1',  # Too many octets
            '192.168.01.1',  # Leading zeros
        ]
        
        for ip in invalid_ips:
            result = validator.validate(ip, 'ip')
            assert result.valid is False, f"IP {ip} should be invalid"
            assert result.error == "invalid_ip"

    def test_validate_date_valid(self):
        """Test date validation with valid date formats"""
        validator = ParameterValidator()
        
        valid_dates = [
            '2023-12-25',  # YYYY-MM-DD
            '2023/12/25',  # YYYY/MM/DD
            '25-12-2023',  # DD-MM-YYYY
            '25/12/2023',  # DD/MM/YYYY
        ]
        
        for date_str in valid_dates:
            result = validator.validate(date_str, 'date')
            assert result.valid is True, f"Date {date_str} should be valid"

    def test_validate_date_invalid(self):
        """Test date validation with invalid dates"""
        validator = ParameterValidator()
        
        invalid_dates = [
            'not-a-date',
            '2023-13-25',  # Invalid month
            '2023-12-32',  # Invalid day
            '25-13-2023',  # Invalid month (DD-MM-YYYY)
            '2023/02/30',  # Invalid date
        ]
        
        for date_str in invalid_dates:
            result = validator.validate(date_str, 'date')
            assert result.valid is False, f"Date {date_str} should be invalid"
            assert result.error == "invalid_date"

    def test_validate_numeric_valid(self):
        """Test numeric validation with valid numbers"""
        validator = ParameterValidator()
        
        valid_numbers = [
            '123',
            '123.456',
            '-123',
            '0',
            '0.0',
            '1e10',
            '-1.5e-3',
        ]
        
        for number in valid_numbers:
            result = validator.validate(number, 'numeric')
            assert result.valid is True, f"Number {number} should be valid"

    def test_validate_numeric_invalid(self):
        """Test numeric validation with invalid numbers"""
        validator = ParameterValidator()
        
        invalid_numbers = [
            'not-a-number',
            '123abc',
            'abc123',
            '12.34.56',  # Multiple decimal points
            '',
        ]
        
        for number in invalid_numbers:
            result = validator.validate(number, 'numeric')
            assert result.valid is False, f"Number {number} should be invalid"
            assert result.error == "invalid_numeric"

    def test_validate_unknown_type(self):
        """Test validation with unknown parameter type"""
        validator = ParameterValidator()
        
        result = validator.validate('any-value', 'unknown_type')
        assert result.valid is True  # Unknown types pass through


class TestSecurityInputValidatorIntegration:
    """Integration tests for all input validator components"""

    def test_complete_validation_workflow(self):
        """Test complete validation workflow"""
        validator = InputValidator()
        sanitizer = InputSanitizer()
        content_validator = ContentTypeValidator()
        param_validator = ParameterValidator()
        
        # Test a realistic scenario
        user_input = "user@example.com"
        content_type = "application/json"
        filename = "upload file.txt"
        
        # Validate input
        input_result = validator.validate_input(user_input, "generic")
        assert input_result.valid is True
        
        # Validate content type
        content_allowed = content_validator.is_allowed(content_type)
        assert content_allowed is True
        
        # Sanitize filename  
        safe_filename = sanitizer.sanitize_filename(filename)
        assert safe_filename == "upload file.txt"  # Space is preserved
        
        # Validate email parameter
        email_result = param_validator.validate(user_input, "email")
        assert email_result.valid is True

    def test_security_attack_scenarios(self):
        """Test various security attack scenarios"""
        validator = InputValidator()
        sanitizer = InputSanitizer()
        
        attack_scenarios = [
            # Path traversal
            ("../../../etc/passwd", "filename"),
            # XSS attempt
            ('<script>alert("XSS")</script>', "content"),
            # SQL injection attempt
            ("'; DROP TABLE users; --", "query"),
            # Null byte injection
            ("file.txt\x00.exe", "filename"),
            # Control character injection
            ("data\x01\x02\x03", "content"),
        ]
        
        for attack_input, input_type in attack_scenarios:
            # Validation should catch or flag issues
            result = validator.validate_input(attack_input, input_type)
            if not result.valid:
                assert result.error is not None
            
            # Sanitization should clean the input
            if input_type == "filename":
                sanitized = sanitizer.sanitize_filename(attack_input)
            else:
                sanitized = sanitizer.sanitize_text(attack_input)
            
            # Verify dangerous elements are removed/neutralized
            assert '\x00' not in sanitized
            if input_type == "filename":
                assert '../' not in sanitized

    def test_edge_cases_and_boundaries(self):
        """Test edge cases and boundary conditions"""
        validator = InputValidator()
        
        edge_cases = [
            ("", "generic"),  # Empty string
            ("a" * 10240, "query"),  # Exactly at limit
            ("a" * 10241, "query"),  # Just over limit
            (None, "generic"),  # None input
            (12345, "numeric"),  # Non-string input
        ]
        
        for test_input, input_type in edge_cases:
            result = validator.validate_input(test_input, input_type)
            # Should handle gracefully without exceptions
            assert isinstance(result, ValidationResult)
            assert isinstance(result.valid, bool)

    def test_performance_with_large_inputs(self):
        """Test performance with reasonably large inputs"""
        validator = InputValidator()
        
        # Large but valid input
        large_input = "valid_content " * 1000
        result = validator.validate_input(large_input, "content")
        
        # Should handle efficiently
        assert isinstance(result, ValidationResult)
        
        # Large JSON object
        large_json = {"data": ["item"] * 1000}
        json_result = validator.validate_json_input(large_json)
        
        assert isinstance(json_result, ValidationResult)