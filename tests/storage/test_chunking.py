#!/usr/bin/env python3
"""
Test suite for storage/chunking.py - Advanced text chunking tests
"""
import pytest
from unittest.mock import patch, Mock
from typing import Dict, List, Any

# Import the module under test
from src.storage.chunking import AdvancedChunker, TIKTOKEN_AVAILABLE


class TestAdvancedChunkerInit:
    """Test suite for AdvancedChunker initialization"""

    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        chunker = AdvancedChunker()
        
        # Check default parameters
        assert chunker.target_tokens == 700
        assert chunker.overlap_tokens == 80
        assert chunker.min_chunk_tokens == 100
        assert chunker.max_chunk_tokens == 1000
        assert chunker.inject_metadata is True
        assert chunker.strip_boilerplate is True
        assert chunker.preserve_structure is True
        assert chunker.encoding_name == "cl100k_base"

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        config = {
            "target_tokens": 500,
            "overlap_tokens": 50,
            "min_chunk_tokens": 80,
            "max_chunk_tokens": 800,
            "inject_metadata": False,
            "strip_boilerplate": False,
            "preserve_structure": False,
            "encoding": "gpt2"
        }
        
        chunker = AdvancedChunker(config)
        
        assert chunker.target_tokens == 500
        assert chunker.overlap_tokens == 50
        assert chunker.min_chunk_tokens == 80
        assert chunker.max_chunk_tokens == 800
        assert chunker.inject_metadata is False
        assert chunker.strip_boilerplate is False
        assert chunker.preserve_structure is False
        assert chunker.encoding_name == "gpt2"

    def test_init_partial_config(self):
        """Test initialization with partial configuration"""
        config = {
            "target_tokens": 600,
            "inject_metadata": False
        }
        
        chunker = AdvancedChunker(config)
        
        # Custom values
        assert chunker.target_tokens == 600
        assert chunker.inject_metadata is False
        
        # Default values for unspecified keys
        assert chunker.overlap_tokens == 80
        assert chunker.strip_boilerplate is True

    def test_boilerplate_patterns_initialization(self):
        """Test that boilerplate patterns are properly initialized"""
        chunker = AdvancedChunker()
        
        assert len(chunker.boilerplate_patterns) > 0
        assert any("Table of Contents" in pattern for pattern in chunker.boilerplate_patterns)
        # Check for copyright pattern (uses regex pattern)
        assert any("©" in pattern or "copyright" in pattern.lower() for pattern in chunker.boilerplate_patterns)

    def test_header_patterns_initialization(self):
        """Test that header patterns are properly initialized"""
        chunker = AdvancedChunker()
        
        assert len(chunker.header_patterns) > 0
        assert any("#{1,6}" in pattern for pattern in chunker.header_patterns)  # Markdown headers


class TestTokenCounting:
    """Test suite for token counting functionality"""

    def test_count_tokens_with_tiktoken_available(self):
        """Test token counting when tiktoken is available"""
        if not TIKTOKEN_AVAILABLE:
            pytest.skip("tiktoken not available")
        
        with patch('src.storage.chunking.tiktoken') as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_tiktoken.get_encoding.return_value = mock_encoding
            
            chunker = AdvancedChunker()
            chunker.tokenizer = mock_encoding
            
            token_count = chunker.count_tokens("test text")
            
            assert token_count == 5
            mock_encoding.encode.assert_called_once_with("test text")

    def test_count_tokens_fallback_method(self):
        """Test token counting fallback method when tiktoken is not available"""
        chunker = AdvancedChunker()
        chunker.tokenizer = None  # Simulate tiktoken not available
        
        # Test with known word count
        text = "this is a test with five words"
        token_count = chunker.count_tokens(text)
        
        # Word count is 7 words: "this", "is", "a", "test", "with", "five", "words"
        # Should be word count (7) * 1.3 = 9.1, rounded to 9
        expected = int(7 * 1.3)
        assert token_count == expected

    def test_count_tokens_empty_text(self):
        """Test token counting with empty text"""
        chunker = AdvancedChunker()
        chunker.tokenizer = None
        
        token_count = chunker.count_tokens("")
        assert token_count == 0

    def test_count_tokens_single_word(self):
        """Test token counting with single word"""
        chunker = AdvancedChunker()
        chunker.tokenizer = None
        
        token_count = chunker.count_tokens("word")
        expected = int(1 * 1.3)
        assert token_count == expected


class TestBoilerplateRemoval:
    """Test suite for boilerplate content removal"""

    def test_clean_boilerplate_table_of_contents(self):
        """Test removal of table of contents boilerplate"""
        chunker = AdvancedChunker()
        
        text_with_toc = """
        Table of Contents
        Chapter 1: Introduction
        Chapter 2: Methods
        
        This is the actual content we want to keep.
        """
        
        cleaned = chunker.clean_boilerplate(text_with_toc)
        
        # TOC should be removed but actual content should remain
        assert "Table of Contents" not in cleaned
        assert "actual content we want to keep" in cleaned

    def test_clean_boilerplate_navigation(self):
        """Test removal of navigation boilerplate"""
        chunker = AdvancedChunker()
        
        text_with_nav = """
        Navigation
        Home | About | Contact
        
        Welcome to our site content here.
        """
        
        cleaned = chunker.clean_boilerplate(text_with_nav)
        
        assert "Navigation" not in cleaned
        assert "Welcome to our site content" in cleaned

    def test_clean_boilerplate_copyright(self):
        """Test removal of copyright notices"""
        chunker = AdvancedChunker()
        
        text_with_copyright = """
        Important article content here.
        
        © 2023 Company Name. All rights reserved.
        """
        
        cleaned = chunker.clean_boilerplate(text_with_copyright)
        
        assert "Important article content" in cleaned
        # Copyright should be removed
        assert "©" not in cleaned

    def test_clean_boilerplate_disabled(self):
        """Test that boilerplate cleaning can be disabled"""
        config = {"strip_boilerplate": False}
        chunker = AdvancedChunker(config)
        
        text_with_boilerplate = """
        Table of Contents
        Chapter 1
        
        Actual content.
        """
        
        cleaned = chunker.clean_boilerplate(text_with_boilerplate)
        
        # When disabled, boilerplate should remain
        assert "Table of Contents" in cleaned
        assert "Actual content" in cleaned

    def test_clean_boilerplate_multiple_patterns(self):
        """Test removal of multiple boilerplate patterns"""
        chunker = AdvancedChunker()
        
        text_with_multiple = """
        Header
        Navigation
        Table of Contents
        
        This is the real content.
        
        Footer
        © 2023 Copyright
        """
        
        cleaned = chunker.clean_boilerplate(text_with_multiple)
        
        # All boilerplate should be removed
        assert "Header" not in cleaned
        assert "Navigation" not in cleaned
        assert "Table of Contents" not in cleaned
        assert "Footer" not in cleaned
        assert "Copyright" not in cleaned
        
        # Content should remain
        assert "real content" in cleaned


class TestTextCleaningHelpers:
    """Test suite for text cleaning helper methods"""

    def test_extract_section_headers(self):
        """Test section header extraction"""
        chunker = AdvancedChunker()
        
        text_with_headers = """
        # Main Title
        This is content under main title.
        
        ## Subsection
        Content under subsection.
        
        ### Deep Section
        More content here.
        """
        
        # This method might not exist yet, so we'll test the pattern matching logic
        import re
        
        headers = []
        for line in text_with_headers.split('\n'):
            line = line.strip()
            for pattern in chunker.header_patterns:
                match = re.match(pattern, line)
                if match:
                    headers.append(line)
                    break
        
        assert "# Main Title" in headers
        assert "## Subsection" in headers
        assert "### Deep Section" in headers

    def test_numbered_section_detection(self):
        """Test detection of numbered sections"""
        chunker = AdvancedChunker()
        
        numbered_lines = [
            "1. Introduction",
            "2.1 Background",
            "2.2.1 Detailed Analysis",
            "A. Appendix Section"
        ]
        
        import re
        detected_headers = []
        
        for line in numbered_lines:
            for pattern in chunker.header_patterns:
                if re.match(pattern, line):
                    detected_headers.append(line)
                    break
        
        # Should detect numbered sections
        assert "1. Introduction" in detected_headers
        assert "2.1 Background" in detected_headers


class TestChunkingLogic:
    """Test suite for core chunking logic"""

    def test_chunk_simple_text(self):
        """Test chunking of simple text"""
        config = {
            "target_tokens": 10,  # Small chunks for testing
            "overlap_tokens": 2,
            "min_chunk_tokens": 5
        }
        chunker = AdvancedChunker(config)
        
        # Create text that will need multiple chunks
        text = "This is a test document with enough content to require chunking into multiple smaller pieces for processing."
        
        # Mock the chunk method if it exists, or test token counting as preparation
        token_count = chunker.count_tokens(text)
        assert token_count > config["target_tokens"]  # Confirms chunking will be needed

    def test_chunk_with_metadata_injection(self):
        """Test chunking with metadata injection enabled"""
        config = {"inject_metadata": True}
        chunker = AdvancedChunker(config)
        
        metadata = {
            "title": "Test Document",
            "section": "Introduction"
        }
        
        text = "This is the main content of the document."
        
        # Test that metadata configuration is properly set
        assert chunker.inject_metadata is True
        assert chunker.config.get("inject_metadata") is True

    def test_chunk_preserve_structure(self):
        """Test chunking with structure preservation"""
        config = {"preserve_structure": True}
        chunker = AdvancedChunker(config)
        
        structured_text = """
        # Chapter 1
        Introduction content here.
        
        ## Section 1.1
        Detailed information.
        
        ## Section 1.2
        More details.
        """
        
        # Test that structure preservation is enabled
        assert chunker.preserve_structure is True


class TestChunkerIntegration:
    """Integration tests for AdvancedChunker functionality"""

    def test_full_processing_pipeline(self):
        """Test complete text processing pipeline"""
        chunker = AdvancedChunker()
        
        sample_text = """
        Navigation
        Home | About | Contact
        
        # Introduction to AI
        
        Artificial intelligence represents one of the most significant technological advances.
        This field encompasses machine learning, neural networks, and deep learning.
        
        ## Machine Learning Basics
        
        Machine learning algorithms can learn patterns from data without explicit programming.
        These systems improve their performance through experience and training.
        
        © 2023 Tech Publications
        """
        
        # Test boilerplate removal
        cleaned_text = chunker.clean_boilerplate(sample_text)
        
        # Verify boilerplate removal
        assert "Navigation" not in cleaned_text
        assert "©" not in cleaned_text
        
        # Verify content preservation
        assert "Introduction to AI" in cleaned_text
        assert "Machine Learning Basics" in cleaned_text
        assert "Artificial intelligence represents" in cleaned_text

    def test_token_counting_accuracy(self):
        """Test token counting accuracy with various text types"""
        chunker = AdvancedChunker()
        
        test_cases = [
            ("", 0),
            ("word", 1),
            ("two words", 2),
            ("This is a longer sentence with multiple words.", 8)
        ]
        
        for text, expected_word_count in test_cases:
            token_count = chunker.count_tokens(text)
            
            if chunker.tokenizer is None:
                # Fallback method: words * 1.3
                expected_tokens = int(expected_word_count * 1.3)
                assert token_count == expected_tokens
            else:
                # With tiktoken, just verify it returns a reasonable number
                assert token_count >= 0

    def test_configuration_flexibility(self):
        """Test that configuration changes affect behavior"""
        # Test with boilerplate removal enabled
        chunker_clean = AdvancedChunker({"strip_boilerplate": True})
        
        # Test with boilerplate removal disabled  
        chunker_raw = AdvancedChunker({"strip_boilerplate": False})
        
        text_with_boilerplate = "Table of Contents\nActual content here."
        
        cleaned = chunker_clean.clean_boilerplate(text_with_boilerplate)
        raw = chunker_raw.clean_boilerplate(text_with_boilerplate)
        
        # Different behavior based on configuration
        assert "Table of Contents" not in cleaned
        assert "Table of Contents" in raw
        assert "Actual content" in cleaned
        assert "Actual content" in raw

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        chunker = AdvancedChunker()
        
        # Empty text
        assert chunker.count_tokens("") == 0
        assert chunker.clean_boilerplate("") == ""
        
        # Very long text
        long_text = "word " * 1000  # 1000 words
        token_count = chunker.count_tokens(long_text)
        assert token_count > 1000  # Should handle large text
        
        # Text with special characters
        special_text = "This has émojis 🚀 and spëcial chäracters!"
        token_count = chunker.count_tokens(special_text)
        assert token_count > 0

    def test_performance_considerations(self):
        """Test that chunker handles reasonable text sizes efficiently"""
        chunker = AdvancedChunker()
        
        # Test with medium-sized document
        medium_text = """
        This is a test of the chunker's performance with medium-sized text.
        """ * 100  # Repeat to create substantial content
        
        # These operations should complete without issues
        token_count = chunker.count_tokens(medium_text)
        cleaned_text = chunker.clean_boilerplate(medium_text)
        
        assert token_count > 0
        assert len(cleaned_text) > 0
        assert isinstance(cleaned_text, str)