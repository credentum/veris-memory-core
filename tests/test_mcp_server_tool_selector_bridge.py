#!/usr/bin/env python3
"""
Comprehensive tests for MCP Tool Selector Bridge - Phase 10 Coverage

This test module provides comprehensive coverage for the MCP tool selector bridge
including tool selection, tool listing, error handling, and module loading.
"""
import pytest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, mock_open
from typing import Dict, Any, List, Optional

# Import tool selector bridge components
try:
    from src.mcp_server.tool_selector_bridge import (
        ToolSelectorBridge,
        get_tool_selector_bridge,
        _load_tool_selector_module
    )
    TOOL_SELECTOR_BRIDGE_AVAILABLE = True
except ImportError:
    TOOL_SELECTOR_BRIDGE_AVAILABLE = False


# Mock classes for testing
class MockTool:
    """Mock tool class for testing"""
    
    def __init__(self, name, description, category, keywords, metadata=None, relevance_score=0.8):
        self.name = name
        self.description = description
        self.category = category
        self.keywords = keywords
        self.metadata = metadata or {}
        self.relevance_score = relevance_score


class MockToolSelector:
    """Mock ToolSelector class for testing"""
    
    def __init__(self, tools_directory):
        self.tools_directory = tools_directory
        self.tools = [
            MockTool("search_documents", "Search through documents", "search", ["search", "find", "query"]),
            MockTool("create_report", "Create a new report", "creation", ["create", "generate", "new"]),
            MockTool("analyze_data", "Analyze data patterns", "analysis", ["analyze", "pattern", "data"]),
            MockTool("export_results", "Export results to file", "io", ["export", "save", "file"]),
            MockTool("validate_input", "Validate user input", "validation", ["validate", "check", "verify"])
        ]
    
    def select_tools(self, query, max_tools=5, method="hybrid"):
        # Simple mock selection based on query
        relevant_tools = []
        query_lower = query.lower()
        
        for tool in self.tools:
            if any(keyword in query_lower for keyword in tool.keywords):
                relevant_tools.append(tool)
        
        return relevant_tools[:max_tools]
    
    def get_all_tools(self):
        return self.tools.copy()
    
    def get_tool_by_name(self, name):
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None


class MockToolSelectorConfig:
    """Mock ToolSelectorConfig class for testing"""
    MAX_TOOLS_LIMIT = 20


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestToolSelectorModuleLoading:
    """Test tool selector module loading functionality"""
    
    def test_load_tool_selector_module_success(self):
        """Test successful tool selector module loading"""
        # Mock the path resolution and module loading
        with patch('pathlib.Path.resolve') as mock_resolve:
            with patch('pathlib.Path.is_relative_to', return_value=True):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('importlib.util.spec_from_file_location') as mock_spec:
                            with patch('importlib.util.module_from_spec') as mock_module:
                                
                                # Mock the module spec and loader
                                mock_spec_obj = MagicMock()
                                mock_loader = MagicMock()
                                mock_spec_obj.loader = mock_loader
                                mock_spec.return_value = mock_spec_obj
                                
                                # Mock the loaded module
                                mock_tool_selector_module = MagicMock()
                                mock_module.return_value = mock_tool_selector_module
                                
                                # Mock resolve to return predictable paths
                                mock_resolve.side_effect = [
                                    Path("/test/repo/root"),  # repo_root
                                    Path("/test/expected/base")  # expected_base
                                ]
                                
                                result = _load_tool_selector_module()
                                
                                assert result == mock_tool_selector_module
                                mock_loader.exec_module.assert_called_once_with(mock_tool_selector_module)
    
    def test_load_tool_selector_module_path_validation_failure(self):
        """Test tool selector module loading with path validation failure"""
        with patch('pathlib.Path.resolve') as mock_resolve:
            with patch('pathlib.Path.is_relative_to', return_value=False):
                with patch('logging.getLogger') as mock_logger:
                    mock_logger_instance = MagicMock()
                    mock_logger.return_value = mock_logger_instance
                    
                    result = _load_tool_selector_module()
                    
                    assert result is None
                    mock_logger_instance.warning.assert_called_once()
    
    def test_load_tool_selector_module_missing_context_tools(self):
        """Test tool selector module loading with missing context/tools directory"""
        with patch('pathlib.Path.resolve') as mock_resolve:
            with patch('pathlib.Path.is_relative_to', return_value=True):
                with patch('pathlib.Path.exists') as mock_exists:
                    with patch('pathlib.Path.is_dir', return_value=True):
                        
                        # Mock exists to return False for context/tools directory
                        def exists_side_effect(path_obj):
                            return str(path_obj).endswith("context/tools") == False
                        
                        mock_exists.side_effect = exists_side_effect
                        
                        result = _load_tool_selector_module()
                        
                        assert result is None
    
    def test_load_tool_selector_module_missing_tool_selector_file(self):
        """Test tool selector module loading with missing tool_selector.py file"""
        with patch('pathlib.Path.resolve') as mock_resolve:
            with patch('pathlib.Path.is_relative_to', return_value=True):
                with patch('pathlib.Path.exists') as mock_exists:
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('logging.getLogger') as mock_logger:
                            
                            # Mock exists - context/tools exists, but tool_selector.py doesn't
                            def exists_side_effect(path_obj):
                                return "tool_selector.py" not in str(path_obj)
                            
                            mock_exists.side_effect = exists_side_effect
                            
                            mock_logger_instance = MagicMock()
                            mock_logger.return_value = mock_logger_instance
                            
                            result = _load_tool_selector_module()
                            
                            assert result is None
                            mock_logger_instance.error.assert_called()
    
    def test_load_tool_selector_module_import_error(self):
        """Test tool selector module loading with import error"""
        with patch('pathlib.Path.resolve') as mock_resolve:
            with patch('pathlib.Path.is_relative_to', return_value=True):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.is_dir', return_value=True):
                        with patch('importlib.util.spec_from_file_location') as mock_spec:
                            with patch('logging.getLogger') as mock_logger:
                                
                                # Mock spec creation failure
                                mock_spec.return_value = None
                                
                                mock_logger_instance = MagicMock()
                                mock_logger.return_value = mock_logger_instance
                                
                                result = _load_tool_selector_module()
                                
                                assert result is None
                                mock_logger_instance.error.assert_called()
    
    def test_load_tool_selector_module_os_error(self):
        """Test tool selector module loading with OS error"""
        with patch('pathlib.Path.resolve', side_effect=OSError("Path resolution failed")):
            with patch('logging.getLogger') as mock_logger:
                mock_logger_instance = MagicMock()
                mock_logger.return_value = mock_logger_instance
                
                result = _load_tool_selector_module()
                
                assert result is None
                mock_logger_instance.warning.assert_called()


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestToolSelectorBridge:
    """Test ToolSelectorBridge basic functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.bridge = ToolSelectorBridge(tools_directory=str(self.temp_path))
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_bridge_initialization(self):
        """Test bridge initialization"""
        assert self.bridge is not None
        assert self.bridge.tools_directory == str(self.temp_path)
        assert self.bridge._selector is None
        assert self.bridge._initialized is False
        assert self.bridge.logger is not None
    
    def test_bridge_initialization_without_tools_directory(self):
        """Test bridge initialization without tools directory"""
        bridge = ToolSelectorBridge()
        
        assert bridge.tools_directory is None
        assert bridge._selector is None
        assert bridge._initialized is False
    
    def test_ensure_initialized_success(self):
        """Test successful initialization"""
        # Mock ToolSelector class in the loaded module
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = self.bridge._ensure_initialized()
            
            assert result is True
            assert self.bridge._initialized is True
            assert self.bridge._selector is not None
    
    def test_ensure_initialized_already_initialized(self):
        """Test initialization when already initialized"""
        # Pre-initialize
        self.bridge._selector = MockToolSelector(self.temp_path)
        self.bridge._initialized = True
        
        result = self.bridge._ensure_initialized()
        
        assert result is True
        assert self.bridge._initialized is True
    
    def test_ensure_initialized_os_error(self):
        """Test initialization with OS error"""
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', side_effect=OSError("File system error")):
            result = self.bridge._ensure_initialized()
            
            assert result is False
            assert self.bridge._initialized is False
    
    def test_ensure_initialized_import_error(self):
        """Test initialization with import error"""
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', side_effect=ImportError("Module not found")):
            result = self.bridge._ensure_initialized()
            
            assert result is False
            assert self.bridge._initialized is False
    
    def test_get_selector_success(self):
        """Test successful selector retrieval"""
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            selector = self.bridge._get_selector()
            
            assert selector is not None
            assert isinstance(selector, MockToolSelector)
    
    def test_get_selector_initialization_failure(self):
        """Test selector retrieval with initialization failure"""
        with patch.object(self.bridge, '_ensure_initialized', return_value=False):
            with pytest.raises(RuntimeError, match="Tool selector not properly initialized"):
                self.bridge._get_selector()


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestToolSelection:
    """Test tool selection functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bridge = ToolSelectorBridge()
        
        # Mock the loaded module globals
        with patch('src.mcp_server.tool_selector_bridge._tool_selector_module') as mock_module:
            mock_module.ToolSelectorConfig = MockToolSelectorConfig
    
    @patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector)
    async def test_select_tools_success(self):
        """Test successful tool selection"""
        arguments = {
            "query": "search for documents",
            "max_tools": 3,
            "method": "hybrid",
            "include_scores": True
        }
        
        result = await self.bridge.select_tools(arguments)
        
        assert result["success"] is True
        assert "tools" in result
        assert len(result["tools"]) > 0
        assert result["total_available"] == 5  # MockToolSelector has 5 tools
        assert result["method_used"] == "hybrid"
        assert "selection_log" in result
        
        # Check tool structure
        first_tool = result["tools"][0]
        assert "name" in first_tool
        assert "description" in first_tool
        assert "category" in first_tool
        assert "keywords" in first_tool
        assert "relevance_score" in first_tool
    
    async def test_select_tools_missing_query(self):
        """Test tool selection with missing query"""
        arguments = {
            "max_tools": 3,
            "method": "hybrid"
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.select_tools(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"
            assert "Query parameter is required" in result["message"]
    
    async def test_select_tools_empty_query(self):
        """Test tool selection with empty query"""
        arguments = {
            "query": "",
            "max_tools": 3
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.select_tools(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"
    
    async def test_select_tools_invalid_max_tools(self):
        """Test tool selection with invalid max_tools parameter"""
        arguments = {
            "query": "search",
            "max_tools": 0  # Invalid
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            with patch('src.mcp_server.tool_selector_bridge._tool_selector_module') as mock_module:
                mock_module.ToolSelectorConfig = MockToolSelectorConfig
                
                result = await self.bridge.select_tools(arguments)
                
                assert result["success"] is False
                assert result["error_type"] == "invalid_parameter"
                assert "max_tools must be between 1 and" in result["message"]
    
    async def test_select_tools_invalid_method(self):
        """Test tool selection with invalid method"""
        arguments = {
            "query": "search",
            "max_tools": 3,
            "method": "invalid_method"
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.select_tools(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "invalid_parameter"
            assert "Invalid method" in result["message"]
    
    async def test_select_tools_with_category_filter(self):
        """Test tool selection with category filter"""
        arguments = {
            "query": "search",
            "max_tools": 5,
            "category_filter": "search"
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.select_tools(arguments)
            
            assert result["success"] is True
            # Should only return tools from 'search' category
            for tool in result["tools"]:
                assert tool["category"].lower() == "search"
    
    async def test_select_tools_without_scores(self):
        """Test tool selection without relevance scores"""
        arguments = {
            "query": "search",
            "max_tools": 3,
            "include_scores": False
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.select_tools(arguments)
            
            assert result["success"] is True
            # Tools should not include relevance_score
            for tool in result["tools"]:
                assert "relevance_score" not in tool
    
    async def test_select_tools_initialization_failure(self):
        """Test tool selection with initialization failure"""
        arguments = {
            "query": "search",
            "max_tools": 3
        }
        
        with patch.object(self.bridge, '_ensure_initialized', return_value=False):
            result = await self.bridge.select_tools(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "initialization_error"
            assert "initialization failed" in result["message"]
    
    async def test_select_tools_exception_handling(self):
        """Test tool selection with exception during processing"""
        arguments = {
            "query": "search",
            "max_tools": 3
        }
        
        # Mock selector that raises exception
        mock_selector = MagicMock()
        mock_selector.tools = []
        mock_selector.select_tools.side_effect = Exception("Selection failed")
        
        with patch.object(self.bridge, '_get_selector', return_value=mock_selector):
            result = await self.bridge.select_tools(arguments)
            
            assert result["success"] is False
            assert "Selection failed" in result["message"]


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestToolListing:
    """Test tool listing functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bridge = ToolSelectorBridge()
    
    @patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector)
    async def test_list_available_tools_success(self):
        """Test successful tool listing"""
        arguments = {
            "include_metadata": True,
            "sort_by": "name"
        }
        
        result = await self.bridge.list_available_tools(arguments)
        
        assert result["success"] is True
        assert "tools" in result
        assert len(result["tools"]) == 5  # MockToolSelector has 5 tools
        assert result["total_count"] == 5
        assert "categories" in result
        assert len(result["categories"]) > 0
        
        # Check tool structure
        first_tool = result["tools"][0]
        assert "name" in first_tool
        assert "description" in first_tool
        assert "category" in first_tool
        assert "keywords" in first_tool
        assert "metadata" in first_tool
    
    async def test_list_available_tools_without_metadata(self):
        """Test tool listing without metadata"""
        arguments = {
            "include_metadata": False
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.list_available_tools(arguments)
            
            assert result["success"] is True
            # Tools should not include metadata
            for tool in result["tools"]:
                assert "metadata" not in tool
    
    async def test_list_available_tools_sort_by_category(self):
        """Test tool listing sorted by category"""
        arguments = {
            "sort_by": "category"
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.list_available_tools(arguments)
            
            assert result["success"] is True
            
            # Verify sorting by category
            categories = [tool["category"] for tool in result["tools"]]
            assert categories == sorted(categories)
    
    async def test_list_available_tools_sort_by_keywords(self):
        """Test tool listing sorted by keyword count"""
        arguments = {
            "sort_by": "keywords"
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.list_available_tools(arguments)
            
            assert result["success"] is True
            
            # Verify sorting by keyword count (descending)
            keyword_counts = [len(tool["keywords"]) for tool in result["tools"]]
            assert keyword_counts == sorted(keyword_counts, reverse=True)
    
    async def test_list_available_tools_with_category_filter(self):
        """Test tool listing with category filter"""
        arguments = {
            "category_filter": "search"
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.list_available_tools(arguments)
            
            assert result["success"] is True
            # All returned tools should be from 'search' category
            for tool in result["tools"]:
                assert tool["category"].lower() == "search"
    
    async def test_list_available_tools_initialization_failure(self):
        """Test tool listing with initialization failure"""
        arguments = {}
        
        with patch.object(self.bridge, '_ensure_initialized', return_value=False):
            result = await self.bridge.list_available_tools(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "initialization_error"
    
    async def test_list_available_tools_exception_handling(self):
        """Test tool listing with exception during processing"""
        arguments = {}
        
        # Mock selector that raises exception
        mock_selector = MagicMock()
        mock_selector.get_all_tools.side_effect = Exception("Listing failed")
        
        with patch.object(self.bridge, '_get_selector', return_value=mock_selector):
            result = await self.bridge.list_available_tools(arguments)
            
            assert result["success"] is False
            assert "Listing failed" in result["message"]


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestToolInfo:
    """Test tool info functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bridge = ToolSelectorBridge()
        
        # Create temporary file for documentation testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test documentation file
        self.doc_file = self.temp_path / "test_tool.md"
        with open(self.doc_file, 'w') as f:
            f.write("""# Test Tool Documentation

This is a test tool for demonstration.

## Usage

Basic usage example:

```python
tool.execute(param="value")
```

## Examples

### Example 1

Simple example:

```bash
tool --help
```

### Example 2

Complex example:

```json
{"config": "value"}
```

Example: `inline_code_example`
""")
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector)
    async def test_get_tool_info_success(self):
        """Test successful tool info retrieval"""
        arguments = {
            "tool_name": "search_documents",
            "include_usage_examples": True
        }
        
        result = await self.bridge.get_tool_info(arguments)
        
        assert result["success"] is True
        assert "tool" in result
        
        tool_info = result["tool"]
        assert tool_info["name"] == "search_documents"
        assert tool_info["description"] == "Search through documents"
        assert tool_info["category"] == "search"
        assert tool_info["keywords"] == ["search", "find", "query"]
    
    async def test_get_tool_info_missing_tool_name(self):
        """Test tool info with missing tool name"""
        arguments = {
            "include_usage_examples": True
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"
            assert "tool_name parameter is required" in result["message"]
    
    async def test_get_tool_info_empty_tool_name(self):
        """Test tool info with empty tool name"""
        arguments = {
            "tool_name": "",
            "include_usage_examples": True
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "missing_parameter"
    
    async def test_get_tool_info_tool_not_found(self):
        """Test tool info with non-existent tool"""
        arguments = {
            "tool_name": "nonexistent_tool",
            "include_usage_examples": True
        }
        
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector', MockToolSelector):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "tool_not_found"
            assert "not found" in result["message"]
    
    async def test_get_tool_info_with_documentation(self):
        """Test tool info with documentation file"""
        arguments = {
            "tool_name": "search_documents",
            "include_usage_examples": True
        }
        
        # Create mock tool with file path metadata
        mock_tool = MockTool(
            "search_documents", 
            "Search through documents", 
            "search", 
            ["search", "find"],
            metadata={"file_path": str(self.doc_file)}
        )
        
        mock_selector = MagicMock()
        mock_selector.get_tool_by_name.return_value = mock_tool
        
        with patch.object(self.bridge, '_get_selector', return_value=mock_selector):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is True
            tool_info = result["tool"]
            
            assert "documentation" in tool_info
            assert "Test Tool Documentation" in tool_info["documentation"]
            assert "usage_examples" in tool_info
            assert len(tool_info["usage_examples"]) > 0
    
    async def test_get_tool_info_without_usage_examples(self):
        """Test tool info without usage examples"""
        arguments = {
            "tool_name": "search_documents",
            "include_usage_examples": False
        }
        
        # Create mock tool with file path metadata
        mock_tool = MockTool(
            "search_documents", 
            "Search through documents", 
            "search", 
            ["search", "find"],
            metadata={"file_path": str(self.doc_file)}
        )
        
        mock_selector = MagicMock()
        mock_selector.get_tool_by_name.return_value = mock_tool
        
        with patch.object(self.bridge, '_get_selector', return_value=mock_selector):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is True
            tool_info = result["tool"]
            
            assert "documentation" in tool_info
            assert "usage_examples" not in tool_info
    
    async def test_get_tool_info_missing_documentation_file(self):
        """Test tool info with missing documentation file"""
        arguments = {
            "tool_name": "search_documents",
            "include_usage_examples": True
        }
        
        # Create mock tool with non-existent file path
        mock_tool = MockTool(
            "search_documents", 
            "Search through documents", 
            "search", 
            ["search", "find"],
            metadata={"file_path": "/nonexistent/file.md"}
        )
        
        mock_selector = MagicMock()
        mock_selector.get_tool_by_name.return_value = mock_tool
        
        with patch.object(self.bridge, '_get_selector', return_value=mock_selector):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is True
            tool_info = result["tool"]
            
            # Should not crash, just not include documentation
            assert "documentation" not in tool_info
    
    async def test_get_tool_info_initialization_failure(self):
        """Test tool info with initialization failure"""
        arguments = {
            "tool_name": "search_documents"
        }
        
        with patch.object(self.bridge, '_ensure_initialized', return_value=False):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is False
            assert result["error_type"] == "initialization_error"
    
    async def test_get_tool_info_exception_handling(self):
        """Test tool info with exception during processing"""
        arguments = {
            "tool_name": "search_documents"
        }
        
        # Mock selector that raises exception
        mock_selector = MagicMock()
        mock_selector.get_tool_by_name.side_effect = Exception("Info retrieval failed")
        
        with patch.object(self.bridge, '_get_selector', return_value=mock_selector):
            result = await self.bridge.get_tool_info(arguments)
            
            assert result["success"] is False
            assert "Info retrieval failed" in result["message"]


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestErrorHandling:
    """Test error handling functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bridge = ToolSelectorBridge()
    
    def test_handle_async_error_data_error(self):
        """Test async error handling for data errors"""
        error = ValueError("Invalid data format")
        operation = "tool selection"
        default_response = {"tools": [], "total_available": 0}
        
        result = self.bridge._handle_async_error(error, operation, default_response)
        
        assert result["success"] is False
        assert result["error_type"] == "data_error"
        assert "Tool selection failed" in result["message"]
        assert "Invalid data format" in result["message"]
        assert result["tools"] == []
        assert result["total_available"] == 0
    
    def test_handle_async_error_filesystem_error(self):
        """Test async error handling for filesystem errors"""
        error = OSError("Permission denied")
        operation = "file reading"
        default_response = {}
        
        result = self.bridge._handle_async_error(error, operation, default_response)
        
        assert result["success"] is False
        assert result["error_type"] == "filesystem_error"
        assert "File reading failed" in result["message"]
    
    def test_handle_async_error_import_error(self):
        """Test async error handling for import errors"""
        error = ImportError("Module not found")
        operation = "module loading"
        default_response = {}
        
        result = self.bridge._handle_async_error(error, operation, default_response)
        
        assert result["success"] is False
        assert result["error_type"] == "import_error"
        assert "Module loading failed" in result["message"]
    
    def test_handle_async_error_runtime_error(self):
        """Test async error handling for runtime errors"""
        error = RuntimeError("Execution failed")
        operation = "tool execution"
        default_response = {}
        
        result = self.bridge._handle_async_error(error, operation, default_response)
        
        assert result["success"] is False
        assert result["error_type"] == "runtime_error"
        assert "Tool execution failed" in result["message"]
    
    def test_handle_async_error_unknown_error(self):
        """Test async error handling for unknown errors"""
        error = Exception("Unknown error")
        operation = "unknown operation"
        default_response = {}
        
        result = self.bridge._handle_async_error(error, operation, default_response)
        
        assert result["success"] is False
        assert result["error_type"] == "unknown_error"
        assert "Unknown operation failed" in result["message"]
    
    def test_handle_sync_error_data_error(self):
        """Test sync error handling for data errors"""
        error = KeyError("Missing key")
        operation = "data processing"
        default_response = {"status": "error"}
        
        result = self.bridge._handle_sync_error(error, operation, default_response)
        
        assert result["success"] is False
        assert result["error_type"] == "data_error"
        assert "Data processing failed" in result["message"]
        assert result["status"] == "error"
    
    def test_handle_sync_error_consistency(self):
        """Test that sync and async error handling are consistent"""
        error = ValueError("Test error")
        operation = "test operation"
        default_response = {"test": "value"}
        
        async_result = self.bridge._handle_async_error(error, operation, default_response)
        sync_result = self.bridge._handle_sync_error(error, operation, default_response)
        
        # Should have same error categorization
        assert async_result["error_type"] == sync_result["error_type"]
        assert async_result["success"] == sync_result["success"]
        assert async_result["test"] == sync_result["test"]


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestUsageExampleExtraction:
    """Test usage example extraction functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bridge = ToolSelectorBridge()
    
    def test_extract_usage_examples_basic(self):
        """Test basic usage example extraction"""
        markdown_content = """
# Tool Documentation

## Usage

Basic usage:

```python
tool.execute()
```

## Examples

Example usage:

```bash
tool --help
```
"""
        
        examples = self.bridge._extract_usage_examples(markdown_content)
        
        assert len(examples) >= 2
        
        # Check for Python example
        python_examples = [ex for ex in examples if ex["language"] == "python"]
        assert len(python_examples) >= 1
        assert "tool.execute()" in python_examples[0]["code"]
        
        # Check for Bash example
        bash_examples = [ex for ex in examples if ex["language"] == "bash"]
        assert len(bash_examples) >= 1
        assert "tool --help" in bash_examples[0]["code"]
    
    def test_extract_usage_examples_with_descriptions(self):
        """Test usage example extraction with descriptions"""
        markdown_content = """
## Examples

### Simple Example

This shows basic usage:

```python
result = tool.run()
```

### Advanced Example

Complex configuration:

```json
{"config": {"param": "value"}}
```
"""
        
        examples = self.bridge._extract_usage_examples(markdown_content)
        
        assert len(examples) >= 2
        
        # Examples should have meaningful descriptions
        descriptions = [ex["description"] for ex in examples]
        assert any("Simple" in desc or "basic" in desc for desc in descriptions)
        assert any("Advanced" in desc or "Complex" in desc for desc in descriptions)
    
    def test_extract_usage_examples_inline(self):
        """Test inline usage example extraction"""
        markdown_content = """
# Tool Documentation

Quick example: `tool.quick_action()`

Another example: `tool --version`
"""
        
        examples = self.bridge._extract_usage_examples(markdown_content)
        
        assert len(examples) >= 2
        
        # Check for inline examples
        inline_examples = [ex for ex in examples if ex["description"] == "Inline example"]
        assert len(inline_examples) >= 2
    
    def test_extract_usage_examples_empty_content(self):
        """Test usage example extraction with empty content"""
        markdown_content = ""
        
        examples = self.bridge._extract_usage_examples(markdown_content)
        
        assert examples == []
    
    def test_extract_usage_examples_no_examples(self):
        """Test usage example extraction with no code examples"""
        markdown_content = """
# Tool Documentation

This tool is very useful for various tasks.

## Description

It provides functionality for processing data.
"""
        
        examples = self.bridge._extract_usage_examples(markdown_content)
        
        assert examples == []
    
    def test_extract_usage_examples_mixed_sections(self):
        """Test usage example extraction from mixed sections"""
        markdown_content = """
# Tool Documentation

## Getting Started

Initial setup:

```bash
pip install tool
```

## How to Use

Basic operation:

```python
from tool import Tool
t = Tool()
t.process()
```

## Configuration

Config example:

```yaml
setting: value
```
"""
        
        examples = self.bridge._extract_usage_examples(markdown_content)
        
        assert len(examples) >= 3
        
        # Should extract from all usage-related sections
        languages = [ex["language"] for ex in examples]
        assert "bash" in languages
        assert "python" in languages
        assert "yaml" in languages


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestBridgeUtilities:
    """Test bridge utility functions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bridge = ToolSelectorBridge()
    
    def test_get_status_not_initialized(self):
        """Test status retrieval when not initialized"""
        status = self.bridge.get_status()
        
        assert isinstance(status, dict)
        assert status["initialized"] is False
        assert status["tools_loaded"] == 0
        assert status["available"] is False
        assert "tools_directory" in status
    
    def test_get_status_initialized(self):
        """Test status retrieval when initialized"""
        # Mock initialized state
        mock_selector = MockToolSelector("/test/tools")
        self.bridge._selector = mock_selector
        self.bridge._initialized = True
        self.bridge.tools_directory = "/test/tools"
        
        status = self.bridge.get_status()
        
        assert status["initialized"] is True
        assert status["tools_loaded"] == 5  # MockToolSelector has 5 tools
        assert status["available"] is True
        assert status["tools_directory"] == "/test/tools"
    
    def test_get_status_with_exception(self):
        """Test status retrieval with exception"""
        # Mock selector that raises exception when accessing tools
        mock_selector = MagicMock()
        mock_selector.tools = property(lambda self: exec('raise Exception("Tools access failed")'))
        
        self.bridge._selector = mock_selector
        self.bridge._initialized = True
        
        status = self.bridge.get_status()
        
        # Should handle exception gracefully
        assert isinstance(status, dict)
        assert status["success"] is False
        assert "Tools access failed" in status["message"]


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestGlobalBridgeInstance:
    """Test global bridge instance functionality"""
    
    def test_get_tool_selector_bridge_singleton(self):
        """Test that get_tool_selector_bridge returns singleton"""
        # Clear any existing global instance
        import src.mcp_server.tool_selector_bridge as bridge_module
        bridge_module._tool_selector_bridge = None
        
        # Get first instance
        bridge1 = get_tool_selector_bridge("/test/tools1")
        
        # Get second instance (should be same)
        bridge2 = get_tool_selector_bridge("/test/tools2")
        
        assert bridge1 is bridge2
        # Should use tools directory from first creation
        assert bridge1.tools_directory == "/test/tools1"
    
    def test_get_tool_selector_bridge_without_directory(self):
        """Test getting bridge without tools directory"""
        # Clear any existing global instance
        import src.mcp_server.tool_selector_bridge as bridge_module
        bridge_module._tool_selector_bridge = None
        
        bridge = get_tool_selector_bridge()
        
        assert bridge is not None
        assert bridge.tools_directory is None
    
    def test_get_tool_selector_bridge_existing_instance(self):
        """Test getting bridge when instance already exists"""
        # Clear and set existing instance
        import src.mcp_server.tool_selector_bridge as bridge_module
        
        existing_bridge = ToolSelectorBridge("/existing/tools")
        bridge_module._tool_selector_bridge = existing_bridge
        
        # Should return existing instance
        bridge = get_tool_selector_bridge("/new/tools")
        
        assert bridge is existing_bridge
        assert bridge.tools_directory == "/existing/tools"


@pytest.mark.skipif(not TOOL_SELECTOR_BRIDGE_AVAILABLE, reason="Tool selector bridge not available")
class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bridge = ToolSelectorBridge()
        
        # Create comprehensive mock tool setup
        self.setup_comprehensive_tools()
    
    def setup_comprehensive_tools(self):
        """Setup comprehensive tool collection for testing"""
        self.comprehensive_tools = [
            MockTool("file_search", "Search through files", "search", 
                    ["search", "find", "file", "grep"], relevance_score=0.9),
            MockTool("data_analyzer", "Analyze datasets", "analysis", 
                    ["analyze", "data", "statistics"], relevance_score=0.8),
            MockTool("report_generator", "Generate reports", "creation", 
                    ["generate", "create", "report"], relevance_score=0.7),
            MockTool("file_converter", "Convert file formats", "conversion", 
                    ["convert", "transform", "format"], relevance_score=0.6),
            MockTool("backup_tool", "Backup data", "maintenance", 
                    ["backup", "save", "archive"], relevance_score=0.5),
            MockTool("config_validator", "Validate configurations", "validation", 
                    ["validate", "check", "config"], relevance_score=0.4),
            MockTool("log_parser", "Parse log files", "analysis", 
                    ["parse", "log", "analyze"], relevance_score=0.3),
            MockTool("network_scanner", "Scan network", "security", 
                    ["scan", "network", "security"], relevance_score=0.2)
        ]
    
    @patch('src.mcp_server.tool_selector_bridge.ToolSelector')
    async def test_comprehensive_tool_workflow(self, mock_tool_selector_class):
        """Test comprehensive tool workflow"""
        # Setup mock selector
        mock_selector = MagicMock()
        mock_selector.tools = self.comprehensive_tools
        mock_selector.get_all_tools.return_value = self.comprehensive_tools
        mock_selector.select_tools.return_value = self.comprehensive_tools[:3]
        mock_selector.get_tool_by_name.side_effect = lambda name: next(
            (tool for tool in self.comprehensive_tools if tool.name == name), None
        )
        mock_tool_selector_class.return_value = mock_selector
        
        # 1. List all tools
        list_result = await self.bridge.list_available_tools({})
        assert list_result["success"] is True
        assert list_result["total_count"] == 8
        
        # 2. Select tools with query
        select_result = await self.bridge.select_tools({
            "query": "search and analyze data",
            "max_tools": 3,
            "method": "hybrid"
        })
        assert select_result["success"] is True
        assert len(select_result["tools"]) == 3
        
        # 3. Get info for specific tool
        info_result = await self.bridge.get_tool_info({
            "tool_name": "file_search",
            "include_usage_examples": True
        })
        assert info_result["success"] is True
        assert info_result["tool"]["name"] == "file_search"
        
        # 4. Check bridge status
        status = self.bridge.get_status()
        assert status["initialized"] is True
        assert status["available"] is True
    
    @patch('src.mcp_server.tool_selector_bridge.ToolSelector')
    async def test_error_recovery_workflow(self, mock_tool_selector_class):
        """Test error recovery workflow"""
        # Setup mock selector that fails on first attempt, succeeds on second
        mock_selector = MagicMock()
        mock_selector.tools = self.comprehensive_tools
        
        # Make select_tools fail first time, succeed second time
        mock_selector.select_tools.side_effect = [
            Exception("Temporary failure"),
            self.comprehensive_tools[:2]
        ]
        mock_tool_selector_class.return_value = mock_selector
        
        # First attempt should fail
        result1 = await self.bridge.select_tools({
            "query": "search",
            "max_tools": 3
        })
        assert result1["success"] is False
        
        # Reset selector for second attempt
        mock_selector.select_tools.side_effect = None
        mock_selector.select_tools.return_value = self.comprehensive_tools[:2]
        
        # Second attempt should succeed
        result2 = await self.bridge.select_tools({
            "query": "search",
            "max_tools": 3
        })
        assert result2["success"] is True
        assert len(result2["tools"]) == 2
    
    async def test_concurrent_operations(self):
        """Test concurrent operations on bridge"""
        import asyncio
        
        # Setup mock selector
        with patch('src.mcp_server.tool_selector_bridge.ToolSelector') as mock_class:
            mock_selector = MagicMock()
            mock_selector.tools = self.comprehensive_tools
            mock_selector.get_all_tools.return_value = self.comprehensive_tools
            mock_selector.select_tools.return_value = self.comprehensive_tools[:3]
            mock_class.return_value = mock_selector
            
            # Run multiple operations concurrently
            tasks = [
                self.bridge.select_tools({"query": "search", "max_tools": 3}),
                self.bridge.list_available_tools({}),
                self.bridge.get_tool_info({"tool_name": "file_search"}),
                self.bridge.select_tools({"query": "analyze", "max_tools": 2}),
                self.bridge.list_available_tools({"sort_by": "category"})
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All operations should succeed
            for result in results:
                assert not isinstance(result, Exception)
                assert result["success"] is True
    
    @patch('src.mcp_server.tool_selector_bridge.ToolSelector')
    async def test_parameter_validation_edge_cases(self, mock_tool_selector_class):
        """Test parameter validation edge cases"""
        mock_selector = MagicMock()
        mock_selector.tools = self.comprehensive_tools
        mock_tool_selector_class.return_value = mock_selector
        
        # Test edge case parameters
        test_cases = [
            # Max tools at limits
            {"query": "test", "max_tools": 1},  # Minimum
            {"query": "test", "max_tools": 20},  # Maximum (with MockToolSelectorConfig)
            
            # Valid methods
            {"query": "test", "method": "dot_product"},
            {"query": "test", "method": "rule_based"},
            {"query": "test", "method": "hybrid"},
            
            # Category filters
            {"query": "test", "category_filter": "SEARCH"},  # Case insensitive
            {"query": "test", "category_filter": "analysis"},
            
            # Score inclusion
            {"query": "test", "include_scores": True},
            {"query": "test", "include_scores": False}
        ]
        
        with patch('src.mcp_server.tool_selector_bridge._tool_selector_module') as mock_module:
            mock_module.ToolSelectorConfig = MockToolSelectorConfig
            
            for test_params in test_cases:
                result = await self.bridge.select_tools(test_params)
                assert result["success"] is True, f"Failed for params: {test_params}"