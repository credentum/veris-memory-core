#!/usr/bin/env python3
"""
Tool Selector Bridge for MCP Server Integration.

This module provides a bridge between the MCP server and the tool selector module,
allowing tool selection functionality to be exposed via MCP endpoints.
"""

import importlib.util
import logging

# Secure module import using relative path resolution instead of sys.path manipulation
from pathlib import Path
from typing import Any, Dict, List, Optional

# Store repository root for later use
_REPO_ROOT = None
_TOOL_SELECTOR_MODULE = None


def _load_tool_selector_module():
    """
    Safely load the tool selector module using importlib instead of sys.path manipulation.

    Returns:
        The loaded ToolSelector class or None if loading failed
    """
    global _REPO_ROOT, _TOOL_SELECTOR_MODULE

    if _TOOL_SELECTOR_MODULE is not None:
        return _TOOL_SELECTOR_MODULE

    try:
        # Get repository root using pathlib resolve (canonical path with symlink resolution)
        repo_root = Path(__file__).resolve().parent.parent.parent.parent

        # Define expected base directory for security validation
        expected_base = Path(__file__).resolve().parent.parent.parent.parent.parent

        # Simple containment check using is_relative_to
        if not repo_root.is_relative_to(expected_base):
            logging.getLogger(__name__).warning(
                f"Path validation failed - outside expected boundaries: {repo_root}"
            )
            return None

        # Basic existence and type validation
        if not repo_root.exists() or not repo_root.is_dir():
            return None

        # Verify this looks like a repository with context/tools directory
        context_tools_dir = repo_root / "context" / "tools"
        if not context_tools_dir.exists():
            return None

        # Store the repo root for later use
        _REPO_ROOT = str(repo_root)

        # Load the tool selector module using importlib
        tool_selector_path = context_tools_dir / "tool_selector.py"
        if not tool_selector_path.exists():
            logging.getLogger(__name__).error(
                f"Tool selector module not found at {tool_selector_path}"
            )
            return None

        # Load module using importlib.util
        spec = importlib.util.spec_from_file_location("tool_selector", tool_selector_path)
        if spec is None or spec.loader is None:
            logging.getLogger(__name__).error("Failed to create module spec for tool_selector")
            return None

        tool_selector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool_selector_module)

        # Cache the loaded module
        _TOOL_SELECTOR_MODULE = tool_selector_module
        return tool_selector_module

    except (OSError, ValueError, AttributeError, PermissionError) as e:
        # Log the error but don't expose details
        logging.getLogger(__name__).warning(
            f"Failed to load tool selector module: {type(e).__name__}"
        )
        return None
    except ImportError as e:
        logging.getLogger(__name__).error(f"Import error loading tool selector: {e}")
        return None


# Load the tool selector module safely
_tool_selector_module = _load_tool_selector_module()
if _tool_selector_module is None:
    raise ImportError(
        "Unable to safely locate and load tool selector module - "
        "repository structure validation failed"
    )

# Get the ToolSelector class from the loaded module
ToolSelector = getattr(_tool_selector_module, "ToolSelector", None)
if ToolSelector is None:
    raise ImportError("ToolSelector class not found in tool selector module")

logger = logging.getLogger(__name__)


class ToolSelectorBridge:
    """
    Bridge class for integrating tool selector functionality with MCP server.

    This class provides MCP-compatible interfaces for tool selection operations,
    handling initialization, error management, and response formatting.
    """

    def __init__(self, tools_directory: Optional[str] = None):
        """
        Initialize the tool selector bridge.

        Args:
            tools_directory: Path to directory containing tool documentation.
                           If None, uses the default context/tools/ directory.
        """
        self.tools_directory = tools_directory
        self._selector: Optional[ToolSelector] = None
        self._initialized = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _ensure_initialized(self) -> bool:
        """
        Ensure the tool selector is initialized.

        Returns:
            bool: True if initialization successful, False otherwise.
        """
        if self._initialized and self._selector:
            return True

        try:
            if self.tools_directory:
                self._selector = ToolSelector(self.tools_directory)
            else:
                # Use default path relative to repository root
                if _REPO_ROOT:
                    default_tools_dir = Path(_REPO_ROOT) / "context" / "tools"
                else:
                    # Fallback to current directory structure
                    default_tools_dir = (
                        Path(__file__).parent.parent.parent.parent / "context" / "tools"
                    )
                self._selector = ToolSelector(default_tools_dir)

            self._initialized = True
            if self._selector:
                self.logger.info(
                    f"Tool selector initialized with {len(self._selector.tools)} tools"
                )
            return True

        except (OSError, ValueError, TypeError) as e:
            self.logger.error(f"Data processing error initializing tool selector: {e}")
            self._initialized = False
            return False
        except (ImportError, AttributeError, RuntimeError) as e:
            self.logger.error(
                f"Module or initialization error in tool selector: {type(e).__name__}: {e}"
            )
            self._initialized = False
            return False

    def _get_selector(self) -> ToolSelector:
        """Get the tool selector, ensuring it's initialized."""
        if not self._ensure_initialized() or self._selector is None:
            raise RuntimeError("Tool selector not properly initialized")
        return self._selector

    async def select_tools(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select relevant tools based on query.

        Args:
            arguments: Dictionary containing:
                - query (str): Search query or context description (required)
                - max_tools (int, optional): Maximum number of tools to return (default: 5)
                - method (str, optional): Selection method (default: "hybrid")
                - category_filter (str, optional): Filter by category
                - include_scores (bool, optional): Include relevance scores (default: True)

        Returns:
            Dict containing:
                - success (bool): Whether the operation succeeded
                - tools (list): Selected tools with metadata
                - total_available (int): Total number of tools available
                - method_used (str): Selection method used
                - selection_log (dict): YAML-formatted selection log
                - message (str): Success or error message
        """
        try:
            # Ensure selector is initialized
            if not self._ensure_initialized():
                return {
                    "success": False,
                    "tools": [],
                    "total_available": 0,
                    "message": "Tool selector initialization failed",
                    "error_type": "initialization_error",
                }

            # Get selector to ensure it's available for error responses
            selector = self._get_selector()

            # Extract parameters
            query = arguments.get("query", "")
            if not query:
                return {
                    "success": False,
                    "tools": [],
                    "total_available": len(selector.tools),
                    "message": "Query parameter is required",
                    "error_type": "missing_parameter",
                }

            max_tools = arguments.get("max_tools", 5)
            method = arguments.get("method", "hybrid")
            category_filter = arguments.get("category_filter")
            include_scores = arguments.get("include_scores", True)

            # Validate parameters (import config from loaded module)
            ToolSelectorConfig = getattr(_tool_selector_module, "ToolSelectorConfig", None)
            max_tools_limit = ToolSelectorConfig.MAX_TOOLS_LIMIT if ToolSelectorConfig else 20

            if max_tools < 1 or max_tools > max_tools_limit:
                return {
                    "success": False,
                    "tools": [],
                    "total_available": len(selector.tools),
                    "message": f"max_tools must be between 1 and {max_tools_limit}",
                    "error_type": "invalid_parameter",
                }

            if method not in ["dot_product", "rule_based", "hybrid"]:
                return {
                    "success": False,
                    "tools": [],
                    "total_available": len(selector.tools),
                    "message": (
                        f"Invalid method: {method}. "
                        "Must be one of: dot_product, rule_based, hybrid"
                    ),
                    "error_type": "invalid_parameter",
                }

            # Perform tool selection
            selected_tools = selector.select_tools(query=query, max_tools=max_tools, method=method)

            # Apply category filter if specified
            if category_filter:
                selected_tools = [
                    tool
                    for tool in selected_tools
                    if tool.category.lower() == category_filter.lower()
                ]

            # Format response
            tools_response = []
            for tool in selected_tools:
                tool_dict = {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "keywords": tool.keywords,
                    "metadata": tool.metadata or {},
                }

                if include_scores:
                    tool_dict["relevance_score"] = tool.relevance_score

                tools_response.append(tool_dict)

            # Create selection log for audit trail
            selection_log = {
                "tool_selection": {
                    "query": query,
                    "method": method,
                    "max_tools_requested": max_tools,
                    "tools_selected": len(tools_response),
                    "total_available": len(selector.tools),
                    "category_filter": category_filter,
                    "include_scores": include_scores,
                }
            }

            return {
                "success": True,
                "tools": tools_response,
                "total_available": len(selector.tools),
                "method_used": method,
                "selection_log": selection_log,
                "message": f"Selected {len(tools_response)} tools using {method} method",
            }

        except Exception as e:
            return self._handle_async_error(
                error=e,
                operation="tool selection",
                default_response={
                    "tools": [],
                    "total_available": 0,
                },
            )

    async def list_available_tools(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        List all available tools in the system.

        Args:
            arguments: Dictionary containing:
                - category_filter (str, optional): Filter by category
                - include_metadata (bool, optional): Include metadata (default: True)
                - sort_by (str, optional): Sort by field (default: "name")

        Returns:
            Dict containing:
                - success (bool): Whether the operation succeeded
                - tools (list): List of available tools
                - total_count (int): Total number of tools
                - categories (list): Available categories
                - message (str): Success or error message
        """
        try:
            # Ensure selector is initialized
            if not self._ensure_initialized():
                return {
                    "success": False,
                    "tools": [],
                    "total_count": 0,
                    "categories": [],
                    "message": "Tool selector initialization failed",
                    "error_type": "initialization_error",
                }

            # Extract parameters
            category_filter = arguments.get("category_filter")
            include_metadata = arguments.get("include_metadata", True)
            sort_by = arguments.get("sort_by", "name")

            # Get all tools
            selector = self._get_selector()
            all_tools = selector.get_all_tools()

            # Apply category filter if specified
            if category_filter:
                all_tools = [
                    tool for tool in all_tools if tool.category.lower() == category_filter.lower()
                ]

            # Sort tools
            if sort_by == "name":
                all_tools.sort(key=lambda t: t.name.lower())
            elif sort_by == "category":
                all_tools.sort(key=lambda t: (t.category.lower(), t.name.lower()))
            elif sort_by == "keywords":
                all_tools.sort(key=lambda t: len(t.keywords), reverse=True)

            # Format response
            tools_response = []
            categories = set()

            for tool in all_tools:
                categories.add(tool.category)

                tool_dict = {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category,
                    "keywords": tool.keywords,
                }

                if include_metadata:
                    tool_dict["metadata"] = tool.metadata or {}

                tools_response.append(tool_dict)

            return {
                "success": True,
                "tools": tools_response,
                "total_count": len(tools_response),
                "categories": sorted(list(categories)),
                "message": f"Listed {len(tools_response)} available tools",
            }

        except Exception as e:
            return self._handle_async_error(
                error=e,
                operation="tool listing",
                default_response={
                    "tools": [],
                    "total_count": 0,
                    "categories": [],
                },
            )

    async def get_tool_info(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.

        Args:
            arguments: Dictionary containing:
                - tool_name (str): Name of the tool to get info about (required)
                - include_usage_examples (bool, optional): Include usage examples (default: True)

        Returns:
            Dict containing:
                - success (bool): Whether the tool was found
                - tool (dict, optional): Detailed tool information
                - message (str): Success or error message
        """
        try:
            # Ensure selector is initialized
            if not self._ensure_initialized():
                return {
                    "success": False,
                    "message": "Tool selector initialization failed",
                    "error_type": "initialization_error",
                }

            # Extract parameters
            tool_name = arguments.get("tool_name", "")
            if not tool_name:
                return {
                    "success": False,
                    "message": "tool_name parameter is required",
                    "error_type": "missing_parameter",
                }

            include_usage_examples = arguments.get("include_usage_examples", True)

            # Find the tool
            selector = self._get_selector()
            tool = selector.get_tool_by_name(tool_name)
            if not tool:
                return {
                    "success": False,
                    "message": f"Tool '{tool_name}' not found",
                    "error_type": "tool_not_found",
                }

            # Format detailed response
            tool_info = {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "keywords": tool.keywords,
                "metadata": tool.metadata or {},
            }

            # Add documentation and usage examples if available
            if tool.metadata and "file_path" in tool.metadata:
                try:
                    file_path = Path(tool.metadata["file_path"])
                    if file_path.exists():
                        content = file_path.read_text()
                        tool_info["documentation"] = content

                        # Extract usage examples if requested
                        if include_usage_examples:
                            usage_examples = self._extract_usage_examples(content)
                            if usage_examples:
                                tool_info["usage_examples"] = usage_examples
                except (OSError, UnicodeDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to read documentation for {tool_name}: {e}")
                except Exception as e:
                    self.logger.warning(
                        f"Unexpected error reading documentation for {tool_name}: "
                        f"{type(e).__name__}: {e}"
                    )

            return {
                "success": True,
                "tool": tool_info,
                "message": "Tool information retrieved successfully",
            }

        except Exception as e:
            return self._handle_async_error(
                error=e, operation="tool info retrieval", default_response={}
            )

    def _handle_async_error(
        self, error: Exception, operation: str, default_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Standardized async error handling for all bridge methods.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            default_response: Default response structure with success=False

        Returns:
            Standardized error response dictionary
        """
        if isinstance(error, (AttributeError, KeyError, ValueError, TypeError)):
            self.logger.error(f"Data processing error in {operation}: {error}")
            error_type = "data_error"
        elif isinstance(error, (OSError, IOError, PermissionError)):
            self.logger.error(f"File system error in {operation}: {error}")
            error_type = "filesystem_error"
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            self.logger.error(f"Module import error in {operation}: {error}")
            error_type = "import_error"
        elif isinstance(error, RuntimeError):
            self.logger.error(f"Runtime error in {operation}: {error}")
            error_type = "runtime_error"
        else:
            # Log the specific error type for debugging
            self.logger.error(
                f"Unhandled error type {type(error).__name__} in {operation}: {error}"
            )
            error_type = "unknown_error"

        response = default_response.copy()
        response.update(
            {
                "success": False,
                "message": f"{operation.title()} failed: {str(error)}",
                "error_type": error_type,
            }
        )
        return response

    def _handle_sync_error(
        self, error: Exception, operation: str, default_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Standardized sync error handling for all bridge methods.

        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            default_response: Default response structure with success=False

        Returns:
            Standardized error response dictionary
        """
        # Use the same error categorization as async handler for consistency
        if isinstance(error, (AttributeError, KeyError, ValueError, TypeError)):
            self.logger.error(f"Data processing error in {operation}: {error}")
            error_type = "data_error"
        elif isinstance(error, (OSError, IOError, PermissionError)):
            self.logger.error(f"File system error in {operation}: {error}")
            error_type = "filesystem_error"
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            self.logger.error(f"Module import error in {operation}: {error}")
            error_type = "import_error"
        elif isinstance(error, RuntimeError):
            self.logger.error(f"Runtime error in {operation}: {error}")
            error_type = "runtime_error"
        else:
            # Log the specific error type for debugging
            self.logger.error(
                f"Unhandled error type {type(error).__name__} in {operation}: {error}"
            )
            error_type = "unknown_error"

        response = default_response.copy()
        response.update(
            {
                "success": False,
                "message": f"{operation.title()} failed: {str(error)}",
                "error_type": error_type,
            }
        )
        return response

    def _extract_usage_examples(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Extract usage examples from markdown content.

        Looks for code blocks under "Usage", "Examples", or "Example" sections.

        Args:
            markdown_content: The markdown content to parse

        Returns:
            List of usage example dictionaries with 'description' and 'code' keys
        """
        import re

        examples = []

        # Find usage/examples sections
        usage_pattern = (
            r"(?i)#{1,3}\s*(usage|examples?|how to use|getting started).*?\n(.*?)(?=\n#{1,3}|\Z)"
        )
        usage_matches = re.findall(usage_pattern, markdown_content, re.DOTALL)

        for _, section_content in usage_matches:
            # Extract code blocks from the section
            code_block_pattern = r"```(\w+)?\n(.*?)\n```"
            code_matches = re.findall(code_block_pattern, section_content, re.DOTALL)

            for language, code in code_matches:
                # Look for description before code block
                lines = section_content.split("\n")
                description = ""

                for i, line in enumerate(lines):
                    if f"```{language}" in line or "```" in line:
                        # Get previous non-empty line as description
                        for j in range(i - 1, -1, -1):
                            if lines[j].strip() and not lines[j].startswith("#"):
                                description = lines[j].strip()
                                break
                        break

                if not description:
                    description = f"Example usage{' in ' + language if language else ''}"

                examples.append(
                    {
                        "description": description,
                        "language": language or "text",
                        "code": code.strip(),
                    }
                )

        # Also look for inline code examples with specific markers
        inline_pattern = r"(?i)example:?\s*`([^`]+)`"
        inline_matches = re.findall(inline_pattern, markdown_content)

        for code in inline_matches:
            examples.append(
                {"description": "Inline example", "language": "text", "code": code.strip()}
            )

        return examples

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the tool selector bridge.

        Returns:
            Dict containing bridge status information.
        """
        try:
            return {
                "initialized": self._initialized,
                "tools_loaded": len(self._selector.tools) if self._selector else 0,
                "tools_directory": str(self.tools_directory) if self.tools_directory else "default",
                "available": self._initialized and self._selector is not None,
            }
        except Exception as e:
            return self._handle_sync_error(
                error=e,
                operation="status retrieval",
                default_response={
                    "initialized": False,
                    "tools_loaded": 0,
                    "tools_directory": "error",
                    "available": False,
                },
            )


# Global bridge instance
_tool_selector_bridge = None


def get_tool_selector_bridge(tools_directory: Optional[str] = None) -> ToolSelectorBridge:
    """
    Get or create the global tool selector bridge instance.

    Args:
        tools_directory: Path to tools directory (used only on first creation)

    Returns:
        ToolSelectorBridge: The global bridge instance
    """
    global _tool_selector_bridge

    if _tool_selector_bridge is None:
        _tool_selector_bridge = ToolSelectorBridge(tools_directory)

    return _tool_selector_bridge
