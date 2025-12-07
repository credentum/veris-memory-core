# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Cross-team memory sharing** (Issue #2): Added `shared` flag to enable contexts to be visible across teams
  - `store_context`: New `shared: bool` parameter (default: false) - when true, context is visible to all teams
  - `retrieve_context`: New `include_shared: bool` parameter (default: true) - controls whether to include shared contexts
  - Research team can now publish findings for Herald team to see without sharing API keys
  - Teams maintain isolated private spaces while sharing selected contexts
- **Scratchpad namespace isolation**: Scratchpads now filtered by API key user_id
  - `update_scratchpad`: New `shared: bool` parameter (default: false) - when true, scratchpad is visible to all teams
  - `list_scratchpads`: Now filters by API key user_id namespace. New `include_shared: bool` parameter (default: true)
  - Metadata stored in parallel Redis key (`scratchpad_meta:{agent_id}:{key}`) for filtering
  - Backwards compatible: legacy scratchpads without metadata are still visible
- Initial project structure
- Python project configuration with pyproject.toml
- TypeScript project configuration
- Docker Compose setup for development
- GitHub Actions CI/CD pipeline
- Pre-commit hooks configuration
- Basic documentation structure

### Changed

- N/A

### Deprecated

- N/A

### Removed

- N/A

### Fixed

- N/A

### Security

- N/A

## [0.1.0] - 2025-07-18

### Added

- Initial repository setup
- Basic project structure
- Configuration files for Python and TypeScript
- Docker development environment
- CI/CD pipeline templates
- Documentation framework
