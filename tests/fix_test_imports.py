#!/usr/bin/env python3
"""
Fix Test Imports Script
========================

This script systematically fixes import statements in test files to use the 'src.' prefix,
resolving the "attempted relative import beyond top-level package" errors in CI.

Root cause: Tests import modules directly (e.g., 'from storage.module import X') 
but those modules use relative imports that fail outside package context.

Solution: Convert all test imports to use 'src.' prefix (e.g., 'from src.storage.module import X')
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any


def find_problematic_imports(file_path: Path) -> List[Dict[str, Any]]:
    """Find all imports that need the src. prefix."""
    problematic_patterns = [
        r'^(\s*)from (storage|core|monitoring|mcp_server|validators|api|interfaces)\.([^\s]+) import (.+)$',
        r'^(\s*)import (storage|core|monitoring|mcp_server|validators|api|interfaces)\.([^\s]+)(.*)$'
    ]
    
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return issues
    
    for line_num, line in enumerate(lines, 1):
        for pattern in problematic_patterns:
            match = re.match(pattern, line)
            if match:
                issues.append({
                    'line_num': line_num,
                    'original_line': line.rstrip(),
                    'pattern_match': match,
                    'line_index': line_num - 1
                })
    
    return issues


def fix_import_line(line: str) -> str:
    """Fix a single import line by adding src. prefix."""
    # Pattern 1: from module.submodule import X
    pattern1 = r'^(\s*)from (storage|core|monitoring|mcp_server|validators|api|interfaces)\.([^\s]+) import (.+)$'
    match1 = re.match(pattern1, line)
    if match1:
        indent, module, submodule, imports = match1.groups()
        return f"{indent}from src.{module}.{submodule} import {imports}"
    
    # Pattern 2: import module.submodule
    pattern2 = r'^(\s*)import (storage|core|monitoring|mcp_server|validators|api|interfaces)\.([^\s]+)(.*)$'
    match2 = re.match(pattern2, line)
    if match2:
        indent, module, submodule, rest = match2.groups()
        return f"{indent}import src.{module}.{submodule}{rest}"
    
    return line


def fix_file_imports(file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Fix all imports in a single file."""
    try:
        relative_path = file_path.relative_to(Path.cwd())
    except ValueError:
        relative_path = file_path
    print(f"\n📁 Processing: {relative_path}")
    
    issues = find_problematic_imports(file_path)
    if not issues:
        print("   ✅ No import issues found")
        return {'fixed': 0, 'issues': []}
    
    print(f"   🔍 Found {len(issues)} import issues")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return {'fixed': 0, 'issues': [], 'error': str(e)}
    
    fixes_made = 0
    for issue in issues:
        line_index = issue['line_index']
        original_line = issue['original_line']
        fixed_line = fix_import_line(original_line)
        
        if fixed_line != original_line:
            print(f"   Line {issue['line_num']:3}: {original_line}")
            print(f"   Fixed to:    {fixed_line}")
            
            if not dry_run:
                lines[line_index] = fixed_line + '\n'
            fixes_made += 1
    
    if not dry_run and fixes_made > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"   ✅ Applied {fixes_made} fixes to file")
        except Exception as e:
            print(f"   ❌ Error writing file: {e}")
            return {'fixed': 0, 'issues': issues, 'error': str(e)}
    
    return {'fixed': fixes_made, 'issues': issues}


def find_test_files() -> List[Path]:
    """Find all Python test files."""
    test_files = []
    tests_dir = Path('tests')
    
    if not tests_dir.exists():
        print("❌ tests/ directory not found")
        return test_files
    
    for py_file in tests_dir.rglob('*.py'):
        if py_file.name != '__init__.py':  # Skip __init__.py files
            test_files.append(py_file)
    
    return sorted(test_files)


def main():
    """Main execution function."""
    print("🔧 Test Import Fixer")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('src').exists():
        print("❌ src/ directory not found. Run this script from the project root.")
        sys.exit(1)
    
    # Find all test files
    test_files = find_test_files()
    print(f"📊 Found {len(test_files)} test files")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        print("🔍 DRY RUN MODE - No changes will be made")
        dry_run = True
    else:
        print("✏️  FIXING MODE - Changes will be applied")
        dry_run = False
    
    total_files_fixed = 0
    total_imports_fixed = 0
    files_with_errors = []
    
    # Process each file
    for file_path in test_files:
        result = fix_file_imports(file_path, dry_run)
        
        if 'error' in result:
            files_with_errors.append(file_path)
        elif result['fixed'] > 0:
            total_files_fixed += 1
            total_imports_fixed += result['fixed']
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print(f"   Files processed: {len(test_files)}")
    print(f"   Files with fixes: {total_files_fixed}")
    print(f"   Total imports fixed: {total_imports_fixed}")
    
    if files_with_errors:
        print(f"   Files with errors: {len(files_with_errors)}")
        for error_file in files_with_errors:
            print(f"     - {error_file}")
    
    if total_imports_fixed > 0:
        if dry_run:
            print(f"\n🔍 DRY RUN: Would fix {total_imports_fixed} imports in {total_files_fixed} files")
            print("   Run without --dry-run to apply fixes")
        else:
            print(f"\n✅ Successfully fixed {total_imports_fixed} imports in {total_files_fixed} files")
            print("   You can now run tests with: ./scripts/run-tests.sh ci")
    else:
        print("\n✅ No import issues found - all test imports are already correct!")


if __name__ == '__main__':
    main()