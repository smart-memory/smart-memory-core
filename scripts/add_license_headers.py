#!/usr/bin/env python3
"""
Script to add AGPL v3 license headers with dual licensing notice to all Python files.
Run this from the project root directory.
"""

import os
from typing import List

LICENSE_HEADER = '''# Copyright (C) 2025 SmartMemory
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# For commercial licensing options, please contact: help@smartmemory.ai
# Commercial licenses are available for organizations that wish to use
# this software in proprietary applications without the AGPL restrictions.

'''

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in directory and subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common build/cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def has_license_header(file_path: str) -> bool:
    """Check if file already has a license header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Check first 1000 characters
            return 'Copyright (C) 2025 SmartMemory' in content or 'GNU Affero General Public License' in content
    except Exception:
        return False

def add_license_header(file_path: str) -> bool:
    """Add license header to Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Handle shebang line
        lines = original_content.split('\n')
        if lines and lines[0].startswith('#!'):
            # Keep shebang, add license after it
            shebang = lines[0] + '\n'
            rest_content = '\n'.join(lines[1:])
            new_content = shebang + LICENSE_HEADER + rest_content
        else:
            # Add license at the beginning
            new_content = LICENSE_HEADER + original_content
        
        # Write back the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all Python files."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Find all Python files
    python_files = find_python_files(project_root)
    
    # Filter out this script itself and other utility scripts you might not want to modify
    exclude_files = [
        os.path.join(project_root, 'add_license_headers.py'),
        # Add other files to exclude here if needed
    ]
    
    python_files = [f for f in python_files if f not in exclude_files]
    
    print(f"Found {len(python_files)} Python files")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for file_path in python_files:
        relative_path = os.path.relpath(file_path, project_root)
        
        if has_license_header(file_path):
            print(f"SKIP: {relative_path} (already has license header)")
            skipped += 1
            continue
        
        if add_license_header(file_path):
            print(f"ADD:  {relative_path}")
            processed += 1
        else:
            print(f"ERR:  {relative_path}")
            errors += 1
    
    print("\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total files: {len(python_files)}")

if __name__ == "__main__":
    main()