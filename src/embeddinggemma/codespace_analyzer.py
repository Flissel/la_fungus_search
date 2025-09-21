"""
CodeSpace Analyzer Tool

English summary
----------------
Lightweight utility to scan a fixed root folder (default: ``src``), list
Python files, and search for code snippets by plain substring or a regex
pattern. Provides a small CLI for ad‑hoc analysis.
"""

import os
import argparse
import re
from typing import List, Dict, Tuple

class CodeSpaceAnalyzer:
    """Main entry class for the CodeSpace analyzer tool."""
    
    def __init__(self, target_folder: str = "src"):
        """Initialize the analyzer with a fixed folder.

        Args:
            target_folder: Folder to analyze (default: ``src``).
        """
        self.target_folder = target_folder
        self.files_info: List[Dict[str, any]] = []
    
    def scan_folder(self) -> List[str]:
        """Scan the target folder and list Python files.

        Returns:
            List[str]: Relative file paths below ``target_folder``.
        """
        python_files = []
        for root, dirs, files in os.walk(self.target_folder):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.target_folder)
                    self.files_info.append({
                        'path': file_path,
                        'relative_path': relative_path,
                        'size': os.path.getsize(file_path)
                    })
                    python_files.append(relative_path)
        return python_files
    
    def read_file_content(self, file_path: str) -> str:
        """Read and return the file content as a string."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
    def search_snippets(self, file_path: str, query: str, num_lines: int = 5) -> List[Tuple[int, str]]:
        """Search a file for a query and return context snippets.

        Args:
            file_path: Path to the file.
            query: Substring to search or a regex pattern delimited as ``/.../``.
            num_lines: Number of context lines surrounding each match.

        Returns:
            List of tuples ``(line_number, snippet_text)``.
        """
        content = self.read_file_content(file_path)
        if not content:
            return []
        
        lines = content.splitlines()
        snippets = []
        query_lower = query.lower()
        is_regex = query.startswith('/') and query.endswith('/')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if is_regex:
                pattern = query[1:-1]
                if re.search(pattern, line):
                    start = max(0, i - num_lines)
                    end = min(len(lines), i + num_lines + 1)
                    snippet = '\n'.join(lines[start:end])
                    snippets.append((i + 1, snippet))
            else:
                if query_lower in line_lower:
                    start = max(0, i - num_lines)
                    end = min(len(lines), i + num_lines + 1)
                    snippet = '\n'.join(lines[start:end])
                    snippets.append((i + 1, snippet))
        
        return snippets
    
    def analyze_script(self, script_name: str, query: str) -> Dict[str, List[Tuple[int, str]]]:
        """Analyze a single script by name relative to ``target_folder``."""
        results = {}
        # Search recursively for the filename
        for root, dirs, files in os.walk(self.target_folder):
            if script_name in files:
                file_path = os.path.join(root, script_name)
                relative_path = os.path.relpath(file_path, self.target_folder)
                snippets = self.search_snippets(file_path, query)
                results[relative_path] = snippets
                break  # first match
        
        if not results:
            print(f"File '{script_name}' not found in {self.target_folder}.")
        
        return results

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CodeSpace Analyzer Tool")
    parser.add_argument("script", help="Script name (e.g., app.py)")
    parser.add_argument("query", help="Search term or regex (/.../)")
    parser.add_argument("--folder", default="src", help="Target folder (default: src)")
    args = parser.parse_args()
    
    analyzer = CodeSpaceAnalyzer(args.folder)
    results = analyzer.analyze_script(args.script, args.query)
    
    # Print results
    if results is None or not results:
        print("No results found.")
    else:
        for file_path, snippets in results.items():
            print(f"\nFile: {file_path}")
            if snippets:
                for line_num, snippet in snippets:
                    print(f"Line {line_num}:\n{snippet}\n")
            else:
                print("No matching snippets.")

if __name__ == "__main__":
    main()