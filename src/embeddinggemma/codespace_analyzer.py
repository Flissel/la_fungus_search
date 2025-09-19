"""
CodeSpace Analyzer Tool
Ermöglicht die Analyse von Dateien in einem festen Ordner und Query-basierte Suche nach Code-Snippets.
"""

import os
import argparse
import re
from typing import List, Dict, Tuple

class CodeSpaceAnalyzer:
    """
    Hauptklasse für das CodeSpace-Analyse-Tool.
    """
    
    def __init__(self, target_folder: str = "src"):
        """
        Initialisiert den Analyzer mit dem festen Ordner.
        
        Args:
            target_folder (str): Der zu analysierende Ordner (default: 'src').
        """
        self.target_folder = target_folder
        self.files_info: List[Dict[str, any]] = []
    
    def scan_folder(self) -> List[str]:
        """
        Scannt den festen Ordner und listet alle Python-Dateien auf.
        
        Returns:
            List[str]: Liste der Dateipfade.
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
        """
        Liest den Inhalt einer Datei.
        
        Args:
            file_path (str): Pfad zur Datei.
            
        Returns:
            str: Datei-Inhalt als String.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Fehler beim Lesen der Datei {file_path}: {e}")
            return ""
    
    def search_snippets(self, file_path: str, query: str, num_lines: int = 5) -> List[Tuple[int, str]]:
        """
        Sucht nach Code-Snippets in einer Datei basierend auf einer Query.
        
        Args:
            file_path (str): Pfad zur Datei.
            query (str): Suchbegriff oder Regex-Pattern.
            num_lines (int): Anzahl der Kontext-Zeilen um das Match.
            
        Returns:
            List[Tuple[int, str]]: Liste von (Zeilennummer, Snippet)-Tuples.
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
        """
        Analysiert ein spezifisches Script mit einer Query.
        
        Args:
            script_name (str): Name des Scripts (relativ zum target_folder).
            query (str): Suchbegriff.
            
        Returns:
            Dict[str, List[Tuple[int, str]]]: Ergebnisse der Suche.
        """
        results = {}
        # Suche rekursiv für den Dateinamen
        for root, dirs, files in os.walk(self.target_folder):
            if script_name in files:
                file_path = os.path.join(root, script_name)
                relative_path = os.path.relpath(file_path, self.target_folder)
                snippets = self.search_snippets(file_path, query)
                results[relative_path] = snippets
                break  # Erste Übereinstimmung
        
        if not results:
            print(f"Datei '{script_name}' nicht in {self.target_folder} gefunden.")
        
        return results

def main():
    """
    CLI-Einstiegspunkt.
    """
    parser = argparse.ArgumentParser(description="CodeSpace Analyzer Tool")
    parser.add_argument("script", help="Name des Scripts (z.B. app.py)")
    parser.add_argument("query", help="Suchbegriff oder Query")
    parser.add_argument("--folder", default="src", help="Zielordner (default: src)")
    args = parser.parse_args()
    
    analyzer = CodeSpaceAnalyzer(args.folder)
    results = analyzer.analyze_script(args.script, args.query)
    
    # Ausgabe der Ergebnisse
    if results is None or not results:
        print("Keine Ergebnisse gefunden.")
    else:
        for file_path, snippets in results.items():
            print(f"\nDatei: {file_path}")
            if snippets:
                for line_num, snippet in snippets:
                    print(f"Zeile {line_num}:\n{snippet}\n")
            else:
                print("Keine passenden Snippets gefunden.")

if __name__ == "__main__":
    main()