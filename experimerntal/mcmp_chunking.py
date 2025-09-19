#!/usr/bin/env python3
"""
MCPM-Chunking: Monte Carlo Physarum Machine für intelligente Dokumenten-Segmentierung
Kombiniert adaptive Exploration mit strukturierter Extraktion
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import re
from pathlib import Path
import time

# Mock LangExtract - replace with actual implementation
class MockLangExtract:
    """Placeholder for LangExtract functionality"""
    
    def extract_structured_data(self, text: str, target_schema: Dict) -> Dict:
        """Mock structured extraction"""
        # This would be your actual LangExtract call
        return {
            "title": "Extracted Title",
            "summary": "Extracted summary from text...",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "metadata": {"confidence": 0.85, "word_count": len(text.split())}
        }

@dataclass
class ChunkTarget:
    """Zielstruktur für Chunk-Extraktion"""
    chunk_type: str  # "summary", "technical_detail", "example", "definition"
    schema: Dict[str, Any]
    priority: float = 1.0
    min_content_length: int = 50
    max_content_length: int = 1000
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)

@dataclass
class MCPMChunk:
    """Intelligenter Chunk mit MCMP-Metadaten"""
    id: str
    content: str
    chunk_type: str
    structured_data: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    # MCMP-spezifische Metadaten
    exploration_path: List[int] = field(default_factory=list)  # Welche Agenten fanden diesen Chunk
    relevance_score: float = 0.0
    connections: List[str] = field(default_factory=list)  # IDs verbundener Chunks
    extraction_confidence: float = 0.0
    
    # Speicher-Optimierung
    storage_priority: float = 0.0  # Wie wichtig für langfristige Speicherung
    access_frequency: int = 0
    last_accessed: float = field(default_factory=time.time)

@dataclass
class ExplorationAgent:
    """MCPM Agent für Dokument-Exploration"""
    id: int
    position: int  # Position im Dokument (Character-Index)
    target_types: List[str]  # Welche Chunk-Types dieser Agent sucht
    exploration_radius: int = 200  # Characters um aktuelle Position
    energy: float = 1.0
    found_chunks: List[str] = field(default_factory=list)
    specialization: str = "general"  # "technical", "summary", "examples"

class MCPMChunker:
    """Monte Carlo Physarum Machine für intelligente Dokumenten-Segmentierung"""
    
    def __init__(self, 
                 num_agents: int = 50,
                 max_iterations: int = 100,
                 lang_extract_model: Optional[Any] = None):
        
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        self.lang_extract = lang_extract_model or MockLangExtract()
        
        # Exploration State
        self.agents: List[ExplorationAgent] = []
        self.document_text: str = ""
        self.chunk_targets: List[ChunkTarget] = []
        self.discovered_chunks: List[MCPMChunk] = []
        
        # Pheromone Trails für Chunk-Verbindungen
        self.chunk_connections: Dict[Tuple[str, str], float] = {}
        
        # Performance Tracking
        self.exploration_stats = {
            "chunks_found": 0,
            "avg_confidence": 0.0,
            "coverage_percentage": 0.0,
            "connection_density": 0.0
        }
    
    def define_chunk_targets(self, targets: List[Dict[str, Any]]):
        """Definiere Zielstrukturen für Chunk-Extraktion"""
        self.chunk_targets = []
        
        for target_def in targets:
            target = ChunkTarget(
                chunk_type=target_def["chunk_type"],
                schema=target_def["schema"],
                priority=target_def.get("priority", 1.0),
                min_content_length=target_def.get("min_content_length", 50),
                max_content_length=target_def.get("max_content_length", 1000),
                required_fields=target_def.get("required_fields", []),
                optional_fields=target_def.get("optional_fields", [])
            )
            self.chunk_targets.append(target)
    
    def spawn_specialized_agents(self, document_text: str):
        """Spawn Agenten mit verschiedenen Spezialisierungen"""
        self.document_text = document_text
        doc_length = len(document_text)
        self.agents = []
        
        # Agent-Spezialisierungen
        specializations = ["technical", "summary", "examples", "definitions", "procedures"]
        
        for i in range(self.num_agents):
            # Zufällige Startposition
            start_pos = np.random.randint(0, max(1, doc_length - 100))
            
            # Spezialisierung zuweisen
            specialization = specializations[i % len(specializations)]
            
            # Relevante Chunk-Types für diese Spezialisierung
            target_types = self.get_relevant_chunk_types(specialization)
            
            agent = ExplorationAgent(
                id=i,
                position=start_pos,
                target_types=target_types,
                exploration_radius=np.random.randint(100, 300),
                specialization=specialization,
                energy=1.0
            )
            
            self.agents.append(agent)
    
    def get_relevant_chunk_types(self, specialization: str) -> List[str]:
        """Bestimme relevante Chunk-Types für Agent-Spezialisierung"""
        mapping = {
            "technical": ["technical_detail", "code_example", "specification"],
            "summary": ["summary", "overview", "conclusion"],
            "examples": ["example", "use_case", "demonstration"],
            "definitions": ["definition", "concept", "terminology"],
            "procedures": ["procedure", "tutorial", "step_by_step"]
        }
        
        available_types = [target.chunk_type for target in self.chunk_targets]
        relevant = [ct for ct in mapping.get(specialization, []) if ct in available_types]
        
        # Fallback: alle verfügbaren Types
        return relevant if relevant else available_types
    
    def explore_document_region(self, agent: ExplorationAgent) -> Optional[str]:
        """Agent exploriert Dokumentregion auf potentielle Chunks"""
        start_pos = max(0, agent.position - agent.exploration_radius)
        end_pos = min(len(self.document_text), agent.position + agent.exploration_radius)
        
        region_text = self.document_text[start_pos:end_pos]
        
        # Bewerte Region für verschiedene Chunk-Types
        best_match = None
        best_score = 0.0
        
        for chunk_type in agent.target_types:
            score = self.evaluate_region_for_chunk_type(region_text, chunk_type)
            if score > best_score and score > 0.5:  # Minimum threshold
                best_score = score
                best_match = chunk_type
        
        if best_match:
            return region_text
        
        return None
    
    def evaluate_region_for_chunk_type(self, text: str, chunk_type: str) -> float:
        """Bewerte wie gut eine Textregion für einen Chunk-Type geeignet ist"""
        
        # Einfache Heuristiken - in Realität komplexere Bewertung
        indicators = {
            "summary": ["zusammenfassung", "überblick", "fazit", "kurz gesagt"],
            "technical_detail": ["algorithmus", "implementation", "technisch", "spezifikation"],
            "example": ["beispiel", "zum beispiel", "angenommen", "stellen sie sich vor"],
            "definition": ["ist definiert als", "bedeutet", "verstehen wir", "definition"],
            "procedure": ["schritt", "zunächst", "dann", "als nächstes", "abschließend"]
        }
        
        text_lower = text.lower()
        chunk_indicators = indicators.get(chunk_type, [])
        
        matches = sum(1 for indicator in chunk_indicators if indicator in text_lower)
        score = matches / max(1, len(chunk_indicators))
        
        # Längen-Bonus
        target = next((t for t in self.chunk_targets if t.chunk_type == chunk_type), None)
        if target:
            length_score = 1.0 if target.min_content_length <= len(text) <= target.max_content_length else 0.5
            score *= length_score
        
        return min(1.0, score)
    
    def extract_structured_chunk(self, text: str, chunk_type: str) -> Optional[MCPMChunk]:
        """Extrahiere strukturierte Daten aus Textregion"""
        
        # Finde passende Zielstruktur
        target = next((t for t in self.chunk_targets if t.chunk_type == chunk_type), None)
        if not target:
            return None
        
        try:
            # LangExtract für strukturierte Extraktion
            structured_data = self.lang_extract.extract_structured_data(text, target.schema)
            
            # Validiere Required Fields
            missing_fields = [field for field in target.required_fields 
                            if field not in structured_data or not structured_data[field]]
            
            if missing_fields:
                print(f"Missing required fields for {chunk_type}: {missing_fields}")
                return None
            
            # Erstelle MCPM Chunk
            chunk_id = f"{chunk_type}_{int(time.time() * 1000)}_{np.random.randint(1000)}"
            
            chunk = MCPMChunk(
                id=chunk_id,
                content=text,
                chunk_type=chunk_type,
                structured_data=structured_data,
                extraction_confidence=structured_data.get("confidence", 0.5),
                storage_priority=target.priority
            )
            
            return chunk
            
        except Exception as e:
            print(f"Extraction failed for {chunk_type}: {e}")
            return None
    
    def move_agents(self):
        """Update Agent-Positionen basierend auf Exploration"""
        
        for agent in self.agents:
            # Exploration vs. Exploitation
            if np.random.random() < 0.3:  # 30% Exploration
                # Random movement
                movement = np.random.randint(-200, 200)
            else:
                # Move towards promising regions
                movement = self.calculate_attraction_movement(agent)
            
            # Update Position
            new_pos = agent.position + movement
            agent.position = max(0, min(len(self.document_text) - 1, new_pos))
            
            # Energy decay
            agent.energy *= 0.99
    
    def calculate_attraction_movement(self, agent: ExplorationAgent) -> int:
        """Berechne Bewegungsrichtung basierend auf Pheromone und Targets"""
        
        # Vereinfachte Implementierung - in Realität komplexere Logik
        doc_center = len(self.document_text) // 2
        
        if agent.position < doc_center:
            return np.random.randint(50, 150)  # Move towards center/end
        else:
            return np.random.randint(-150, -50)  # Move towards beginning
    
    def discover_chunk_connections(self):
        """Entdecke Verbindungen zwischen Chunks"""
        
        for i, chunk_a in enumerate(self.discovered_chunks):
            for chunk_b in self.discovered_chunks[i+1:]:
                
                # Berechne Verbindungsstärke
                connection_strength = self.calculate_chunk_similarity(chunk_a, chunk_b)
                
                if connection_strength > 0.3:  # Threshold für Verbindung
                    connection_key = tuple(sorted([chunk_a.id, chunk_b.id]))
                    self.chunk_connections[connection_key] = connection_strength
                    
                    # Update Chunk-Verbindungen
                    chunk_a.connections.append(chunk_b.id)
                    chunk_b.connections.append(chunk_a.id)
    
    def calculate_chunk_similarity(self, chunk_a: MCPMChunk, chunk_b: MCPMChunk) -> float:
        """Berechne Ähnlichkeit zwischen zwei Chunks"""
        
        # Content-based similarity (vereinfacht)
        common_words = set(chunk_a.content.lower().split()) & set(chunk_b.content.lower().split())
        total_words = set(chunk_a.content.lower().split()) | set(chunk_b.content.lower().split())
        
        content_similarity = len(common_words) / max(1, len(total_words))
        
        # Type-based bonus
        type_bonus = 0.2 if chunk_a.chunk_type == chunk_b.chunk_type else 0.0
        
        # Structured data similarity
        struct_similarity = self.compare_structured_data(
            chunk_a.structured_data, 
            chunk_b.structured_data
        )
        
        return (content_similarity + type_bonus + struct_similarity) / 3
    
    def compare_structured_data(self, data_a: Dict, data_b: Dict) -> float:
        """Vergleiche strukturierte Daten zwischen Chunks"""
        
        common_keys = set(data_a.keys()) & set(data_b.keys())
        if not common_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in common_keys:
            val_a, val_b = data_a[key], data_b[key]
            
            if isinstance(val_a, str) and isinstance(val_b, str):
                # String similarity
                common = set(val_a.lower().split()) & set(val_b.lower().split())
                total = set(val_a.lower().split()) | set(val_b.lower().split())
                similarity_sum += len(common) / max(1, len(total))
            elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                # Numeric similarity
                max_val = max(abs(val_a), abs(val_b))
                similarity_sum += 1.0 - (abs(val_a - val_b) / max(1, max_val))
        
        return similarity_sum / len(common_keys)
    
    def chunk_document(self, document_text: str, verbose: bool = True) -> List[MCPMChunk]:
        """Haupt-Chunking-Funktion mit MCPM"""
        
        if not self.chunk_targets:
            raise ValueError("No chunk targets defined. Call define_chunk_targets() first.")
        
        if verbose:
            print(f"🧬 MCMP-Chunking gestartet: {self.num_agents} Agenten, {len(self.chunk_targets)} Zielstrukturen")
        
        # Setup
        self.spawn_specialized_agents(document_text)
        self.discovered_chunks = []
        self.chunk_connections = {}
        
        # MCMP Exploration
        for iteration in range(self.max_iterations):
            
            # Agent Movement
            self.move_agents()
            
            # Exploration & Extraction
            new_chunks = []
            for agent in self.agents:
                
                # Exploriere aktuelle Region
                region_text = self.explore_document_region(agent)
                
                if region_text:
                    # Bestimme besten Chunk-Type für diese Region
                    best_type = None
                    best_score = 0.0
                    
                    for chunk_type in agent.target_types:
                        score = self.evaluate_region_for_chunk_type(region_text, chunk_type)
                        if score > best_score:
                            best_score = score
                            best_type = chunk_type
                    
                    if best_type and best_score > 0.6:
                        # Strukturierte Extraktion
                        chunk = self.extract_structured_chunk(region_text, best_type)
                        
                        if chunk and chunk.extraction_confidence > 0.4:
                            chunk.exploration_path.append(agent.id)
                            chunk.relevance_score = best_score
                            new_chunks.append(chunk)
                            agent.found_chunks.append(chunk.id)
                            agent.energy += 0.1  # Energy bonus for success
            
            # Deduplizierung neuer Chunks
            unique_chunks = self.deduplicate_chunks(new_chunks)
            self.discovered_chunks.extend(unique_chunks)
            
            # Progress
            if verbose and iteration % 20 == 0:
                print(f"  Iteration {iteration}: {len(self.discovered_chunks)} Chunks gefunden")
        
        # Post-Processing
        self.discover_chunk_connections()
        self.calculate_storage_priorities()
        
        # Statistiken
        self.update_exploration_stats()
        
        if verbose:
            print(f"✅ MCMP-Chunking abgeschlossen:")
            print(f"   Chunks gefunden: {len(self.discovered_chunks)}")
            print(f"   Chunk-Verbindungen: {len(self.chunk_connections)}")
            print(f"   Durchschnittliche Confidence: {self.exploration_stats['avg_confidence']:.3f}")
        
        return self.discovered_chunks
    
    def deduplicate_chunks(self, chunks: List[MCPMChunk]) -> List[MCPMChunk]:
        """Entferne duplizierte Chunks"""
        
        unique_chunks = []
        seen_contents = set()
        
        for chunk in chunks:
            # Content-based deduplication
            content_hash = hash(chunk.content[:100])  # First 100 chars
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunks.append(chunk)
            else:
                # Merge with existing chunk
                existing = next(c for c in unique_chunks if hash(c.content[:100]) == content_hash)
                existing.exploration_path.extend(chunk.exploration_path)
                existing.relevance_score = max(existing.relevance_score, chunk.relevance_score)
        
        return unique_chunks
    
    def calculate_storage_priorities(self):
        """Berechne Speicher-Prioritäten für Chunks"""
        
        for chunk in self.discovered_chunks:
            
            # Basis-Priorität von Zielstruktur
            base_priority = chunk.storage_priority
            
            # Confidence-Bonus
            confidence_bonus = chunk.extraction_confidence * 0.3
            
            # Connection-Bonus (gut vernetzte Chunks sind wichtiger)
            connection_bonus = min(0.4, len(chunk.connections) * 0.1)
            
            # Agent-Diversität-Bonus (von vielen verschiedenen Agenten gefunden)
            diversity_bonus = min(0.3, len(set(chunk.exploration_path)) * 0.05)
            
            # Finale Storage Priority
            chunk.storage_priority = base_priority + confidence_bonus + connection_bonus + diversity_bonus
    
    def update_exploration_stats(self):
        """Update Exploration-Statistiken"""
        
        if not self.discovered_chunks:
            return
        
        self.exploration_stats.update({
            "chunks_found": len(self.discovered_chunks),
            "avg_confidence": np.mean([c.extraction_confidence for c in self.discovered_chunks]),
            "connection_density": len(self.chunk_connections) / max(1, len(self.discovered_chunks)),
            "coverage_percentage": self.estimate_document_coverage()
        })
    
    def estimate_document_coverage(self) -> float:
        """Schätze Dokumenten-Abdeckung durch Chunks"""
        
        if not self.document_text or not self.discovered_chunks:
            return 0.0
        
        covered_chars = sum(len(chunk.content) for chunk in self.discovered_chunks)
        total_chars = len(self.document_text)
        
        # Approximation (Chunks können überlappen)
        coverage = min(1.0, covered_chars / total_chars)
        return coverage
    
    def get_optimized_chunks_for_storage(self, max_chunks: Optional[int] = None) -> List[MCPMChunk]:
        """Erhalte optimierte Chunks für effiziente Speicherung"""
        
        # Sortiere nach Storage Priority
        sorted_chunks = sorted(self.discovered_chunks, 
                              key=lambda c: c.storage_priority, 
                              reverse=True)
        
        if max_chunks:
            sorted_chunks = sorted_chunks[:max_chunks]
        
        return sorted_chunks
    
    def export_chunk_network(self, output_path: str):
        """Exportiere Chunk-Netzwerk für Visualisierung"""
        
        network_data = {
            "chunks": [
                {
                    "id": chunk.id,
                    "type": chunk.chunk_type,
                    "content_preview": chunk.content[:100] + "...",
                    "structured_data": chunk.structured_data,
                    "confidence": chunk.extraction_confidence,
                    "storage_priority": chunk.storage_priority,
                    "connections": chunk.connections
                }
                for chunk in self.discovered_chunks
            ],
            "connections": [
                {
                    "source": connection[0],
                    "target": connection[1], 
                    "strength": strength
                }
                for connection, strength in self.chunk_connections.items()
            ],
            "statistics": self.exploration_stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Chunk-Netzwerk exportiert: {output_path}")

# Demonstration
def demo_mcmp_chunking():
    """Demo der MCMP-Chunking Funktionalität"""
    
    print("🧬 MCMP-Chunking Demo")
    print("=" * 40)
    
    # Beispiel-Dokument
    document = """
    Machine Learning Tutorial
    
    Definition: Machine Learning ist ein Teilbereich der künstlichen Intelligenz, 
    der es Computern ermöglicht zu lernen ohne explizit programmiert zu werden.
    
    Technische Details: ML-Algorithmen verwenden statistische Verfahren um Muster 
    in Daten zu identifizieren. Supervised Learning nutzt gelabelte Trainingsdaten,
    während Unsupervised Learning Strukturen in ungelabelten Daten findet.
    
    Beispiel: Ein Spam-Filter lernt aus tausenden von E-Mails zu unterscheiden
    zwischen erwünschten und unerwünschten Nachrichten. Nach dem Training kann
    er neue E-Mails automatisch klassifizieren.
    
    Zusammenfassung: Machine Learning revolutioniert viele Bereiche durch seine
    Fähigkeit aus Daten zu lernen und Vorhersagen zu treffen. Anwendungen reichen
    von Empfehlungssystemen bis hin zu autonomen Fahrzeugen.
    """
    
    # Zielstrukturen definieren
    chunk_targets = [
        {
            "chunk_type": "definition",
            "schema": {
                "term": "str",
                "definition": "str", 
                "category": "str",
                "confidence": "float"
            },
            "required_fields": ["term", "definition"],
            "priority": 1.0
        },
        {
            "chunk_type": "technical_detail",
            "schema": {
                "topic": "str",
                "technical_explanation": "str",
                "methods": "list",
                "confidence": "float"
            },
            "required_fields": ["topic", "technical_explanation"],
            "priority": 0.8
        },
        {
            "chunk_type": "example",
            "schema": {
                "example_title": "str",
                "description": "str",
                "use_case": "str",
                "confidence": "float"
            },
            "required_fields": ["description"],
            "priority": 0.9
        },
        {
            "chunk_type": "summary",
            "schema": {
                "main_points": "list",
                "conclusion": "str",
                "applications": "list",
                "confidence": "float"
            },
            "required_fields": ["conclusion"],
            "priority": 1.0
        }
    ]
    
    # MCMP Chunker
    chunker = MCPMChunker(num_agents=20, max_iterations=30)
    chunker.define_chunk_targets(chunk_targets)
    
    # Chunking ausführen
    chunks = chunker.chunk_document(document, verbose=True)
    
    # Ergebnisse anzeigen
    print("\n📊 Extrahierte Chunks:")
    print("-" * 30)
    
    for chunk in chunks:
        print(f"\n🔹 {chunk.chunk_type.upper()} (ID: {chunk.id[:8]}...)")
        print(f"   Confidence: {chunk.extraction_confidence:.3f}")
        print(f"   Storage Priority: {chunk.storage_priority:.3f}")
        print(f"   Content: {chunk.content[:80]}...")
        print(f"   Structured Data: {json.dumps(chunk.structured_data, indent=2, ensure_ascii=False)[:200]}...")
        
        if chunk.connections:
            print(f"   Verbunden mit: {len(chunk.connections)} anderen Chunks")
    
    # Netzwerk exportieren
    chunker.export_chunk_network("mcmp_chunk_network.json")
    
    return chunks

if __name__ == "__main__":
    chunks = demo_mcmp_chunking()
