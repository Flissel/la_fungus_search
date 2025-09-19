#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCMP-RAG Command Line Interface (CLI)

Subcommands:
  - search:      Einmalige Suche über gegebene Dokumente
  - interactive: Interaktiver Modus im Terminal
  - benchmark:   Mehrere Queries messen und Ergebnisse exportieren
  - demo:        MCMP-RAG Demo ausführen

Utilities:
  - load_documents_from_file(path)
  - save_results(data, output_path)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from mcmp_rag import MCPMRetriever


def load_documents_from_file(file_path: str) -> List[str]:
    """Lade Dokumente (eine Zeile pro Dokument) aus einer Textdatei."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dokumentendatei nicht gefunden: {file_path}")
    texts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def save_results(data: Any, output_path: str) -> None:
    """Speichere Ergebnisse als JSON-Datei."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_documents_from_markdown(file_path: str, include_text: bool = False) -> List[str]:
    """Extrahiere komplette ```code```-Blöcke aus einer Markdown-Datei.

    - Gibt jeden Codeblock als eigenständiges 'Dokument' zurück
    - Bewahrt die Fences inkl. Sprache (```lang\n...\n```), damit ganze Beispiele retrieved werden
    - Optional: include_text=True -> Nicht-Code Abschnitte als zusätzliche Dokumente
    """
    import re
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown-Datei nicht gefunden: {file_path}")

    content = path.read_text(encoding="utf-8", errors="ignore")

    # Fenced codeblocks erfassen
    code_pattern = re.compile(r"```([a-zA-Z0-9_+\-]*)\n([\s\S]*?)```", re.MULTILINE)
    documents: List[str] = []

    last_end = 0
    for match in code_pattern.finditer(content):
        lang = match.group(1) or ""
        code = match.group(2).strip("\n")
        full_block = f"```{lang}\n{code}\n```"
        documents.append(full_block)

        if include_text:
            # Text zwischen Blöcken als Dokument aufnehmen
            prefix_text = content[last_end:match.start()].strip()
            if prefix_text:
                # Kürzen langer Bereiche in kleinere Absätze
                for para in re.split(r"\n\n+", prefix_text):
                    para = para.strip()
                    if para:
                        documents.append(para)
        last_end = match.end()

    # Restlicher Nachlauf-Text
    if include_text:
        tail = content[last_end:].strip()
        if tail:
            for para in re.split(r"\n\n+", tail):
                para = para.strip()
                if para:
                    documents.append(para)

    # Falls keine Codeblöcke gefunden: Zeilenweise Fallback
    if not documents:
        return load_documents_from_file(file_path)
    return documents


def build_mcmp(args: argparse.Namespace) -> MCPMRetriever:
    """Erzeuge und konfiguriere einen MCPMRetriever aus CLI-Argumenten."""
    return MCPMRetriever(
        embedding_model_name=getattr(args, "model", "google/embeddinggemma-300m"),
        num_agents=getattr(args, "agents", 300),
        max_iterations=getattr(args, "iterations", 80),
        pheromone_decay=getattr(args, "pheromone_decay", 0.95),
        exploration_bonus=getattr(args, "exploration_bonus", 0.1),
        device_mode=getattr(args, "device", "auto"),
        use_embedding_model=not getattr(args, "mock", False),
    )


def ensure_documents_loaded(mcmp: MCPMRetriever, documents: List[str]) -> bool:
    """Versuche Dokumente regulär zu laden, fallback zu Mock-Embeddings falls nötig."""
    try:
        success = mcmp.add_documents(documents)
        if success:
            return True
    except Exception:
        pass

    # Fallback: Mock-Embeddings, falls EmbeddingGemma nicht verfügbar ist
    try:
        import numpy as np
        from mcmp_rag import Document

        mcmp.documents = []
        for i, text in enumerate(documents):
            embedding = np.random.normal(0, 1, 768)
            embedding = embedding / (np.linalg.norm(embedding) or 1.0)
            mcmp.documents.append(Document(id=i, content=text, embedding=embedding))
        print("[INFO] Verwende Mock-Embeddings (EmbeddingGemma nicht verfügbar)")
        return True
    except Exception as e:
        print(f"[ERROR] Konnte Dokumente nicht vorbereiten: {e}")
        return False


def cmd_search(args: argparse.Namespace) -> int:
    documents: List[str] = []
    if args.docs:
        documents = args.docs
    elif args.file:
        if args.md_codeblocks or (args.file.lower().endswith(".md") and args.md_auto):
            documents = load_documents_from_markdown(args.file, include_text=args.md_include_text)
        else:
            documents = load_documents_from_file(args.file)
    else:
        # Fallback auf Beispiel-Dokumente
        example = Path(__file__).with_name("example_documents.txt")
        if example.exists():
            documents = load_documents_from_file(str(example))
        else:
            documents = [
                "Python ist eine Programmiersprache",
                "Machine Learning nutzt Algorithmen",
                "Künstliche Intelligenz ist zukunftsweisend",
            ]

    mcmp = build_mcmp(args)
    if not ensure_documents_loaded(mcmp, documents):
        print("❌ Keine Dokumente verfügbar.")
        return 2

    top_k = getattr(args, "top_k", 5)
    results = mcmp.search(args.query, top_k=top_k, verbose=not args.quiet)

    if "error" in results:
        print(f"❌ Suche fehlgeschlagen: {results['error']}")
        return 3

    # Konsolen-Ausgabe
    print(f"\n📊 Gefunden: {len(results.get('results', []))} Dokumente")
    for i, item in enumerate(results.get("results", [])[:min(5, top_k)], 1):
        print(f"  {i}. [{item['relevance_score']:.3f}] {item['content'][:80]}...")

    if args.output:
        save_results(results, args.output)
        print(f"💾 Ergebnisse gespeichert: {args.output}")

    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    # Dokumente initialisieren
    documents: List[str] = []
    if args.docs:
        documents = args.docs
    elif args.file:
        documents = load_documents_from_file(args.file)
    else:
        example = Path(__file__).with_name("example_documents.txt")
        if example.exists():
            documents = load_documents_from_file(str(example))
        else:
            documents = [
                "Python ist eine Programmiersprache",
                "Machine Learning nutzt Algorithmen",
                "Künstliche Intelligenz ist zukunftsweisend",
            ]

    mcmp = build_mcmp(args)
    if not ensure_documents_loaded(mcmp, documents):
        print("❌ Keine Dokumente verfügbar.")
        return 2

    print("MCMP-RAG Interactive Mode (exit mit 'quit'/'q')")
    while True:
        try:
            query = input("\n🔎 Frage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBeende Interaktiv-Modus.")
            break

        if query.lower() in {"", "quit", "exit", "q"}:
            print("Beende Interaktiv-Modus.")
            break

        start = time.time()
        results = mcmp.search(query, top_k=getattr(args, "top_k", 5), verbose=False)
        dur_ms = int((time.time() - start) * 1000)

        if "error" in results:
            print(f"❌ Fehler: {results['error']}")
            continue

        print(f"⏱️  {dur_ms} ms | Ergebnisse: {len(results['results'])}")
        for i, item in enumerate(results["results"][:5], 1):
            print(f"  {i}. [{item['relevance_score']:.3f}] {item['content'][:100]}...")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    # Dokumente
    if args.file:
        if args.md_codeblocks or (args.file.lower().endswith(".md") and args.md_auto):
            documents = load_documents_from_markdown(args.file, include_text=args.md_include_text)
        else:
            documents = load_documents_from_file(args.file)
    else:
        example = Path(__file__).with_name("example_documents.txt")
        if example.exists():
            documents = load_documents_from_file(str(example))
        else:
            documents = [
                "Python ist eine Programmiersprache",
                "Machine Learning nutzt Algorithmen",
                "Künstliche Intelligenz ist zukunftsweisend",
                "Cloud Computing ermöglicht skalierbare IT-Infrastrukturen",
            ]

    # Queries
    queries: List[str] = args.queries or [
        "Was ist Machine Learning?",
        "Welche Cloud-Services gibt es?",
        "Wie funktioniert Cybersecurity?",
        "Was sind die neuesten AI-Trends?",
    ]

    mcmp = build_mcmp(args)
    if not ensure_documents_loaded(mcmp, documents):
        print("❌ Keine Dokumente verfügbar.")
        return 2

    report: Dict[str, Any] = {
        "config": {
            "agents": args.agents,
            "iterations": args.iterations,
            "top_k": args.top_k,
            "model": args.model,
        },
        "results": [],
    }

    print("MCMP-RAG Benchmark startet...")
    for q in queries:
        start = time.time()
        res = mcmp.search(q, top_k=args.top_k, verbose=False)
        elapsed = time.time() - start
        if "error" in res:
            print(f"❌ Fehler bei Query '{q}': {res['error']}")
            continue
        top_preview = [r["content"][:60] for r in res.get("results", [])[:3]]
        print(f"- '{q}' -> {len(res.get('results', []))} Treffer in {elapsed:.2f}s | {top_preview}")
        report["results"].append({
            "query": q,
            "elapsed_seconds": elapsed,
            "num_results": len(res.get("results", [])),
            "top_results": res.get("results", [])[:5],
            "network_stats": res.get("network_stats", {}),
        })

    if args.output:
        save_results(report, args.output)
        print(f"💾 Benchmark gespeichert: {args.output}")

    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    try:
        from mcmp_rag import demo_mcpm
        demo_mcpm()
        return 0
    except Exception as e:
        print(f"❌ Demo fehlgeschlagen: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCMP-RAG CLI")
    sub = parser.add_subparsers(dest="command")

    # Gemeinsame Defaults
    default_agents = 300
    default_iterations = 80
    default_top_k = 5

    # search
    p_search = sub.add_parser("search", help="Einmalige Suche ausführen")
    p_search.add_argument("query", type=str, help="Suchanfrage")
    p_search.add_argument("--docs", nargs="+", help="Dokumente als Liste von Strings")
    p_search.add_argument("--file", type=str, help="Pfad zu einer Datei mit Dokumenten")
    p_search.add_argument("--output", type=str, help="Optionaler JSON-Exportpfad")
    p_search.add_argument("--agents", type=int, default=default_agents)
    p_search.add_argument("--iterations", type=int, default=default_iterations)
    p_search.add_argument("--top-k", dest="top_k", type=int, default=default_top_k)
    p_search.add_argument("--model", type=str, default="google/embeddinggemma-300m")
    p_search.add_argument("--pheromone-decay", dest="pheromone_decay", type=float, default=0.95)
    p_search.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p_search.add_argument("--exploration-bonus", dest="exploration_bonus", type=float, default=0.1)
    p_search.add_argument("--quiet", action="store_true", help="Weniger Ausgabe")
    p_search.add_argument("--mock", action="store_true", help="Ohne Embedding-Modell (Mock-Embeddings)")
    p_search.add_argument("--md-codeblocks", dest="md_codeblocks", action="store_true", help="Markdown-Codeblöcke als Dokumente extrahieren")
    p_search.add_argument("--md-include-text", dest="md_include_text", action="store_true", help="Zusätzlich Nicht-Code Abschnitte als Dokumente aufnehmen")
    p_search.add_argument("--md-auto", dest="md_auto", action="store_true", help="Bei .md-Dateien automatisch Codeblock-Loader verwenden")
    p_search.set_defaults(func=cmd_search)

    # interactive
    p_inter = sub.add_parser("interactive", help="Interaktiver Modus")
    p_inter.add_argument("--docs", nargs="+", help="Dokumente als Liste von Strings")
    p_inter.add_argument("--file", type=str, help="Pfad zu einer Datei mit Dokumenten")
    p_inter.add_argument("--agents", type=int, default=default_agents)
    p_inter.add_argument("--iterations", type=int, default=default_iterations)
    p_inter.add_argument("--top-k", dest="top_k", type=int, default=default_top_k)
    p_inter.add_argument("--model", type=str, default="google/embeddinggemma-300m")
    p_inter.add_argument("--pheromone-decay", dest="pheromone_decay", type=float, default=0.95)
    p_inter.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p_inter.add_argument("--exploration-bonus", dest="exploration_bonus", type=float, default=0.1)
    p_inter.add_argument("--mock", action="store_true", help="Ohne Embedding-Modell (Mock-Embeddings)")
    p_inter.add_argument("--md-codeblocks", dest="md_codeblocks", action="store_true")
    p_inter.add_argument("--md-include-text", dest="md_include_text", action="store_true")
    p_inter.add_argument("--md-auto", dest="md_auto", action="store_true")
    p_inter.set_defaults(func=cmd_interactive)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Mehrere Queries messen")
    p_bench.add_argument("--file", type=str, help="Pfad zu einer Datei mit Dokumenten")
    p_bench.add_argument("--queries", nargs="+", help="Liste von Queries")
    p_bench.add_argument("--output", type=str, help="Optionaler JSON-Exportpfad")
    p_bench.add_argument("--agents", type=int, default=default_agents)
    p_bench.add_argument("--iterations", type=int, default=default_iterations)
    p_bench.add_argument("--top-k", dest="top_k", type=int, default=default_top_k)
    p_bench.add_argument("--model", type=str, default="google/embeddinggemma-300m")
    p_bench.add_argument("--pheromone-decay", dest="pheromone_decay", type=float, default=0.95)
    p_bench.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p_bench.add_argument("--exploration-bonus", dest="exploration_bonus", type=float, default=0.1)
    p_bench.add_argument("--mock", action="store_true", help="Ohne Embedding-Modell (Mock-Embeddings)")
    p_bench.add_argument("--md-codeblocks", dest="md_codeblocks", action="store_true")
    p_bench.add_argument("--md-include-text", dest="md_include_text", action="store_true")
    p_bench.add_argument("--md-auto", dest="md_auto", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    # demo
    p_demo = sub.add_parser("demo", help="MCMP-RAG Demo ausführen")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
