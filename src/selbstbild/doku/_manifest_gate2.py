"""Call-graph judgement manifest over a real Python corpus."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    document_id: str
    file: str
    start_line: int
    end_line: int
    symbol: str
    source: str


@dataclass(frozen=True)
class Manifest:
    manifest_id: str
    corpus_root: str
    commit_sha: str
    documents: tuple[Document, ...]
    callees_by_document: dict[str, frozenset[str]]
    callers_by_document: dict[str, frozenset[str]]
    discarded_names: tuple[str, ...]
    unresolved_names: tuple[str, ...]


_DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _module_path(path: Path, corpus_root: Path) -> str:
    relative = path.relative_to(corpus_root).with_suffix("")
    return ".".join(relative.parts)


def _call_names(node: ast.AST, *, skip_method_bodies: bool) -> set[str]:
    names: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if skip_method_bodies and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for descendant in ast.walk(child):
            if isinstance(descendant, ast.Call):
                function = descendant.func
                if isinstance(function, ast.Name):
                    names.add(function.id)
                elif isinstance(function, ast.Attribute):
                    names.add(function.attr)
    return names


def _collect(node: ast.AST, prefix: str, lines: list[str], file: str,
             documents: list[Document], raw_calls: dict[str, set[str]]) -> None:
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, _DEFINITION_NODES):
            continue
        document_id = f"{prefix}.{child.name}"
        end_line = getattr(child, "end_lineno", child.lineno)
        documents.append(
            Document(
                document_id=document_id,
                file=file,
                start_line=child.lineno,
                end_line=end_line,
                symbol=child.name,
                source="\n".join(lines[child.lineno - 1 : end_line]),
            )
        )
        is_class = isinstance(child, ast.ClassDef)
        raw_calls[document_id] = _call_names(child, skip_method_bodies=is_class)
        if is_class:
            _collect(child, document_id, lines, file, documents, raw_calls)


def build_manifest(
    corpus_root: Path,
    commit_sha: str,
    manifest_id: str,
    exclude_dirs: frozenset[str] = frozenset(),
) -> Manifest:
    """Extract documents and a fail-closed call graph from a Python corpus.

    ``exclude_dirs`` prunes by directory *name*, checked against the path
    relative to ``corpus_root`` — relative on purpose: a corpus that itself
    lives under an excluded-sounding parent (say a `.claude` tree) must not go
    blank because of where it is mounted. Default empty keeps the original
    behaviour for every existing caller; the maintainer-evidence CLI passes the
    vendor set, without which walking a repo means AST-parsing its virtualenv.
    """
    documents: list[Document] = []
    raw_calls: dict[str, set[str]] = {}
    for path in sorted(corpus_root.rglob("*.py")):
        relative_parts = path.relative_to(corpus_root).parts
        if "__pycache__" in relative_parts:
            continue
        if exclude_dirs and exclude_dirs.intersection(relative_parts[:-1]):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        file = path.relative_to(corpus_root).as_posix()
        _collect(tree, _module_path(path, corpus_root), lines, file, documents, raw_calls)

    by_symbol: dict[str, list[str]] = {}
    for document in documents:
        by_symbol.setdefault(document.symbol, []).append(document.document_id)

    callees: dict[str, frozenset[str]] = {}
    discarded: set[str] = set()
    unresolved: set[str] = set()
    for document_id, names in raw_calls.items():
        resolved: set[str] = set()
        for name in sorted(names):
            targets = by_symbol.get(name, [])
            if len(targets) == 1:
                if targets[0] != document_id:
                    resolved.add(targets[0])
            elif len(targets) > 1:
                discarded.add(name)
            else:
                unresolved.add(name)
        callees[document_id] = frozenset(resolved)

    callers: dict[str, set[str]] = {document.document_id: set() for document in documents}
    for source_id, targets in callees.items():
        for target in targets:
            callers[target].add(source_id)

    return Manifest(
        manifest_id=manifest_id,
        corpus_root=corpus_root.as_posix(),
        commit_sha=commit_sha,
        documents=tuple(documents),
        callees_by_document=callees,
        callers_by_document={key: frozenset(value) for key, value in callers.items()},
        discarded_names=tuple(sorted(discarded)),
        unresolved_names=tuple(sorted(unresolved)),
    )


def relevant_for(manifest: Manifest, document_id: str) -> frozenset[str]:
    """Direct callers and callees of a document, excluding itself."""
    neighbours = set(manifest.callees_by_document.get(document_id, frozenset()))
    neighbours |= set(manifest.callers_by_document.get(document_id, frozenset()))
    neighbours.discard(document_id)
    return frozenset(neighbours)


def query_candidates(manifest: Manifest) -> tuple[str, ...]:
    """Documents with at least one resolved call-graph neighbour."""
    return tuple(
        document.document_id
        for document in manifest.documents
        if relevant_for(manifest, document.document_id)
    )


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_id": manifest.manifest_id,
        "corpus_root": manifest.corpus_root,
        "commit_sha": manifest.commit_sha,
        "documents": [
            {
                "document_id": document.document_id,
                "file": document.file,
                "start_line": document.start_line,
                "end_line": document.end_line,
                "symbol": document.symbol,
                "source": document.source,
            }
            for document in manifest.documents
        ],
        "callees_by_document": {
            key: sorted(value) for key, value in sorted(manifest.callees_by_document.items())
        },
        "callers_by_document": {
            key: sorted(value) for key, value in sorted(manifest.callers_by_document.items())
        },
        "discarded_names": list(manifest.discarded_names),
        "unresolved_names": list(manifest.unresolved_names),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Manifest(
        manifest_id=payload["manifest_id"],
        corpus_root=payload["corpus_root"],
        commit_sha=payload["commit_sha"],
        documents=tuple(Document(**entry) for entry in payload["documents"]),
        callees_by_document={
            key: frozenset(value) for key, value in payload["callees_by_document"].items()
        },
        callers_by_document={
            key: frozenset(value) for key, value in payload["callers_by_document"].items()
        },
        discarded_names=tuple(payload["discarded_names"]),
        unresolved_names=tuple(payload["unresolved_names"]),
    )


def manifest_digest(manifest: Manifest) -> str:
    """Stable digest over ids, edges and provenance."""
    result = hashlib.sha256()
    result.update(b"Gate2Manifest/v1\0")
    for part in (manifest.manifest_id, manifest.corpus_root, manifest.commit_sha):
        result.update(part.encode("utf-8") + b"\0")
    for document in manifest.documents:
        result.update(document.document_id.encode("utf-8") + b"\0")
        # The source text is what both snapshot builders embed. Without it a
        # snapshot built from an older corpus revision would pass the digest
        # check against a newer manifest whenever symbol names and call
        # structure are unchanged -- exactly what refactoring a body looks like.
        result.update(document.source.encode("utf-8") + b"\0")
    for key in sorted(manifest.callees_by_document):
        result.update(key.encode("utf-8") + b":")
        result.update(",".join(sorted(manifest.callees_by_document[key])).encode("utf-8") + b"\0")
    return result.hexdigest()
