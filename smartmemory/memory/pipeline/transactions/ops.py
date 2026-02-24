from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from smartmemory.memory.pipeline.transactions.change_set import ChangeOp, ChangeSet, CommitResult, RollbackResult


class ChangeSetRegistry:
    """
    Minimal in-memory registry for change sets.
    NOTE: Replace with a persistent store (DB or filesystem) in production.
    """

    _SETS: Dict[str, ChangeSet] = {}

    @classmethod
    def save(cls, cs: ChangeSet) -> None:
        cls._SETS[cs.change_set_id] = cs

    @classmethod
    def get(cls, change_set_id: str) -> ChangeSet:
        cs = cls._SETS.get(change_set_id)
        if cs is None:
            raise KeyError(f"ChangeSet not found: {change_set_id}")
        return cs


class OperationApplier:
    """
    Applies ChangeSets to concrete stores and supports rollback using recorded inverse ops.

    Stores dict can include adapters like:
      - graph: object exposing add_edge(source, target, relation_type, properties),
               remove_edge(source, target, relation_type),
               set_properties(node_id, properties: dict),
               add_node(node_id, properties: dict),
               remove_node(node_id),
               archive(node_id), unarchive(node_id)
      - vector: object exposing upsert(item_id, embedding, metadata, node_ids, is_global), delete(item_id)
    """

    def __init__(self, stores: Dict[str, Any], strict_mode: bool = False):
        self.stores = stores or {}
        self.strict_mode = strict_mode

    # ---- Public API ---------------------------------------------------------

    def apply(self, change_set: ChangeSet) -> CommitResult:
        errors: List[str] = []
        inverses: List[ChangeOp] = []
        diffs: List[Dict[str, Any]] = []
        for op in change_set.ops:
            try:
                inv, diff = self._apply_op_with_diff(op)
                inverses.append(inv)
                if diff is not None:
                    diffs.append(diff)
            except Exception as e:
                errors.append(f"{op.op_type}: {e}")
        # Save inverses for rollback
        for op, inv in zip(change_set.ops, inverses, strict=False):
            op.inverse = inv
        # Attach diffs to metadata for audit
        if diffs:
            change_set.metadata.setdefault("diffs", []).extend(diffs)
        ChangeSetRegistry.save(change_set)
        return CommitResult(change_set_id=change_set.change_set_id, applied_ops=len(change_set.ops) - len(errors), errors=errors)

    def rollback(self, change_set_id: str) -> RollbackResult:
        cs = ChangeSetRegistry.get(change_set_id)
        errors: List[str] = []
        reverted = 0
        # Roll back in reverse order
        for op in reversed(cs.ops):
            inv = op.inverse
            if inv is None:
                # If inverse wasn't captured, try to compute trivial inverse
                try:
                    inv = self._compute_inverse_fallback(op)
                except Exception as e:
                    errors.append(f"no-inverse:{op.op_type}: {e}")
                    continue
            try:
                # We don't accumulate diffs during rollback
                self._apply_op_with_diff(inv, is_inverse=True)
                reverted += 1
            except Exception as e:
                errors.append(f"rollback:{op.op_type}: {e}")
        return RollbackResult(change_set_id=change_set_id, reverted_ops=reverted, errors=errors)

    # ---- Internals ----------------------------------------------------------

    def _apply_op_with_diff(self, op: ChangeOp, is_inverse: bool = False) -> Tuple[ChangeOp, Optional[Dict[str, Any]]]:
        t = op.op_type
        a = op.args or {}
        diff: Optional[Dict[str, Any]] = None
        # Graph operations
        if t == "add_edge":
            existed = self._edge_exists(a)
            prev_props = self._get_edge_properties(a) if existed else None
            self._graph().add_edge(a["source"], a["target"], a.get("relation_type", "RELATED"), a.get("properties", {}))
            inv = ChangeOp(op_type="remove_edge",
                           args={"source": a["source"], "target": a["target"], "relation_type": a.get("relation_type", "RELATED")}) if not existed else ChangeOp(op_type="add_edge",
                                                                                                                                                                 args={
                                                                                                                                                                     "source": a[
                                                                                                                                                                         "source"],
                                                                                                                                                                     "target": a[
                                                                                                                                                                         "target"],
                                                                                                                                                                     "relation_type": a.get(
                                                                                                                                                                         "relation_type",
                                                                                                                                                                         "RELATED"),
                                                                                                                                                                     "properties": prev_props or {}
                                                                                                                                                                 })
            diff = {"store": "graph", "kind": "insert" if not existed else "modify", "op": t, "before": prev_props, "after": a.get("properties", {})}
            return inv, diff
        if t == "remove_edge":
            prev_props = self._get_edge_properties(a)
            self._graph().remove_edge(a["source"], a["target"], a.get("relation_type", "RELATED"))
            inv = ChangeOp(op_type="add_edge",
                           args={"source": a["source"], "target": a["target"], "relation_type": a.get("relation_type", "RELATED"), "properties": prev_props or {}})
            diff = {"store": "graph", "kind": "delete", "op": t, "before": prev_props, "after": None}
            return inv, diff
        if t == "set_properties":
            node_id = a["node_id"]
            new_props = a.get("properties", {})
            prev_props = a.get("previous_properties")
            if prev_props is None:
                prev_props = self._graph_get_properties(node_id)
            self._graph().set_properties(node_id, new_props)
            inv = ChangeOp(op_type="restore_properties", args={"node_id": node_id, "properties": prev_props})
            diff = {"store": "graph", "kind": "modify", "op": t, "before": prev_props, "after": new_props}
            return inv, diff
        if t == "restore_properties":
            node_id = a["node_id"]
            props = a.get("properties", {})
            prev = self._graph_get_properties(node_id)
            self._graph().set_properties(node_id, props)
            inv = ChangeOp(op_type="set_properties", args={"node_id": node_id, "properties": prev})
            diff = {"store": "graph", "kind": "modify", "op": t, "before": prev, "after": props}
            return inv, diff
        if t == "add_node":
            node_id = a["node_id"]
            existed = self._graph_node_exists(node_id)
            prev_props = self._graph_get_properties(node_id) if existed else None
            self._graph().add_node(node_id, a.get("properties", {}))
            inv = ChangeOp(op_type="remove_node", args={"node_id": node_id, "properties": prev_props or {}}) if not existed else ChangeOp(op_type="add_node",
                                                                                                                                          args={
                                                                                                                                              "node_id": node_id,
                                                                                                                                              "properties": prev_props or {}
                                                                                                                                          })
            diff = {"store": "graph", "kind": "insert" if not existed else "modify", "op": t, "before": prev_props, "after": a.get("properties", {})}
            return inv, diff
        if t == "remove_node":
            node_id = a["node_id"]
            prev_props = self._graph_get_properties(node_id)
            incident_edges = self._graph_list_edges(node_id)
            self._graph().remove_node(node_id)
            inv = ChangeOp(op_type="add_node", args={"node_id": node_id, "properties": prev_props or {}})
            # Re-attach incident edges during rollback by appending additional ops into inverse metadata (not supported in single-op inverse)
            # Store them in diff so callers can reconstruct if needed
            diff = {"store": "graph", "kind": "delete", "op": t, "before": {"properties": prev_props, "edges": incident_edges}, "after": None}
            return inv, diff
        if t == "archive":
            node_id = a["node_id"]
            self._graph().archive(node_id)  # type: ignore[attr-defined]
            inv = ChangeOp(op_type="unarchive", args={"node_id": node_id})
            diff = {"store": "graph", "kind": "modify", "op": t, "before": {"archived": False}, "after": {"archived": True}}
            return inv, diff
        if t == "unarchive":
            node_id = a["node_id"]
            self._graph().unarchive(node_id)  # type: ignore[attr-defined]
            inv = ChangeOp(op_type="archive", args={"node_id": node_id})
            diff = {"store": "graph", "kind": "modify", "op": t, "before": {"archived": True}, "after": {"archived": False}}
            return inv, diff
        # Vector operations
        if t == "vector_upsert":
            vs = self._vector()
            item_id = a["item_id"]
            prev = self._vector_get(item_id)
            vs.upsert(
                item_id=item_id,
                embedding=a.get("embedding"),
                metadata=a.get("metadata", {}),
                node_ids=a.get("node_ids", []),
                is_global=bool(a.get("is_global", True)),
            )
            if prev is None:
                inv = ChangeOp(op_type="vector_delete", args={"item_id": item_id})
                diff = {"store": "vector", "kind": "insert", "op": t, "before": None, "after": {k: a.get(k) for k in ("embedding", "metadata", "node_ids", "is_global")}}
            else:
                inv = ChangeOp(op_type="vector_upsert", args={
                    "item_id": item_id,
                    "embedding": prev.get("embedding"),
                    "metadata": prev.get("metadata", {}),
                    "node_ids": prev.get("node_ids", []),
                    "is_global": bool(prev.get("is_global", True)),
                })
                diff = {"store": "vector", "kind": "modify", "op": t, "before": prev, "after": {k: a.get(k) for k in ("embedding", "metadata", "node_ids", "is_global")}}
            return inv, diff
        if t == "vector_delete":
            item_id = a["item_id"]
            prev = self._vector_get(item_id)
            self._vector().delete(item_id)  # type: ignore[call-arg]
            inv = ChangeOp(op_type="vector_upsert", args={
                "item_id": item_id,
                "embedding": None if prev is None else prev.get("embedding"),
                "metadata": {} if prev is None else prev.get("metadata", {}),
                "node_ids": [] if prev is None else prev.get("node_ids", []),
                "is_global": True if prev is None else bool(prev.get("is_global", True)),
            })
            diff = {"store": "vector", "kind": "delete", "op": t, "before": prev, "after": None}
            return inv, diff
        raise ValueError(f"Unsupported ChangeOp: {t}")

    def _compute_inverse_fallback(self, op: ChangeOp) -> ChangeOp:
        # Best-effort inverse when none recorded; may be lossy if properties/embeddings not provided
        t = op.op_type
        a = op.args or {}
        if t == "add_edge":
            return ChangeOp(op_type="remove_edge", args={"source": a["source"], "target": a["target"], "relation_type": a.get("relation_type", "RELATED")})
        if t == "remove_edge":
            return ChangeOp(op_type="add_edge",
                            args={"source": a["source"], "target": a["target"], "relation_type": a.get("relation_type", "RELATED"), "properties": a.get("properties", {})})
        if t == "set_properties":
            return ChangeOp(op_type="restore_properties", args={"node_id": a["node_id"], "properties": a.get("previous_properties", {})})
        if t == "restore_properties":
            return ChangeOp(op_type="set_properties", args={"node_id": a["node_id"], "properties": a.get("previous_properties", {})})
        if t == "add_node":
            return ChangeOp(op_type="remove_node", args={"node_id": a["node_id"]})
        if t == "remove_node":
            return ChangeOp(op_type="add_node", args={"node_id": a["node_id"], "properties": a.get("properties", {})})
        if t == "archive":
            return ChangeOp(op_type="unarchive", args={"node_id": a["node_id"]})
        if t == "unarchive":
            return ChangeOp(op_type="archive", args={"node_id": a["node_id"]})
        if t == "vector_upsert":
            return ChangeOp(op_type="vector_delete", args={"item_id": a["item_id"]})
        if t == "vector_delete":
            return ChangeOp(op_type="vector_upsert",
                            args={
                                "item_id": a["item_id"],
                                "embedding": a.get("embedding"),
                                "metadata": a.get("metadata", {}),
                                "node_ids": a.get("node_ids", []),
                                "is_global": bool(a.get("is_global", True))
                            })
        raise ValueError(f"Cannot compute inverse for {t}")

    # ---- Store helpers ------------------------------------------------------

    def _graph(self):
        g = self.stores.get("graph")
        if g is None:
            raise RuntimeError("Graph store not provided to OperationApplier")
        return g

    def _vector(self):
        v = self.stores.get("vector")
        if v is None:
            raise RuntimeError("Vector store not provided to OperationApplier")
        return v

    # ---- Store read helpers -------------------------------------------------

    def _graph_get_properties(self, node_id: str) -> Dict[str, Any]:
        g = self._graph()
        getter = getattr(g, "get_properties", None)
        if callable(getter):
            try:
                return getter(node_id) or {}
            except Exception:
                return {}
        return {}

    def _graph_node_exists(self, node_id: str) -> bool:
        g = self._graph()
        exists = getattr(g, "node_exists", None)
        if callable(exists):
            try:
                return bool(exists(node_id))
            except Exception:
                return False
        # Fallback: if no method, assume not exists (conservative insert semantics)
        return False

    def _edge_exists(self, a: Dict[str, Any]) -> bool:
        g = self._graph()
        exists = getattr(g, "edge_exists", None)
        if callable(exists):
            try:
                return bool(exists(a["source"], a["target"], a.get("relation_type", "RELATED")))
            except Exception:
                return False
        return False

    def _get_edge_properties(self, a: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        g = self._graph()
        getter = getattr(g, "get_edge_properties", None)
        if callable(getter):
            try:
                return getter(a["source"], a["target"], a.get("relation_type", "RELATED"))
            except Exception:
                return None
        return None

    def _graph_list_edges(self, node_id: str) -> List[Dict[str, Any]]:
        g = self._graph()
        lister = getattr(g, "list_edges", None)
        if callable(lister):
            try:
                return list(lister(node_id) or [])
            except Exception:
                return []
        return []

    def _vector_get(self, item_id: str) -> Optional[Dict[str, Any]]:
        v = self._vector()
        getter = getattr(v, "get", None)
        if callable(getter):
            try:
                return getter(item_id)
            except Exception:
                return None
        return None
