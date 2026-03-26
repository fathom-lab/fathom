"""
COGNITIVE TOPOLOGY MAP — Feature Co-Activation Graph Analysis
=============================================================

Feature co-activation is not random. Certain SAE features consistently fire
together during deep reasoning but not during surface recall. This module
builds a co-activation graph from batches of DepthResults, identifies
reasoning hubs (the features that distinguish thinking from lookup), and
exports topology data for visualization.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np

# Optional: networkx for proper graph algorithms
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


@dataclass
class ReasoningHub:
    """A feature-layer pair that disproportionately appears in deep reasoning."""
    layer: int
    frequency_high_depth: float   # how often this layer appears in high-depth results
    frequency_low_depth: float    # how often this layer appears in low-depth results
    lift: float                   # ratio of high-depth to low-depth frequency
    hub_score: float              # composite score ranking this as a reasoning hub


class FeatureTopology:
    """Builds and analyzes co-activation topology from batches of depth results.

    Maps which layers consistently fire together across prompts, identifies
    the layer combinations that distinguish deep reasoning from surface recall,
    and exports the topology as a JSON graph for D3.js visualization. This is
    the structural fingerprint of a model's reasoning circuitry.
    """

    def __init__(self, depth_threshold: Optional[float] = None):
        """
        Args:
            depth_threshold: Depth score below which a result is considered 'deep'.
                If None, uses the median of the provided results as the split point.
        """
        self.depth_threshold = depth_threshold
        self._results = []
        self._graph = None

    def add_results(self, results: list):
        """Add DepthResult objects to the topology dataset.

        Args:
            results: List of DepthResult objects (None entries are skipped).
        """
        for r in results:
            if r is not None and hasattr(r, "layer_profile") and r.layer_profile:
                self._results.append(r)
        self._graph = None  # invalidate cached graph

    def build_coactivation_graph(self, min_cooccurrence: int = 2) -> dict:
        """Build a co-activation graph where nodes are layers and edges
        represent how often two layers have active features in the same prompt.

        Args:
            min_cooccurrence: Minimum co-occurrence count to create an edge.

        Returns:
            Dict with 'nodes' and 'edges' suitable for graph construction.
        """
        if not self._results:
            return {"nodes": [], "edges": []}

        # Count layer co-occurrences across all results
        cooccurrence = Counter()
        layer_counts = Counter()

        for result in self._results:
            active_layers = sorted(result.layer_profile.keys())
            for layer in active_layers:
                layer_counts[layer] += 1
            # Pairwise co-activation
            for i, l1 in enumerate(active_layers):
                for l2 in active_layers[i + 1:]:
                    pair = (min(l1, l2), max(l1, l2))
                    cooccurrence[pair] += 1

        n_results = len(self._results)

        # Build node list
        nodes = []
        for layer, count in sorted(layer_counts.items()):
            nodes.append({
                "id": int(layer),
                "label": f"L{layer}",
                "frequency": count / n_results,
                "count": count,
            })

        # Build edge list
        edges = []
        for (l1, l2), count in cooccurrence.items():
            if count >= min_cooccurrence:
                # Jaccard coefficient as edge weight
                union = layer_counts[l1] + layer_counts[l2] - count
                jaccard = count / union if union > 0 else 0
                edges.append({
                    "source": int(l1),
                    "target": int(l2),
                    "weight": jaccard,
                    "cooccurrence": count,
                })

        self._graph = {"nodes": nodes, "edges": edges}

        if HAS_NETWORKX:
            self._nx_graph = nx.Graph()
            for node in nodes:
                self._nx_graph.add_node(node["id"], **node)
            for edge in edges:
                self._nx_graph.add_edge(
                    edge["source"], edge["target"],
                    weight=edge["weight"],
                    cooccurrence=edge["cooccurrence"],
                )

        return self._graph

    def find_reasoning_hubs(
        self,
        results: Optional[list] = None,
        top_n: int = 10,
    ) -> List[ReasoningHub]:
        """Identify layers that appear disproportionately in high-depth results.

        Splits results into high-depth and low-depth groups, computes per-layer
        frequency in each group, and ranks layers by their lift ratio. High-lift
        layers are the reasoning hubs — the nodes that distinguish thinking from lookup.

        Args:
            results: Optional list of DepthResult objects. Uses stored results if None.
            top_n: Number of top hubs to return.

        Returns:
            List of ReasoningHub objects sorted by hub_score.
        """
        if results is not None:
            self.add_results(results)

        if len(self._results) < 4:
            return []

        # Split into high-depth and low-depth
        scores = [r.depth_score for r in self._results if not math.isnan(r.depth_score)]
        if not scores:
            return []

        threshold = self.depth_threshold
        if threshold is None:
            threshold = float(np.median(scores))

        high_depth = [r for r in self._results if not math.isnan(r.depth_score) and r.depth_score < threshold]
        low_depth = [r for r in self._results if not math.isnan(r.depth_score) and r.depth_score >= threshold]

        if not high_depth or not low_depth:
            return []

        # Layer frequency in each group
        def layer_freqs(group):
            freq = Counter()
            for r in group:
                for layer in r.layer_profile:
                    freq[layer] += 1
            return {layer: count / len(group) for layer, count in freq.items()}

        high_freq = layer_freqs(high_depth)
        low_freq = layer_freqs(low_depth)

        all_layers = set(high_freq.keys()) | set(low_freq.keys())

        hubs = []
        for layer in all_layers:
            hf = high_freq.get(layer, 0.0)
            lf = low_freq.get(layer, 0.0)
            # Lift: how much more likely this layer is in deep reasoning
            lift = (hf / lf) if lf > 0 else (hf * 10 if hf > 0 else 0)
            # Hub score: lift weighted by frequency (rare layers with high lift are less interesting)
            hub_score = lift * hf
            hubs.append(ReasoningHub(
                layer=int(layer),
                frequency_high_depth=hf,
                frequency_low_depth=lf,
                lift=lift,
                hub_score=hub_score,
            ))

        hubs.sort(key=lambda h: h.hub_score, reverse=True)
        return hubs[:top_n]

    def visualize_topology(
        self,
        results: Optional[list] = None,
        output_path: str = "topology.json",
    ) -> dict:
        """Export the co-activation topology as a JSON graph for D3.js rendering.

        Nodes are layers with size proportional to frequency. Edges connect
        co-activated layers with width proportional to Jaccard similarity.
        Reasoning hubs are tagged for visual emphasis.

        Args:
            results: Optional list of DepthResult objects.
            output_path: Path to write the JSON file.

        Returns:
            The graph dict that was written.
        """
        import json

        if results is not None:
            self.add_results(results)

        if self._graph is None:
            self.build_coactivation_graph()

        # Tag reasoning hubs
        hubs = self.find_reasoning_hubs()
        hub_layers = {h.layer for h in hubs[:5]}

        graph = {
            "nodes": [],
            "edges": self._graph["edges"],
            "metadata": {
                "n_results": len(self._results),
                "depth_threshold": self.depth_threshold or "median",
            },
        }

        # Enrich nodes
        hub_lookup = {h.layer: h for h in hubs}
        for node in self._graph["nodes"]:
            enriched = dict(node)
            enriched["is_reasoning_hub"] = node["id"] in hub_layers
            if node["id"] in hub_lookup:
                h = hub_lookup[node["id"]]
                enriched["lift"] = h.lift
                enriched["hub_score"] = h.hub_score
            graph["nodes"].append(enriched)

        # Compute graph metrics if networkx available
        if HAS_NETWORKX and hasattr(self, "_nx_graph") and self._nx_graph:
            try:
                betweenness = nx.betweenness_centrality(self._nx_graph)
                for node in graph["nodes"]:
                    node["betweenness_centrality"] = betweenness.get(node["id"], 0.0)
            except Exception:
                pass

        with open(output_path, "w") as f:
            json.dump(graph, f, indent=2)

        print(f"[Topology] Wrote {len(graph['nodes'])} nodes, {len(graph['edges'])} edges to {output_path}")
        return graph

    def summary(self) -> str:
        """Print a text summary of the topology."""
        if not self._results:
            return "No results loaded."

        if self._graph is None:
            self.build_coactivation_graph()

        hubs = self.find_reasoning_hubs()

        lines = [
            f"Topology: {len(self._results)} results, "
            f"{len(self._graph['nodes'])} active layers, "
            f"{len(self._graph['edges'])} co-activation edges",
        ]

        if hubs:
            lines.append("Reasoning hubs (layers most associated with deep processing):")
            for h in hubs[:5]:
                lines.append(
                    f"  L{h.layer}: lift={h.lift:.2f}x "
                    f"(deep={h.frequency_high_depth:.0%}, shallow={h.frequency_low_depth:.0%})"
                )

        return "\n".join(lines)
