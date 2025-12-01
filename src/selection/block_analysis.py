import torch
import torch.nn as nn
from collections import defaultdict, deque
import networkx as nx
from typing import Dict, List, Set, Tuple

class BlockAnalyzer:
    """
    Analyzes model topology to identify contiguous computational blocks.
    Prevents scattered quantization that creates excessive format conversions.
    """
    
    def __init__(self, model):
        self.model = model
        self.graph = None
        self.blocks = []
        self.layer_to_block = {}
        
    def build_dependency_graph(self) -> nx.DiGraph:
        """
        Constructs a directed graph of layer dependencies.
        Nodes = Linear layers, Edges = data flow connections.
        """
        G = nx.DiGraph()
        
        # Step 1: Add all Linear layers as nodes
        linear_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module
                G.add_node(name)
        
        # Step 2: Trace execution to find edges
        # Use forward hooks to detect which layer outputs feed into which inputs
        connections = defaultdict(set)
        layer_outputs = {}
        
        def make_hook(layer_name):
            def hook(module, input, output):
                # Store output tensor ID
                layer_outputs[layer_name] = id(output)
                
                # Check if input came from another tracked layer
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                input_id = id(input_tensor)
                for prev_name, prev_output_id in layer_outputs.items():
                    if prev_output_id == input_id and prev_name != layer_name:
                        connections[prev_name].add(layer_name)
            return hook
        
        # Register hooks
        hooks = []
        for name, module in linear_layers.items():
            hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Run dummy forward pass
        try:
            dummy_input = torch.randint(0, 1000, (1, 16), dtype=torch.long)
            dummy_input = dummy_input.to(next(self.model.parameters()).device)
            with torch.no_grad():
                self.model(dummy_input)
        except Exception as e:
            print(f"[WARNING] Dummy forward pass failed: {e}")
            print("[INFO] Falling back to structural analysis")
            # Fallback: use model structure (transformer blocks, etc.)
            connections = self._structural_fallback(linear_layers)
        finally:
            # Remove hooks
            for h in hooks:
                h.remove()
        
        # Add edges to graph
        for src, targets in connections.items():
            for tgt in targets:
                G.add_edge(src, tgt)
        
        self.graph = G
        return G
    
    def _structural_fallback(self, linear_layers: Dict) -> Dict[str, Set[str]]:
        """
        Fallback method: infer connections from module hierarchy.
        Assumes layers in same parent module are connected sequentially.
        """
        connections = defaultdict(set)
        
        # Group layers by parent module
        parent_groups = defaultdict(list)
        for name in linear_layers.keys():
            if '.' in name:
                parent = '.'.join(name.split('.')[:-1])
            else:
                parent = 'root'
            parent_groups[parent].append(name)
        
        # Connect layers within each parent sequentially
        for parent, layers in parent_groups.items():
            sorted_layers = sorted(layers)  # Alphabetical = typical execution order
            for i in range(len(sorted_layers) - 1):
                connections[sorted_layers[i]].add(sorted_layers[i + 1])
        
        return connections
    
    def identify_blocks(self, method: str = "weakly_connected") -> List[List[str]]:
        """
        Partition the graph into computational blocks.
        
        Methods:
        - 'weakly_connected': Standard graph components (best for most models)
        - 'transformer_blocks': Detect transformer layer boundaries
        - 'depth_based': Group by network depth
        """
        if self.graph is None:
            self.build_dependency_graph()
        
        if method == "weakly_connected":
            # Find weakly connected components (ignores edge direction)
            components = list(nx.weakly_connected_components(self.graph))
            self.blocks = [sorted(list(comp)) for comp in components]
        
        elif method == "transformer_blocks":
            self.blocks = self._detect_transformer_blocks()
        
        elif method == "depth_based":
            self.blocks = self._group_by_depth()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create layer -> block mapping
        self.layer_to_block = {}
        for block_id, block in enumerate(self.blocks):
            for layer in block:
                self.layer_to_block[layer] = block_id
        
        print(f"[BLOCK ANALYSIS] Identified {len(self.blocks)} computational blocks")
        return self.blocks
    
    def _detect_transformer_blocks(self) -> List[List[str]]:
        """
        Specialized detection for transformer architectures.
        Groups layers by transformer block (e.g., model.layers.0.*, model.layers.1.*, etc.)
        """
        blocks = defaultdict(list)
        
        for node in self.graph.nodes():
            # Extract transformer block number from name
            # Common patterns: "model.layers.0.mlp.gate_proj", "transformer.h.5.attn.c_proj"
            parts = node.split('.')
            
            # Find the layer index
            block_idx = None
            for i, part in enumerate(parts):
                if part.isdigit():
                    block_idx = int(part)
                    break
            
            if block_idx is not None:
                blocks[block_idx].append(node)
            else:
                blocks[-1].append(node)  # Outliers (embeddings, final layer, etc.)
        
        return [sorted(layers) for layers in blocks.values()]
    
    def _group_by_depth(self, max_depth: int = 5) -> List[List[str]]:
        """
        Groups layers by their depth in the computation graph.
        Useful for non-transformer architectures.
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            print("[WARNING] Graph has cycles, using topological generations")
        
        try:
            # Get layers at each depth
            depth_groups = defaultdict(list)
            for node in nx.topological_sort(self.graph):
                # Compute depth as longest path from any root
                depth = nx.dag_longest_path_length(self.graph.subgraph(
                    nx.ancestors(self.graph, node) | {node}
                ))
                depth_groups[depth // max_depth].append(node)
            
            return [sorted(layers) for layers in depth_groups.values()]
        except:
            # Fallback: just group by name prefix
            return self._detect_transformer_blocks()
    
    def get_block_sensitivity(self, sensitivity_map: Dict[str, float]) -> Dict[int, float]:
        """
        Compute aggregate sensitivity for each block.
        Uses mean sensitivity of layers in the block.
        """
        block_scores = {}
        
        for block_id, layers in enumerate(self.blocks):
            scores = [sensitivity_map.get(layer, 0.0) for layer in layers]
            if scores:
                block_scores[block_id] = sum(scores) / len(scores)
            else:
                block_scores[block_id] = 0.0
        
        return block_scores
    
    def select_blocks_to_keep(self, 
                              sensitivity_map: Dict[str, float],
                              keep_ratio: float = 0.3) -> Set[str]:
        """
        Select entire blocks to keep in FP16 based on aggregate sensitivity.
        
        Args:
            sensitivity_map: Layer sensitivity scores
            keep_ratio: Fraction of blocks to keep in FP16 (0.0 to 1.0)
        
        Returns:
            Set of layer names to keep in FP16
        """
        if not self.blocks:
            self.identify_blocks()
        
        # Compute block sensitivities
        block_scores = self.get_block_sensitivity(sensitivity_map)
        
        # Sort blocks by sensitivity
        sorted_blocks = sorted(block_scores.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
        
        # Select top blocks
        num_keep = max(1, int(len(sorted_blocks) * keep_ratio))
        blocks_to_keep = [block_id for block_id, _ in sorted_blocks[:num_keep]]
        
        # Get all layers in selected blocks
        layers_to_keep = set()
        for block_id in blocks_to_keep:
            layers_to_keep.update(self.blocks[block_id])
        
        print(f"[BLOCK SELECTION] Keeping {len(blocks_to_keep)}/{len(self.blocks)} blocks ({len(layers_to_keep)} layers)")
        return layers_to_keep
    
    def visualize_blocks(self, sensitivity_map: Dict[str, float] = None, 
                        output_path: str = None):
        """
        Creates a visualization of the block structure.
        Requires matplotlib and graphviz.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("[WARNING] matplotlib not available, skipping visualization")
            return
        
        if not self.blocks:
            self.identify_blocks()
        
        # Create color map
        if sensitivity_map:
            block_scores = self.get_block_sensitivity(sensitivity_map)
            max_score = max(block_scores.values()) if block_scores else 1.0
            colors = {block_id: score / max_score 
                     for block_id, score in block_scores.items()}
        else:
            colors = {i: 0.5 for i in range(len(self.blocks))}
        
        # Simple text-based visualization for now
        print("\n" + "="*80)
        print("BLOCK STRUCTURE VISUALIZATION")
        print("="*80)
        
        for block_id, layers in enumerate(self.blocks):
            score = colors.get(block_id, 0.0)
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            print(f"\nBlock {block_id:2d} [{bar}] {score:.3f}")
            print(f"  Layers ({len(layers)}): {', '.join(layers[:3])}{'...' if len(layers) > 3 else ''}")


def block_based_quantization(model, 
                             tokenizer,
                             sensitivity_map: Dict[str, float],
                             keep_ratio: float = 0.3,
                             block_method: str = "transformer_blocks") -> Set[str]:
    """
    High-level wrapper for block-based layer selection.
    
    Args:
        model: The neural network model
        tokenizer: Tokenizer (needed for dummy forward pass)
        sensitivity_map: Layer-wise sensitivity scores
        keep_ratio: Fraction of blocks to keep in FP16
        block_method: Method for identifying blocks
    
    Returns:
        Set of layer names to keep in FP16
    """
    analyzer = BlockAnalyzer(model)
    analyzer.identify_blocks(method=block_method)
    
    # Optional: visualize
    analyzer.visualize_blocks(sensitivity_map)
    
    # Select blocks
    layers_to_keep = analyzer.select_blocks_to_keep(sensitivity_map, keep_ratio)
    
    return layers_to_keep
