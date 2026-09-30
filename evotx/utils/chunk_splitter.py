import json
import traceback
import networkx as nx


class SmartChunker:
    
    def __init__(self, max_nodes=200):
        self.max_nodes = max_nodes
        
    def chunk_transaction_log(self, token_transfer):
        if len(token_transfer['edges']) < self.max_nodes:
            return [token_transfer]
        G = nx.node_link_graph(token_transfer,link='edges')
        node_importance = self._calculate_node_importance(G)        
        communities = list(nx.weakly_connected_components(G))
        
        refined_communities = []
        for community in communities:
            if len(community) > self.max_nodes:
                subcommunities = self._split_large_community(G, community, self.max_nodes)
                refined_communities.extend(subcommunities)
            else:
                refined_communities.append(community)
        communities = refined_communities
        
        # Calculate community importance
        community_importance = []
        for i, community in enumerate(communities):
            total_importance = sum(node_importance.get(node, 0) for node in community)
            importance_score = total_importance * (1 + 0.1 * min(len(community), 50))
            community_importance.append((i, importance_score))
        sorted_communities = sorted(community_importance, key=lambda x: x[1], reverse=True)
        
        # Create chunks
        chunks = []
        current_chunk_nodes = set()
        current_chunk_edges = []
        
        for comm_idx, _ in sorted_communities:
            community = communities[comm_idx]
            
            # If adding this community would exceed max_nodes, finalize current chunk
            if len(current_chunk_nodes) + len(community) > self.max_nodes and current_chunk_nodes:
                chunk = self._create_chunk(token_transfer, current_chunk_nodes, current_chunk_edges)
                chunks.append(chunk)
                current_chunk_nodes = set()
                current_chunk_edges = []
            
            # Add community to current chunk
            current_chunk_nodes.update(community)
            
            # Add edges within this community and connecting to existing nodes in chunk
            for edge in token_transfer['edges']:
                source, target = edge['source'], edge['target']
                if source in community and target in community:
                    # Both nodes in the community - include edge
                    current_chunk_edges.append(edge)
                elif (source in community and target in current_chunk_nodes) or \
                    (target in community and source in current_chunk_nodes):
                    # Edge connects this community to nodes already in the chunk
                    current_chunk_edges.append(edge)
        
        # Add final chunk if not empty
        if current_chunk_nodes:
            chunk = self._create_chunk(token_transfer, current_chunk_nodes, current_chunk_edges)
            chunks.append(chunk)
        
        return chunks
        
    def _split_large_community(self, G: nx.DiGraph, community, max_size):
        subgraph = G.subgraph(community)
        try:
            nodes_by_degree = sorted(
                community, 
                key=lambda node: subgraph.degree(node), 
                reverse=True
            )
            subcommunities = []
            assigned_nodes = set()
            for center_node in nodes_by_degree:
                if center_node in assigned_nodes:
                    continue
                subcommunity = {center_node}
                assigned_nodes.add(center_node)
                neighbors = set(subgraph.neighbors(center_node)) - assigned_nodes
                for neighbor in sorted(neighbors, key=lambda n: subgraph.degree(n), reverse=True):
                    if len(subcommunity) >= max_size:
                        break
                        
                    subcommunity.add(neighbor)
                    assigned_nodes.add(neighbor)
                
                subcommunities.append(subcommunity)
                
                # If we've assigned all nodes, stop
                if len(assigned_nodes) == len(community):
                    break
            
            # Assign any remaining nodes to subcommunities
            remaining_nodes = community - assigned_nodes
            if remaining_nodes:
                # Distribute remaining nodes across existing subcommunities
                for node in remaining_nodes:
                    # Find subcommunity with fewest nodes
                    smallest_subcomm_idx = min(
                        range(len(subcommunities)), 
                        key=lambda i: len(subcommunities[i])
                    )
                    subcommunities[smallest_subcomm_idx].add(node)
            
            return subcommunities
        except Exception as e:
            pass
        
        # Option 3: Last resort - simple chunking
        subcommunities = []
        nodes_list = list(community)
        for i in range(0, len(nodes_list), max_size):
            subcommunity = set(nodes_list[i:i+max_size])
            subcommunities.append(subcommunity)
        
        return subcommunities

    def _calculate_node_importance(self, G: nx.DiGraph):
        importance = {}
        for node in G.nodes():
            in_deg = G.in_degree(node)
            out_deg = G.out_degree(node)
            importance[node] = 0.6 * in_deg + 0.4 * out_deg
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            for node, score in pagerank.items():
                importance[node] = importance.get(node, 0) + 10 * score  
        except:
            traceback.print_exc()
        max_importance = max(importance.values()) if importance else 1.0
        for node in importance:
            importance[node] /= max_importance
        return importance

    def _create_chunk(self, token_transfer, nodes, edges):
        # Filter nodes
        chunk_nodes = [
            node for node in token_transfer['nodes']
            if node['id'] in nodes
        ]
        
        # Create the chunk
        chunk = {
                'directed': token_transfer.get('directed', True),
                'multigraph': token_transfer.get('multigraph', False),
                'graph': token_transfer.get('graph', {}),
                'nodes': chunk_nodes,
                'edges': edges
            }
       
        return chunk      