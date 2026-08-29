import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

@dataclass
class TreeNode:
    node_id: int
    token_id: int
    parent_id: int
    depth: int
    branch_id: int
    active: bool = True

class CandidateTree:
    """
    Estrutura canônica de árvore de candidatos com topologia hierárquica explícita.
    Permite a representação de múltiplos ramos disjuntos a partir de uma raiz comum
    sem achatamento linear inválido.
    """
    def __init__(self, root_token: int):
        self.root_token = root_token
        self.nodes: List[TreeNode] = [
            TreeNode(node_id=0, token_id=root_token, parent_id=-1, depth=0, branch_id=-1)
        ]
        self.branch_map: Dict[int, List[int]] = defaultdict(list)

    def add_candidate(self, token_id: int, parent_id: int, branch_id: int) -> int:
        parent = self.nodes[parent_id]
        new_id = len(self.nodes)
        node = TreeNode(
            node_id=new_id,
            token_id=token_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            branch_id=branch_id
        )
        self.nodes.append(node)
        self.branch_map[branch_id].append(new_id)
        return new_id

    def get_branch_sequence(self, branch_id: int) -> List[int]:
        node_ids = [0] + self.branch_map[branch_id]
        return [self.nodes[nid].token_id for nid in node_ids]

    def get_all_branches(self) -> List[List[int]]:
        if not self.branch_map:
            return [[self.root_token]]
        return [self.get_branch_sequence(b_id) for b_id in sorted(self.branch_map.keys())]

    def to_batched_tensor(self, device: str = "cuda:0") -> Tuple[torch.Tensor, List[int]]:
        branches = self.get_all_branches()
        b_lens = [len(b) for b in branches]
        max_len = max(b_lens)
        B = len(branches)
        
        tensor = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i, branch in enumerate(branches):
            tensor[i, :len(branch)] = torch.tensor(branch, dtype=torch.long, device=device)
            if len(branch) < max_len:
                tensor[i, len(branch):] = branch[-1]
                
        return tensor, b_lens
