from collections import defaultdict
from typing import Dict, List, Tuple, Optional

class CommittedNGramTable:
    """
    Tabela N-Gram ultra-rápida residente em SRAM/RAM.
    Opera estritamente sobre tokens committed confirmados pelo Target Verify.
    Nunca ingere tokens especulativos rejeitados.
    """
    def __init__(self, n: int = 2, max_continuation: int = 4):
        self.n = n
        self.max_continuation = max_continuation
        self.table = defaultdict(lambda: defaultdict(int))
        self.history: List[int] = []

    def update_many(self, accepted_tokens: List[int]):
        """
        Ingere uma lista ordenada de tokens confirmados e atualiza as contagens
        de n-gramas para todos os comprimentos de continuação até max_continuation.
        """
        for tok in accepted_tokens:
            self.history.append(tok)
            l = len(self.history)
            if l >= self.n + 1:
                for k in range(1, self.max_continuation + 1):
                    if l >= self.n + k:
                        prefix = tuple(self.history[l - self.n - k : l - k])
                        cont = tuple(self.history[l - k : l])
                        self.table[prefix][cont] += 1

    def lookup(self, prefix: Tuple[int, ...], min_frequency: int = 1) -> List[int]:
        """
        Retorna a melhor continuação conhecida para o prefixo dado,
        priorizando maior frequência e, em caso de empate, maior comprimento.
        """
        if prefix in self.table:
            conts = self.table[prefix]
            if conts:
                best_cont, count = max(conts.items(), key=lambda x: (x[1], len(x[0])))
                if count >= min_frequency:
                    return list(best_cont)
        return []

    def lookup_adaptive(self, prefix: Tuple[int, ...], max_depth: int = 2) -> Tuple[List[int], int]:
        """
        Retorna a continuação adaptativa baseada na frequência:
        - Frequência >= 2: expande até max_depth tokens.
        - Frequência == 1: expande 1 token.
        - Frequência == 0: sem expansão (retorna []).
        - Guarda anti-repetição: rejeita continuações degeneradas de tokens repetidos.
        """
        # Se os últimos 2 tokens do histórico forem idênticos, não expandir
        if len(self.history) >= 2 and self.history[-1] == self.history[-2]:
            return [], 0
            
        # Se todos os tokens do prefixo forem idênticos, não expandir
        if len(prefix) > 1 and len(set(prefix)) == 1:
            return [], 0

        if prefix in self.table:
            conts = self.table[prefix]
            if conts:
                best_cont, count = max(conts.items(), key=lambda x: (x[1], len(x[0])))
                # Se a continuação for o mesmo token repetido, descartar
                if len(set(best_cont)) == 1 and best_cont[0] == prefix[-1]:
                    return [], 0
                if count >= 2:
                    return list(best_cont[:max_depth]), count
                elif count == 1:
                    return list(best_cont[:1]), count
        return [], 0

    def clear(self):
        """Limpa a tabela e o histórico."""
        self.table.clear()
        self.history.clear()
