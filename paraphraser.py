from itertools import permutations
from random import sample
from typing import Callable
from nltk import ParentedTree


class Paraphraser:
    tree: ParentedTree
    limit: int
    paraphrased_trees: list[ParentedTree]
    paraphrase_methods: dict[str, str] = {
        'Noun phrases': 'paraphrase_noun_phrases'
    }
    subtrees_to_paraphrase_indexes: list[tuple[int]]
    shuffled_subtrees: list[list[ParentedTree]]

    def __init__(self, tree: str, limit: int):
        self.tree = ParentedTree.fromstring(tree)
        self.paraphrased_trees = [self.tree]
        self.limit = limit

    def paraphrase(self, methods: list[str]) -> list[ParentedTree] | str:
        for method in methods:
            method_name = self.paraphrase_methods.get(method, None)

            if not method_name:
                return f'There is no method for paraphrasing by {method}'

            method_func = getattr(self, method_name)

            trees_to_paraphrase = list(self.paraphrased_trees)
            self.paraphrased_trees = []
            for tree in trees_to_paraphrase:
                self.paraphrased_trees.extend(method_func(tree))

        return sample(self.paraphrased_trees, self.limit) if self.limit else self.paraphrased_trees

    def paraphrase_noun_phrases(self, tree: ParentedTree) -> list[ParentedTree]:
        def condition(root: ParentedTree) -> bool:
            subtree_labels = [subtree.label() for subtree in root]
            return root.label() == 'NP' and all((label in ['NP', 'CC', ','] for label in subtree_labels))

        self.subtrees_to_paraphrase_indexes = []
        self.search_nodes(tree, condition)

        if len(self.subtrees_to_paraphrase_indexes) == 0:
            return [tree]

        self.shuffled_subtrees = []
        for subtree_index in self.subtrees_to_paraphrase_indexes:
            subtree = self.tree[subtree_index]
            self.shuffled_subtrees.append(self.shuffle_subtrees_by_label(subtree, 'NP'))

        return self.combine_shuffled_subtrees()

    def search_nodes(self, root: ParentedTree, condition: Callable[[ParentedTree], bool]):
        if root.height() <= 2:
            return

        if condition(root):
            self.subtrees_to_paraphrase_indexes.append(root.treeposition())

        for subtree in root:
            self.search_nodes(subtree, condition)

    def shuffle_subtrees_by_label(self, root: ParentedTree, label: str) -> list[ParentedTree]:
        labeled_subtrees_indexes = []
        i = 0
        for subtree in root:
            if subtree.label() == label:
                labeled_subtrees_indexes.append(i)
            i += 1

        trees_with_shuffled_subtrees = []
        for index_permutation in list(permutations(labeled_subtrees_indexes)):
            subtrees = []
            j = 0
            for i, subtree in enumerate(root):
                if i in labeled_subtrees_indexes:
                    subtrees.append(root[index_permutation[j]].copy(deep=True))
                    j += 1
                else:
                    subtrees.append(subtree.copy(deep=True))

            trees_with_shuffled_subtrees.append(ParentedTree(root.label(), subtrees))

        return trees_with_shuffled_subtrees

    def combine_shuffled_subtrees(self) -> list[ParentedTree]:
        shuffled_subtrees_combinations = [[subtree] for subtree in self.shuffled_subtrees[0]]

        for i in range(1, len(self.subtrees_to_paraphrase_indexes)):
            temp_shuffled_subtrees_combinations = []
            for subtree in self.shuffled_subtrees[i]:
                for combination in shuffled_subtrees_combinations:
                    temp_shuffled_subtrees_combinations.append(combination + [subtree])

            shuffled_subtrees_combinations = list(temp_shuffled_subtrees_combinations)

        paraphrased_trees = []
        for combination in shuffled_subtrees_combinations:
            tree = self.tree.copy(deep=True)
            for new_subtree, index in zip(combination, self.subtrees_to_paraphrase_indexes):
                subtree = tree[index]
                subtree_parent = subtree.parent()
                subtree_parent.remove(subtree)
                subtree_parent.insert(index[-1], new_subtree.copy(deep=True))

            paraphrased_trees.append(tree)

        return paraphrased_trees
