# coding: utf-8

import numpy as np
import random
import math
import json

import networkx as nx
from EduSim.Envs.meta import MetaLearner, MetaInfinityLearnerGroup, MetaLearningModel, Item
from EduSim.Envs.shared.KSS_KES.KS import influence_control

__all__ = ["Learner", "LearnerGroup"]


class LearningModel(MetaLearningModel):
    def __init__(self, state, learning_target, knowledge_structure, last_visit=None):
        self._state = state
        self._target = learning_target
        self._ks = knowledge_structure
        self._ks_last_visit = last_visit

    def step(self, state, knowledge):

        if self._ks_last_visit is not None:
            if knowledge not in influence_control(
                    self._ks, state, self._ks_last_visit, allow_shortcut=False, target=self._target,
            )[0]:
                return
        self._ks_last_visit = knowledge

        # capacity growth function
        discount = math.exp(sum([(5 - state[node-1 if node == 835 else node]) for node in self._ks.predecessors(knowledge)] + [0]))
        ratio = 1 / discount
        inc = (5 - state[knowledge]) * ratio * 0.5

        def _promote(_ind, _inc):
            state[_ind-1] += _inc
            if state[_ind-1] > 5:
                state[_ind-1] = 5
            for node in self._ks.successors(_ind):
                _promote(node, _inc * 0.5)

        # _promote(knowledge, inc)


class Learner(MetaLearner):
    def __init__(self,
                 initial_state,
                 knowledge_structure: nx.DiGraph,
                 learning_target: set,
                 _id=None,
                 seed=None):
        super(Learner, self).__init__(user_id=_id)

        self.learning_model = LearningModel(
            initial_state,
            learning_target,
            knowledge_structure,
        )

        self.structure = knowledge_structure
        self._state = initial_state
        self._target = learning_target
        self._logs = []
        self.random_state = np.random.RandomState(seed)

    def update_logs(self, logs):
        self._logs = logs

    @property
    def profile(self):
        return {
            "id": self.id,
            "logs": self._logs,
            "target": self.target
        }

    def learn(self, learning_item: Item):
        self.learning_model.step(self._state, learning_item.knowledge)

    @property
    def state(self):
        return self._state

    def response(self, test_item: Item) -> ...:
        if test_item.knowledge == 442:
            return self._state[test_item.knowledge-1]
        return self._state[test_item.knowledge]

    @property
    def target(self):
        return self._target


class LearnerGroup(MetaInfinityLearnerGroup):
    def __init__(self, knowledge_structure, seed=None):
        super(LearnerGroup, self).__init__()
        self.knowledge_structure = knowledge_structure
        self.random_state = np.random.RandomState(seed)

    def __next__(self):
        knowledge = self.knowledge_structure.nodes



        # self.knowledge_structure = Know_G

        random_list = [random.uniform(-3, 0.5) for _ in range(835)]

        # Sort the list in descending order
        sorted_list = sorted(random_list, reverse=True)

        # Perform calculations on the sorted list
        result = [num - (0.1 * i) for i, num in enumerate(sorted_list)]

        return Learner(
            result,
            # [self.random_state.randint(-3, 0) - (0.1 * i) for i, _ in enumerate(knowledge)],

            self.knowledge_structure,
            # Know_G,
            # set(self.random_state.choice(len(knowledge)+1, self.random_state.randint(1, int(0.5 * len(knowledge))))),
            set(self.random_state.choice(835, self.random_state.randint(1, 835))),

        )
