import copy
from datetime import datetime
import numpy as np
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from tianshou.data.buffer.base import ReplayBuffer


class Policy:
    """Base policy

    A policy is something giving an action probability distribution given a state of the game
    """

    def __init__(self, player_id, num_actions):
        """Initial a policy

        Args:
            player_id: An id identify the player
            num_actions: The total amount of action in the game
        """

        self._player_id = player_id
        self._num_actions = num_actions

    def action_probabilities(self, state, legal_action_mask=None):
        """ Return a dictionary {action:probability} for all actions

        Args:
            state: A game state
            legal_action_mask: A list show whether an action legal of all actions
            if None, all actions are legal
        Returns:
            A `dict` of `{action:probability}` for the giving game state
        """
        return NotImplemented()

    def choose_action(self, state, legal_action_mask=None):
        """ Return the chosen action by the policy

        Args:
            state: A game state
            legal_action_mask (_type_, optional): A list show whether an action legal of all actions
            if None, all actions are legal
        Returns:
            index of action
        """
        return NotImplemented()

    def __call__(self, state, legal_action_mask=None):
        """Turns the policy into a callable
        Args:
            state: A game state

        Returns:
            A `dict` of `{action:probability}` for the giving game state
        """
        return self.choose_action(state, legal_action_mask)


class RandomPolicy(Policy):

    def __init__(self, player_id, num_actions):
        super(RandomPolicy, self).__init__(player_id, num_actions)
    
    def action_probabilities(self, state, legal_action_mask=None):
        return self.equal_probabilities(state, legal_action_mask)
    
    def equal_probabilities(self, state, legal_action_mask=None):
        """

        Args:
            state:
            legal_action_mask:

        Returns:
            Uniform random policy, contain all legal actions, each with the same probability
        """
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]

        if np.sum(legal_action_mask) == 0:
            return {action: 0 for action in range(self._num_actions)}

        action_probs = np.array(legal_action_mask / np.sum(legal_action_mask))
        return {action: action_probs[action] for action in range(self._num_actions)}

    def choose_action(self, state, legal_action_mask=None):
        action_probs = self.action_probabilities(state, legal_action_mask)
        return np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))


class TabularPolicy(RandomPolicy):
    """Tabular policy

    Tabular policy use a table to store the probability of choosing an action given a state of the game
    """

    def __init__(self, player_id, num_actions):
        super().__init__(player_id, num_actions)

        self.state_lookup = {}
        self.action_probability_array = np.ndarray((0, self._num_actions))
        self.legal_action_list = []
        self.epsilon = 1E-9

    def action_probabilities(self, state, legal_action_mask=None):
        """

        Args:
            state: A game state
            legal_action_mask: A list show whether an action legal of all actions

        Returns:
            A dictionary {action:probability} given the state
            if the state in tabular, return the record action probabilities
            else, return random action probabilities and record them in tabular
        """
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]

        if np.sum(legal_action_mask) == 0:
            return {action: 0 for action in range(self._num_actions)}

        state_key = self._state_key(state)
        if state_key in self.state_lookup:
            state_index = self.state_lookup[state_key]
            action_probs = self.action_probability_array[state_index]
            action_probs *= legal_action_mask
            if np.sum(action_probs) > 0:
                action_probs /= np.sum(action_probs)
            else:
                action_probs = np.array(
                    legal_action_mask / np.sum(legal_action_mask))
        else:
            action_probs = np.array(
                legal_action_mask / np.sum(legal_action_mask))
            self.set_action_probabilities(
                state, action_probs, legal_action_mask)

        return {action: action_probs[action] for action in range(self._num_actions)}

    def set_action_probabilities(self, state, action_probs, legal_action_mask=None):
        if legal_action_mask is None:
            legal_action_mask = [1 for _ in range(self._num_actions)]
        state_key = self._state_key(state)
        if state_key in self.state_lookup:
            state_index = self.state_lookup[state_key]
            self.legal_action_list[state_index] = legal_action_mask
            self.action_probability_array[state_index] = action_probs
        else:
            state_index = len(self.state_lookup)
            self.state_lookup[state_key] = state_index
            self.legal_action_list.append(legal_action_mask)
            self.action_probability_array = np.append(self.action_probability_array,
                                                      np.array(action_probs).reshape(1, -1), axis=0)

    def _state_key(self, state):
        return repr(state)


class NetPolicy(RandomPolicy):
    """
        A policy implemented by neural network, the action probs could be computing by the neural network, and the policy
        could be optimized by updating the parameters of the neural network
    """

    def __init__(self,
                 player_id,
                 num_actions: int,
                 network: nn.Module,
                 optimizer: optim.Optimizer,
                 buffer_size: int,
                 ):
        super(NetPolicy, self).__init__(player_id, num_actions)
        self._network = network
        self._optimizer = optimizer
        self._train = False
        self._buffer = ReplayBuffer(buffer_size)
        
        now=datetime.now()
        self.writer=SummaryWriter(f"./logs/{now.day}_{now.hour}_{now.minute}")
    
    def choose_action(self, state, legal_action_mask=None):
        action_probs = self.action_probabilities(state, legal_action_mask)
        actions=list(action_probs.keys())
        probs=np.array(list(action_probs.values()))
        action=np.random.choice(actions,p=probs)
        return action

    def copy_network(self):
        return copy.deepcopy(self._network)

    def train_mode(self):
        self._train = True
        self._network.train()

    def eval_mode(self):
        self._train = False
        self._network.eval()
        
    def clear_buffer(self):
        self._buffer.reset()        

    @property
    def network(self):
        return self._network

    @property
    def device(self):
        return next(self._network.parameters()).device

    @property
    def optimizer(self):
        return self._optimizer
    