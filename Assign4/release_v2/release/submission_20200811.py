from cv2 import threshold
import util, math, random
from collections import defaultdict
from util import ValueIteration


############################################################
# Problem 1a: BlackjackMDP

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        # we should divide cases by action take, quit, peek
        totalCardValueInHand,nextCardIndexIfPeeked,deckCardCounts=state
        returnList=[]       # (newState, prob, reward)
        prob = 1
        nextReward = 0
        nextState=(0,None,None)

        if deckCardCounts is None:      # It means game is over
            return []

        elif action =='Take':            # If action is take
            if nextCardIndexIfPeeked is None:   # no peeked card before state
                 for card_num,item in enumerate(deckCardCounts):
                    if item>0:
                        prob = float(item) / sum(deckCardCounts)    # the number of card / total card number
                        card_counting = list(deckCardCounts)
                        card_counting[card_num] = card_counting[card_num] - 1   # draw update
                        ValueInHand = totalCardValueInHand + self.cardValues[card_num]
                        # reward no update
                        if sum(card_counting)==0:               # last card drawing
                            nextState=(ValueInHand,None,None)
                            nextReward=ValueInHand
                        elif ValueInHand>self.threshold:        # crossing the threshold 
                            nextState=(ValueInHand,None,None)
                        else:
                            nextState=(ValueInHand,None,tuple(card_counting))
                        returnList.append((nextState,prob,nextReward))
            else:   # peeked card exist
                card_counting=list(deckCardCounts) # update peeked card before state
                card_counting[nextCardIndexIfPeeked] = card_counting[nextCardIndexIfPeeked] - 1
                ValueInHand=self.cardValues[nextCardIndexIfPeeked]+totalCardValueInHand

                if sum(card_counting)==0:               # last card drawing
                    nextState=(ValueInHand,None,None)
                    nextReward=ValueInHand
                elif ValueInHand>self.threshold:        # crossing the threshold 
                    nextState=(ValueInHand,None,None)
                else:
                    nextState=(ValueInHand,None,tuple(card_counting))
                returnList.append((nextState,prob,0))    # deterministically 

        elif action=='Peek':        # if peek a card
            if nextCardIndexIfPeeked is None:   # not possible to peek twice in a row
                for card_num,item in enumerate(deckCardCounts):
                    if item>0:
                        prob = float(item) / sum(deckCardCounts)  # the number of card / total card number
                        nextState = (totalCardValueInHand,card_num,deckCardCounts)
                        nextReward =- self.peekCost
                        returnList.append((nextState,prob,nextReward))
            else:           # nextCardIfPeeked exist, not possible to peek twice in a row
                return []

        elif sum(deckCardCounts) == 0 or action == 'Quit':      # If quit, or card is none, first and thir case of PDF
            if totalCardValueInHand > self.threshold:           # crossing threshold : no reward
                nextReward=0
            else:
                nextReward = totalCardValueInHand
            nextState = (0,None,None)
            returnList.append((nextState,1,nextReward))

        return returnList
        
        # END_YOUR_ANSWER

    def discount(self):
        return 1

############################################################
# Problem 1b: ValueIterationDP

class ValueIterationDP(ValueIteration):
    '''
    Solve the MDP using value iteration with dynamic programming.
    '''
    def solve(self, mdp):
        V = {}  # state -> value of state

        # BEGIN_YOUR_ANSWER (our solution is 13 lines of code, but don't worry if you deviate from this)
        
        while True:
            newV = {}
            for state in mdp.states:
                newV[state] = max(self.computeQ(mdp, V, state, action) for action in mdp.actions(state))
            if max(abs(V[state] - newV[state]) for state in mdp.states) < 0.001:
                V = newV
                break
            V = newV

        # END_YOUR_ANSWER
        
        # Compute the optimal policy now
        pi = self.computeOptimalPolicy(mdp, V)
        self.pi = pi
        self.V = V

############################################################
# Problem 2a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        V_opt = 0                       # evaluate and improve policy at the same time
        if newState is not None:
            V_opt = max([self.getQ(newState,acting) for acting in self.actions(newState)])
        Q_now = self.getQ(state,action)
        updating = self.getStepSize()*((reward+self.discount*V_opt) - Q_now)
        # alpha * (r + discount*max(Q(s', a')) - Q(s, a))
        for item in self.featureExtractor(state,action):
            key,value=item
            self.weights[key] = updating * value + self.weights[key]
        # END_YOUR_ANSWER



############################################################
# Problem 2b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        Q_next = 0                       # evaluate and improve policy at the same time
        if newState is not None:
            Q_next = ([self.getQ(newState,action)])
        Q_now=self.getQ(state,action)
        updating = self.getStepSize() * ((reward + self.discount * Q_next) - Q_now) # sarsa just update the evaluation of current policy pi
        for item in self.featureExtractor(state,action):
            key,value=item
            self.weights[key] = updating * value + self.weights[key]
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 2c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    feature_list=[]
    feature_val = 1
    feature_key = (total, action)
    feature_list.append((feature_key, feature_val))       # indicator on the total and the action

    if counts is not None:                              # deck != none, add feature
        countsList = list(counts)
        
        for index,item in enumerate(counts):           
            feature_key = (index,item,action)
            feature_val = 1
            feature_list.append((feature_key, feature_val))
            if item>0:
                countsList[index]=1
        feature_key = (tuple(countsList), action)
        feature_val = 1
        feature_list.append((feature_key,feature_val))       # presence / absence of each card and the action
        
    return feature_list
    # END_YOUR_ANSWER
