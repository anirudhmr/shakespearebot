########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import math
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        for j in range(self.L):
            probs[1][j] = np.log(self.A_start[j]) + np.log(self.O[j][x[0]])
            seqs[1][j] = str(j)
        for i in range(2, M+1):
            for j in range(self.L):
                past_probs = []
                for k in range(self.L):
                      past_probs.append(probs[i-1][k] + np.log(self.A[k][j]) + np.log(self.O[j][x[i-1]]))
                best_prob = max(past_probs)
                best_previous_state = past_probs.index(best_prob)
                probs[i][j] = best_prob
                seqs[i][j] = seqs[i-1][best_previous_state] + str(j)
        final_seqs = seqs[:][M]
        final_probs = probs[:][M]
        max_seq = final_seqs[final_probs.index(max(final_probs))]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        for i in range(1, M + 1):
            if i == 1:
                for j in range(self.L):
                    alphas[i][j] = self.A_start[j] * self.O[j][x[i-1]]
                continue
            for j in range(self.L):
                p_x = self.O[j][x[i-1]]
                sums = 0
                for k in range(self.L):
                    sums += alphas[i-1][k] * self.A[k][j]
                res = p_x * sums
                alphas[i][j] = res
            if normalize:
                for row in range(len(alphas)):
                    C = sum(alphas[row])
                    if C == 0:
                        C = 1
                    for col in range(len(alphas[row])):
                        alphas[row][col] /= C
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        for i in range(self.L):
            betas[M][i] = 1
        for i in range(M - 1, -1, -1):
            for j in range(self.L):
                sums = 0
                for k in range(self.L):
                    sums += betas[i+1][k] * self.A[j][k] * self.O[k][x[i]]
                betas[i][j] = sums
            if normalize:
                for row in range(len(betas)):
                    C = sum(betas[row])
                    if C == 0:
                        C = 1
                    for col in range(len(betas[row])):
                        betas[row][col] /= C
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        for i in range(self.L):
            for j in range(self.L):
                self.A[i][j] = 0
            for j in range(self.D):
                self.O[i][j] = 0
        totals = [0. for _ in range(self.L)]
        for i in range(len(Y)):
            for j in range(len(Y[i])-1):
                totals[int(Y[i][j])] += 1.
                self.A[int(Y[i][j])][int(Y[i][j+1])] += 1.
        for i in range(self.L):
            for j in range(self.L):
                self.A[i][j] /= totals[i]


        # Calculate each element of O using the M-step formulas.
        totals = [0. for _ in range(self.L)]
        for i in range(len(X)):
            for j in range(len(X[i])):
                totals[int(Y[i][j])] += 1.
                self.O[int(Y[i][j])][int(X[i][j])] += 1.
        for i in range(self.L):
            for j in range(self.D):
                self.O[i][j] /= totals[i]



    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        # Marginals for step 6:
        # marg6[i][j][k] is P(y^k = j, x_i)
        marg6 = [[[0. for _ in range(len(X[i])+1)] for _ in range(self.L)] for i in range(len(X))]

        # Marginals for step 7:
        # marg7[i][j][k][l] is P(y^(l) = j, y^(l+1) = k, x_i)
        marg7 = [[[[0. for _ in range(len(X[i])+1)] for _ in range(self.L)] for _ in range(self.L)] for i in range(len(X))]
        for epoch in range(N_iters):
            # E Step
            print(epoch)
            # Creating marg6
            for k in range(len(X)):
                # alphas[i][j] = alpha_j[i]
                alphas = self.forward(X[k], normalize=True)
                betas = self.backward(X[k], normalize=True)
                for i in range(1, len(X[k])+1):
                    denom = sum([a*b for a, b in zip(alphas[i], betas[i])])
                    #for a_prime in range(self.L):
                    #denom += alphas[i][a_prime] * betas[i][a_prime]
                    for j in range(self.L):
                        numer = alphas[i][j] * betas[i][j]
                        marg6[k][j][i] = numer / denom

            # Creating marg7
            for l in range(len(X)):
                alphas = self.forward(X[l], normalize=True)
                betas = self.backward(X[l], normalize=True)
                for i in range(1, len(X[l])):
                    denom = 0
                    for a_prime in range(self.L):
                        for b_prime in range(self.L):
                            denom += alphas[i][a_prime] * betas[i+1][b_prime] * self.A[a_prime][b_prime] * self.O[b_prime][X[l][i]]
                    for j in range(self.L):
                        for k in range(self.L):
                            numer = alphas[i][j] * betas[i+1][k] * self.O[k][X[l][i]] * self.A[j][k]
                            marg7[l][j][k][i] = numer / denom

            # M Step
            # Updating A
            for b in range(self.L):
                for a in range(self.L):
                    numer = 0
                    denom = 0
                    for i in range(len(X)):
                        for j in range(1, len(X[i])):
                            numer += marg7[i][a][b][j]
                            denom += marg6[i][a][j]

                    self.A[a][b] = numer / denom

            # Updating O
            for w in range(self.D):
                for a in range(self.L):
                    numer = 0
                    denom = 0
                    for i in range(len(X)):
                        for j in range(1, len(X[i])+1):
                            if X[i][j-1] == w:
                                numer += marg6[i][a][j]
                            denom += marg6[i][a][j]
                    self.O[a][w] = numer / denom






    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        curr_y = random.choices(range(self.L), self.A_start)[0]
        curr_x = random.choices(range(self.D), self.O[curr_y])[0]
        emission.append(curr_x)
        states.append(curr_y)
        for i in range(M-1):
            curr_y = random.choices(range(self.L), self.A[curr_y])[0]
            curr_x = random.choices(range(self.D), self.O[curr_y])[0]
            emission.append(curr_x)
            states.append(curr_y)


        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
