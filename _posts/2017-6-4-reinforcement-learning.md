---
layout: post
title: Reinforcement Learning Exercises
---

In this article, I present some solutions to reinforcement learning exercise. These exercises are
taken from the book "Artificial Intelligence A Modern Approach 3rd edition". I will gradually
update this post with new solutions while I'm learning about the field. I don't pretend all my
solutions are 100 % accurate but I think they make sense. If you do find some mistakes, please
let me know.

<div class="blue-color-box">
<b>15.1</b> For the 4 × 3 world shown in Figure 17.1.1, calculate which squares can be reached
from (1,1) by the action sequence [Up, Up, Right, Right, Right] and with what probabilities.
Explain how this computation is related to the prediction task (see Section 15.2.1) for a hidden
Markov model.
</div>
<div class="centered-img">
<img src="../images/rl_exercices/figure-17-1.png" width="400px" alt="Environment" />
<div class="legend">Figure 17.1.1: (a) A simple 4 x 3 environment that presents the agent with a sequential<br>
					decision problem. (b) Illustration of the transition model of the environment: the "intented"<br>outcome occurs with probability 0.8, but with probability 0.2 the agent moves at right angles<br> to the intended direction. A collision with a wall results in no movement. The two terminal<br> states have reward +1 and -1, respectively, and all other states have a reward of -0.04.</div>
</div>

**Answer** Well, this exercise doesn't seem difficult but we surely need lot's of time to
compute all the solutions with their respective probabilities. So, first of all, we are asking to compute the probabilities of being in each square at the end of the sequence _[Up, Up, Right, Right, Right]_. Hence we don't need to use the Ballman equation here, as we only want to compute **probabilities** and not **utilities**.<br>
Obviously, we can do something a little bit smarter than just enumerate all the solutions. Indeed,
we can compute the probability at each time step and reuse these probabilities to compute the probabilities at
the next time step. 
<br>
For example, we compute the probability of reaching each position after doing one **Up**ward
movement and then we reuse these probabilities (first column of Figure 17.2) to compute the probabilities at the second time step (second column of Figure 17.1.2):
<div class="centered-img">
<img src="../images/rl_exercices/17.1-table.png" width="250px" alt="Table" />
<div class="legend">Figure 17.1.2: occupancy probabilities at each time step for the first 2 movements.</div>
</div>

So for example, to compute 0.24<sup>1</sup> in position **(1,2)**, we reused the probabilities of the first column and we multiplied by the probability given in Figure 17.1.

$$
P((1,2) \text{ in step 2}) = P((1,1) \text{ in step 1}) \times P((1,2)|(1,1) \text{ in step 1}) + P((1,2) \text{ in step 1}) \times P((1,2)|(1,2) \text{ in step 1}) + P((2,1) \text{ in step 1}) \times P((1,2)|(2,1) \text{ in step 1}) \\
= 0.1 \times 0.8 + 0.8 \times (0.1 + 0.1) + 0.1 \times 0 = 0.24
$$

So finally we can come up with the full table, with the asked occupancy probabilities in the last column:

<div class="centered-img">
<img src="../images/rl_exercices/17.2-table.png" width="400px" alt="table 2" />
<div class="legend">Figure 17.1.3: Occupancy probabilities at each time step.<br> 
					The probabilities in the last column are the answer to the question.</div>
</div>

This computation is related to the prediction task for a HMM in the sense that we only need to consider the probabilities of the previous positions to compute the probabilities of being in the next positions.


<div class="blue-color-box">
<b>17.2</b> Select a specific member of the set of policies that are optimal for $R(s) > 0$ as shown
in Figure 17.2.1, and calculate the fraction of time the agent spends in each state, in the limit,
if the policy is executed forever. (Hint: Construct the state-to-state transition probability
matrix corresponding to the policy and see Exercise 15.2.)
</div>
<br>
<div class="centered-img">
<img src="../images/rl_exercices/figure-17-2.png" width="300px" alt="Environment" />
<div class="legend">Figure 17.2.1: Optimal policies for $R(s) > 0$. In the squares where there<br>
					are 4 arrows, the agent can decide to go in any of these directions.</div>
</div>


**Answer** According to Figure 17.2.1, we can take whatever policy we want for the squares:
(1,2), (2,1), (3,1), (1,2), (1,3), (2,3). For example, we can choose to always go **right**. We
must note that, again, if we choose to go right the agent will go right with probability 0.8 and it will go
down or up with probability 0.1. Having said that our Transition matrix looks like:

<div class="centered-img">
<img src="../images/rl_exercices/17.2-transition-matrix.png" width="600px" alt="Transition matrix" />
<div class="legend">Figure 17.2.2: Transition Matrix</div>
</div>

So we have to solve the system:

$$
\pi
\begin{bmatrix}
	0.1 & 0.8 & 0 & 0 & 0.1 & 0 & 0 & 0 & 0\\
	0 & 0.2 & 0.8 & 0 & 0 & 0 & 0 & 0 & 0\\
	0 & 0 & 0.1 & 0.8 & 0 & 0.1 & 0 & 0 & 0\\
	0 & 0 & 0.1 & 0.9 & 0 & 0 & 0 & 0 & 0\\
	0.1 & 0 & 0 & 0 & 0.8 & 0 & 0.1 & 0 & 0\\
	0 & 0 & 0.1 & 0 & 0 & 0.8 & 0 & 0 & 0.1\\
	0 & 0 & 0 & 0 & 0.1 & 0 & 0.1 & 0.8 & 0\\
	0 & 0 & 0 & 0 & 0 & 0 & 0 & 0.2 & 0.8\\
	0 & 0 & 0 & 0 & 0 & 0.1 & 0 & 0.8 & 0.1\\
\end{bmatrix}
= \pi
$$

with:

$$\pi = 
\begin{bmatrix}
\pi_{11} & \pi_{21} & \pi_{31} & \pi_{41} & \pi_{12} & \pi_{32} & \pi_{13} & \pi_{23} & \pi_{33}\\
\end{bmatrix}
$$
and
$$
\sum\limits_{i,j} \pi_{ij} = 1
$$

I let you solve this system...

<div class="blue-color-box">
<b>17.3</b> Suppose that we define the utility of a state sequence to be the maximum reward obtained
in any state in the sequence. Show that this utility function does not result in stationary
preferences between state sequences. Is it still possible to define a utility function on states
such that MEU decision making gives optimal behavior?
</div>

**Answer** To understand the problem, let's write it mathematically. The utility function can be written:

$$U(s_0, a_0, s_1, ..., a_n, s_n) = \max\limits_{i=0}^{n-1} R(s_i, a_i, s_{i+1})$$

We say that a utility function meets the stationary property if the result of applying the utility function to
the sequences $[s_1, s_2, ...]$ and $[s_1', s_2', ...]$ leads to the same solution **and** the result of
applying the utility function to the (next) sequences $[s_2, s_3, ...]$ and $[s_2', s_3', ...]$ leads again to
the same solution.

Obviously, if we take $[2, 1, 0, 0 ...]$ and $[2, 0, 0, 0 ...]$ then the utility function will return the same result: **2**. While in the (next) sequences $[1, 0, 0 ...]$ and $[0, 0, 0 ...]$ the utility function won't return the same value so this utility function does not result in stationary preferences between state sequences.

We can, nonetheless, still define $U^{\pi}(s)$ as the expected maximum reward obtained by using the policy $\pi$ starting in state $s$.

<div class="blue-color-box">
<b>17.4</b> Sometimes MDPs are formulated with a reward function $R(s, a)$ that depends on the
action taken or with a reward function $R(s, a, s')$ that also depends on the outcome state.

<ul>
<li><b>a</b>. Write the Bellman equations for these formulations</li>

<li><b>b</b>. Show how an MDP with reward function $R(s, a, s')$ can be transformed into a different
MDP with reward function $R(s, a)$, such that optimal policies in the new MDP correspond
exactly to optimal policies in the original MDP.</li>

<li><b>c</b>. Now do the same to convert MDPs with $R(s, a)$ into MDPs with $R(s)$.</li>
</ul>
</div>

**Answer** 

+ **a**. In the book, the Bellman equation is written as:

$$U(s) = R(s) + \gamma \max\limits_{a \in A(s)} \sum\limits_{s'}P(s'|s,a)U(s')\tag{17.4.1}$$

In this question we are asking to compute the Bellman equation using $R(s,a)$ and $R(s,a,s')$.
If the reward depends on the action then, as we want to maximize the utility (see the $\max$ in the
equation), we need to maximize our action too, so we can rewrite the Utility function as:

$$U(s) = \max\limits_{a \in A(s)}\left[R(s,a) + \gamma\sum\limits_{s'}P(s'|s,a)U(s')\right]$$

We are then asked to rewrite it using $R(s, a, s')$. This time the action depends on the previous state and
on the resultant state, so we need to put this term in both the max over a and the sum over s':

$$U(s) = \max\limits_{a \in A(s)}\sum\limits_{s'}P(s'|s,a)[R(s,a, s') + \gamma U(s')]$$

+ **b**. The idea here is to define for every s, a, s' a pre-state such that $T'(s, a, pre(s,a,s'))$, i.e executing the action $a$ in state $s$ leads to the pre-state $pre(s, a, s')$ from which there is only one action that always leads to s'. Hence we can rewrite U(s) as follow:

$$
U(s) = \max\limits_{a}\left[R'(s,a) + \gamma ' \sum\limits_{s'}T(s, a, s')U(s')\right] \\
= \max\limits_{a}\left[R'(s,a) + \gamma ' \sum\limits_{s'}T(s, a, pre)(\max\limits_{b}[R'(pre,b) + \gamma ' \sum\limits_{s'}T(pre, b, s')U(s')]\right] \\
= \max\limits_{a}\sum\limits_{s'}P(s'|s,a)[R(s,a, s') + \gamma U(s')]
$$

The second equality comes from the idea of the pre-state and the fact that we expand U(s) by one recursive call
so we can write the pre-state and the state s'. The third equality is how we would like to rewrite the utility function. So, by analyzing the second and the last relations, we can see that the equality is satisfied if we define:

+ $R'(s,a) = 0$
+ $T'(pre, b, s') = 1$
+ $R'(pre, b) = \gamma^{-1/2}R(s,a,s')$
+ $\gamma ' = \gamma^{1/2}$
+ $T'(s, a, pre) = T(s, a, s')$

+ **c**. It's the same principle. I won't detail the computation.


<div class="blue-color-box">
<b>17.6</b> Equation (17.7) on page 654 states that the Bellman operator is a contraction.

<ul>
<li> <b>a</b>.  Show that, for any functions f and g,
$$| \max\limits_{a} f(a) − \max\limits_{a} g(a)| ≤ \max\limits_{a} |f(a) − g(a)|$$
</li>
<li><b>b</b>. Write out an expression for $|(B Ui − B Ui')(s)|$ and then apply the result from (a) to
complete the proof that the Bellman operator is a contraction.
</li>
</ul>
</div>

**Answer** 
**a**. Without loss of generality we can assum that $\max\limits_{a} f(a) \geq \max\limits_{a} g(a)$. We can then rewrite:

$$ | \max\limits_{a} f(a) − \max\limits_{a} g(a)| = \max\limits_{a} f(a) − \max\limits_{a} g(a)$$

Now, let's say that the maximum of f is obtained in $a_1$, or put it differently:
$a_1 = \arg\max\limits_{a} f(a)$. We also define $a_2$ as being: $a_2 = \arg\max\limits_{a} g(a)$. With
these definition we can then write:

$$ | \max\limits_{a} f(a) − \max\limits_{a} g(a)| = \max\limits_{a} f(a) − \max\limits_{a} g(a) \\
= f(a_1) - f(a_2) \leq f(a_1) - g(a_1)$$

because $\forall a \in A, g(a_2) \geq g(a)$, so in particular for $a_1$, $g(a_1) \leq g(a_2)$. So final
we have:

$$ | \max\limits_{a} f(a) − \max\limits_{a} g(a)| \leq f(a_1) - g(a_1) \\
= | f(a_1) - g(a_1) | \leq \max\limits_{a}|f(a) - g(a)|
$$

**b**. This question isn't difficult, we just need to carefully notice that what we want to prove is that:

$$||BUi - BUi'|| \leq \gamma ||Ui - Ui'||$$

and the question asked us to compute $|(B Ui − B Ui')(s)|$ first! So let's compute this quantity. We have:

$$
|(B U_i − B U_i')(s)| = |R(s) + \gamma \max\limits_{a}\sum\limits_{s'}T(s, a, s')U_i(s') -
R(s) + \gamma \max\limits_{a}\sum\limits_{s'}T(s, a, s')U_i'(s')| \\
= \gamma |\max\limits_{a}\sum\limits_{s'}T(s,a,s')U_i(s') - \max\limits_{a}\sum\limits_{s'}T(s,a,s')U_i'(s)| \\
\leq \gamma\max_{a}|\sum\limits_{s'}T(s,a,s')U_i(s') - \sum\limits_{s'}T(s,a,s')U_i'(s)|
$$

+ The first equality comes from the definition of the Bellman operator.
+ The second equality comes from the fact that $\gamma$ is positive (between 0 and 1).
+ The first inequality comes from the **a**.

If we let :

$$a^{*} = \arg \max \limits_{a} (\sum\limits_{s'}T(s,a,s')U_i(s') - \sum\limits_{s'}T(s,a,s')U_i'(s))$$

we can then write (without the max operator):

$$
|(B U_i − B U_i')(s)| \leq \gamma|\sum\limits_{s'}T(s,a^{*},s')U_i(s') - \sum\limits_{s'}T(s,a^{*},s')U_i'(s)| \\
= \gamma|\sum\limits_{s'}T(s,a^{*},s')(U_i(s') - U_i'(s))|
$$

Finally, we can now compute the **max norm** of $ BU_i - B U_i'$ to prove that the Bellman operator is a contraction:

$$
||B U_i − B U_i'|| = \max_{s}|(B U_i − B U_i')(s)| \\
\leq \gamma \max_{s} |\sum\limits_{s'}T(s,a^{*},s')(U_i(s') - U_i'(s))|
\leq \gamma \max_{s} |U_i(s) − U_i'(s)| = \gamma ||U_i - Ui'||
$$

where the last **inequality** comes from the fact that $T(s, a, s')$ are probabilities and so we have a convex inequality.

<div class="blue-color-box">
<b>17.7</b> This exercise considers two-player MDPs that correspond to zero-sum, turn-taking
games like those in Chapter 5. Let the players be $A$ and $B$, and let $R(s)$ be the reward for
player $A$ in state $s$. (The reward for $B$ is always equal and opposite.)

<ul>
<li> <b>a</b>.  Let $U_A(s)$ be the utility of state $s$ when it is $A$’s turn to move in s, and let $U_B(s)$ be the utility of state $s$ when it is $B$’s turn to move in s. All rewards and utilities are calculated
from $A$’s point of view (just as in a minimax game tree). Write down Bellman equations
defining $U_A$(s) and $U_B$(s).
</li>
<li><b>b</b>. Explain how to do two-player value iteration with these equations, and define a suitable
termination criterion.
</li>
<li><b>c</b>. Consider the game described in Figure 17.7.1. Draw the state space (rather
than the game tree), showing the moves by $A$ as solid lines and moves by $B$ as dashed
lines. Mark each state with $R(s)$. You will find it helpful to arrange the states $(sA, sB)$
on a two-dimensional grid, using $sA$ and $sB$ as “coordinates.”
</li>
<li><b>d</b>. Now apply two-player value iteration to solve this game, and derive the optimal policy.
</li>
</ul>
</div>
<div class="centered-img">
<img src="../images/rl_exercices/figure-17-7.png" width="250px" alt="Environment" />
<div class="legend">Figure 17.7.1: The starting position of a simple game. Player A moves first. The two players take turns moving, and each player must move his token to an open adjacent space in either direction. If the opponent occupies an adjacent space, then a player may jump over the opponent to the next open space if an. (For example, if A is on 3 and B is on 2, then A may move back to 1.) The game ends when one player reaches the opposite end of the board. If player A reaches space 4 first, then the value of the game to A is +1; if player B reaches space 1 first, then the value of the game to A is -1.</div>
</div>


**Answer**
+ **a**. When is $A$'s turn to move, $A$ reach a new state $s'$ from s and, in this new state $s'$ it's $B$'s
turn to move. The utility function is written as:

$$U_A(s) = R(s) + \max_{a}\sum\limits_{s'}T(s,a,s')U_B(s')$$

As we want the utility $U_B$ from $A$'s point of view, $A$ will likely take into consideration that $B$ will
want to **minimize** its utility. So we have:

$$U_B(s) = R(s) + \min_{a}\sum\limits_{s'}T(s,a,s')U_A(s')$$

+ **b**. To do two-player value iteration we simply apply the Bellman update for the two-player alternatively.
The process terminates when 2 successive utilities (for the same player) are equal or within a certain (fixed) epsilon.

+ **c**, **d**
To solve these questions we need to iteratively apply the value iteration algorithm starting from the four final states: (4,3), (4,2) and (2,1), (3,1). Note that the state (4,1) is not a final state as, if A reaches 4
then the game ends and B can not reach 1 (and vice-versa). I won't try to solve this by hand. Yet, I might update this exercise in the future to sketch out the first few steps to take.

<div class="blue-color-box">
<b>17.8</b> Consider the 3 × 3 world shown in Figure 17.8.1 below. The transition model is the same
as in the 4 × 3 Figure 17.1.1: 80% of the time the agent goes in the direction it selects; the rest
of the time it moves at right angles to the intended direction. <br>
Implement value iteration for this world for each value of $r$ below. Use discounted
rewards with a discount factor of 0.99. Show the policy obtained in each case. Explain
intuitively why the value of $r$ leads to each policy
<ul>
<li> <b>a</b>. $r = 100$
</li>
<li><b>b</b>. $r = -3$
</li>
<li><b>c</b>. $r = 0$
</li>
<li><b>d</b>. $r= +3$
</li>
</ul>
</div>
<div class="centered-img">
<img src="../images/rl_exercices/figure-17-8.png" width="200px" alt="Environment" />
<div class="legend">Figure 17.8.1: The reward for each state is indicated. The upper right square is a terminal state.</div>
</div>

**Answer** It would be too cumbersome to run the value iteration algorithm on all the 4 cases by hand. And the question asked to implement value iteration, so I think we should write a program to solve this question. I won't do it. Yet we can try to figure out the policy obtained in each case. I draw a figure with all the
different policies for all the different cases _a_, _b_, _c_, _d_:

<div class="centered-img">
<img src="../images/rl_exercices/17.8-policies.png" width="300px" alt="Policies" />
<div class="legend">Figure 17.8.2: policy for each value of r. The red square are the square were the reward is equal to <b>r</b>. The white squares have reward equal to <b>-1</b>, the gray square is the final square with reward <b>+10</b></div>
</div>

**a**. If the reward in the red square is 100, the agent will likely want to stay in this square forever and
hence avoid to go to the final state (in gray). As we are dealing with a stochastic environment (we go in the
direction we want with probability 0.8 and in the perpendicular directions with probability 0.1), the arrow around the final gray state need to point in the opposite direction to avoid going into the final state.

**b**. If the reward in the red square is -3, then, as the reward of the white squares are -1 and the reward in the final square is +10, the agent we likely want to avoid the red square and go as fast as possible to the
gray square. However, we don't have a down arrow in (1,2) because if we were to put a down arrow in (1,2) the agent will likely make a detour that can can cost more than -3 points.

**c**. Here the reward for the red square is 0, so, as the rewards in the white squares are -1, the agent will want to go through the red square before reaching the final gray square. It won't want to stay in the red square as the final square offer a +10 reward. That explains the sense of the arrows.

**d**. Here r = 3, so the agent will want to stay in the red square indefinitely (same explanations as in **a**).

<div class="blue-color-box">
<b>17.9</b> Consider the 101 × 3 world shown in Figure 17.9.1 below. In the start state the agent has
a choice of two deterministic actions, Up or Down, but in the other states the agent has one
deterministic action, Right. Assuming a discounted reward function, for what values of the
discount $\gamma$ should the agent choose Up and for which Down? Compute the utility of each
action as a function of $\gamma$. (Note that this simple example actually reflects many real-world
situations in which one must weigh the value of an immediate action versus the potential
continual long-term consequences, such as choosing to dump pollutants into a lake.)
</div>
<div class="centered-img">
<img src="../images/rl_exercices/figure-17-9.png" width="600px" alt="Environment" />
<div class="legend">Figure 17.9.1: 101 x 3 world for Exercise 17.9 (omitting 93 identical columns in the middle). The start state has reward 0.</div>
</div>


**Answer**: This exercise is quite straightforward. We need to apply the Bellman equation in the two different
situations. Let's assume first that we want to go **UP**. We have:

$$
U(s) = 0 + \gamma (\max_{a}\sum\limits_{s_{13}} P(s_{13}|s_{12},a) U(s_{13})) \\
$$

If the agent goes **UP** we can only reach the state $s_{13}$ with probability 1, so 

$$\sum\limits_{s_{13}} P(s_{13}|s_{12},a) = 1 \times U(s_{13})$$

And finally we have:

$$
U_{up}(s) = \gamma U(s_{13}) = \gamma(50 + \gamma \sum\limits_{s_{23}} P(s_{23} | s_{12},a) U(s_{23})) \\
= \gamma (50 + \gamma U(s_{23})) = 50 \gamma + \gamma^{2} U(s_{23}) \\
= 50 \gamma + \gamma^{2} (-1 + \gamma U(s_{33})) \\
= 50 \gamma - \gamma^{2} + \gamma^{3} (-1 + \gamma U(s_{43})) \\
= 50 \gamma - \gamma^{2} - \gamma^{3} + \gamma^{4} U(s_{43}) \\
= \text{...} \\
= 50 \gamma - \sum\limits_{i = 2}^{101} \gamma^{i} \\
= 50 \gamma - \gamma^{2} \sum\limits_{i=0}^{99} \gamma^{i} \\
= 50 \gamma - \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma}
$$

The last relation comes from the fact that $\gamma \in [0,1]$.

We use the Bellman equation to compute the utility if the agent goes *DOWN*. We obtain:
$$
U_{down}(s) = -50 \gamma + \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma}
$$

We then need to solve the system (with a computer):

$$
50 \gamma - \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma} = -50 \gamma + \gamma^{2} \frac{(1-\gamma^{100})}{1-\gamma}
$$

to find the value of $\gamma$.

<div class="blue-color-box">
<b>17.10</b> Consider an undiscounted MDP having three states, (1, 2, 3), with rewards −1, −2,
0, respectively. State 3 is a terminal state. In states 1 and 2 there are two possible actions: $a$
and $b$. The transition model is as follows:

<ul>
<li>In state 1, action $a$ moves the agent to state 2 with probability 0.8 and makes the agent
stay put with probability 0.2.</li>
<li>In state 2, action $a$ moves the agent to state 1 with probability 0.8 and makes the agent
stay put with probability 0.2.</li>
<li>In either state 1 or state 2, action $b$ moves the agent to state 3 with probability 0.1 and
makes the agent stay put with probability 0.9</li>
</ul>

Answer the following questions:

<ul>
<li><b>a</b>. What can be determined <i>qualitatively</i> about the optimal policy in states 1 and 2?.</li>
<li><b>b</b>. Apply policy iteration, showing each step in full, to determine the optimal policy and
the values of states 1 and 2. Assume that the initial policy has action $b$ in both states.</li>
<li><b>c</b>. What happens to policy iteration if the initial policy has action $a$ in both states? Does
discounting help? Does the optimal policy depend on the discount factor?</li>
</ul>
</div>

**Answer** 
**a**. If the agent is in state 1 it should do action **b** to reach the terminal state (state 3) with reward 0. If the agent is in state 2, it might prefer to do action $a$ in order to reach state $1$ and then action $b$ from state 1 to reach the terminal state. Indeed, if the agent do action $b$ in state 2, he has 0.1 chance to end in state 0 and 0.9 chance to stay in state 2 with reward -2, while, if he is in state 1 and fails to go to state 0 it will cost the agent -1 at each attempt. So there is a trade-off to compute.

**b**. We apply policy iteration to find out what is the best policy in each state.
**Initialization**:
+ $U = (u_1, u_2, u_3) = (-1, -2, 0)$
+ $p = (b, b)$ (initialize policy to b and b for each state 1 and 2)
+ $\gamma = 1$

**Computation**
+ $u_1 = -1 + \gamma \sum\limits_{s'} P(s'|s,\pi{s}) u_1(s') = -1 + 0.1 u_3 + 0.9 u_1$
+ $u_2 = -2 + 0.1 u_3 + 0.9 u_2$
+ $u_3 = 0$

So $u_1 = -10$, $u_2 = -20$

**policy update**:
+ _state 1_:
+ _action a_: $\sum\limits_{i} T(1,a,i) u_i = 0.8 \times -20 + 0.2 \times -10 = -18$ 
+ _action b_: $\sum\limits_{i} T(1,b,i) u_i = 0.1 \times 0 + 0.9 \times -10 = -9$
$-9 \geq -18$ so $\pi = b$ in **state 1**

+ _state 2_:<br><br>
+ _action a_: $\sum\limits_{i} T(1,a,i) u_i = 0.8 \times -10 + 0.2 \times -20 = -12$ 
+ _action b_: $\sum\limits_{i} T(1,b,i) u_i = 0.1 \times 0 + 0.9 \times -20 = -18$
$-12 \geq -18$ so $\pi = a$ in **state 2**

As the action **has changed** for state 2, we need to continue the policy iteration algorithm.


+ $u_1 = -1 + 0.1 u_3 + 0.9 u_1$
+ $u_2 = -2 + 0.8 u_1 + 0.2 u_2$
+ $u_3 = 0$
+ so $u_1 = -10$ and $u_2 = -15$

**policy update**:
+ _state 1_:<br><br>
+ _action a_: -14
+ _action b_: -9
$-9 \geq -14$ so $\pi = b$ in **state 1** (hasn't changed!)

+ _state 2_:<br><br>
+ _action a_: -11
+ _action b_: -13.5
$-11 \geq -13.5$ so $\pi = a$ in **state 2** (hasn't changed!)

As the action for both state hasn't changed, we stop the policy iteration here.<br>

So finally, if we are in state 1 we will choose action $b$ and if we are in state 2 we will choose action $a$.
This result match the analysis from part $a)$.

**c**. If the initial policy has action $a$ in both states then the problem is unsolvable. We only need to
write the initialization to see that the equations are inconsistent:
+ u_1 = -1 + 0.2 u_1 + 0.8 u_2
+ u_2 = -2 + 0.8 u_1 + 0.2 u_2
+ u_3 = 0
<br>
discounting may help actually because it allows us to bound the penalty. Indeed, intuitively if $\gamma$ is near 0 then the cost incurs in the distant future plays are negligible. Hence, the action of the agent depends on the choice of the value of $\gamma$.


<div class="blue-color-box">
<b>17.13</b> Let the initial belief state b0 for the 4 × 3 POMDP of Figure 17.1.1 be the uniform distribution
over the nonterminal states, i.e., $(\frac{1}{9},\frac{1}{9},\frac{1}{9},\frac{1}{9},\frac{1}{9},\frac{1}{9},\frac{1}{9},\frac{1}{9},\frac{1}{9},0,0)$. Calculate the exact
belief state $b_1$ after the agent moves <i>Left</i> and its sensor reports 1 adjacent wall. Also calculate
$b_2$ assuming that the same thing happens again
</div>

**Answer** I think the question is a bit unclear, even if we have the book. So we will make the assumption that the sensor measures the number of adjacent walls, which
happen to be 2 in all the squares beside the square of the third column. Moreover, we will assume that this is a noisy sensor and that it will return the True value (
i.e the exact number of adjacent walls) with probability 0.9 and a wrong value with probability 0.1. <br>
As we are dealing with a POMDP and we want the belief next belief state according to the previous belief states, we will use the formula:

$$
b'(s') = \alpha P(o|s') \sum\limits_{s} P(s'|s,a)b(s)
$$

For example, if we are end in square $(1,1)$ (equivalently in state $s_{11}$), that would have meant that we were either in state $(1,1)$, $(1,2)$ or $(2,1)$ before. So, using
the formula with these notations we have:

$$
b_1(s_{11}) = \alpha P(o|s_{11}) \sum\limits_{s} P(s_{11}|s,a)b(s) \\
= \alpha[P(s_{11}|s_{12},a) \times \frac{1}{9} + P(s_{11}|s_{11},a) \times \frac{1}{9} + P(s_{11}|s_{21},a) \times \frac{1}{9}] \\
= \alpha 0.1 \times \frac{1}{9} \times [0.1 + 0.9 + 0.8] \\
= 0.02 \alpha
$$

For example for state $s_{12}$ (square $(1,2)$), we will have:

$$
b_1(s_{12}) = \alpha P(o|s_{12}) \sum\limits_{s} P(s_{12}|s,a)b(s) \\
= \alpha[P(s_{12}|s_{12},a) \times \frac{1}{9} + P(s_{12}|s_{11},a) \times \frac{1}{9} + P(s_{12}|s_{13},a) \times \frac{1}{9}] \\
= \alpha 0.1 \times \frac{1}{9} \times [0.8 + 0.1 + 0.1] \\
= \frac{0.1}{9} \alpha
$$

where $\alpha$ is a constant such that $\forall i \in S, \sum\limits_{s}b_i(s) = 1$. Hence to find $\alpha$ we need to solve:

$$
b_1(s_{11}) + b_1(s_{12}) + b_1(s_{21}) + ... + b_1{s_{43}} = 1 
$$

i.e

$$
\alpha * [0.02 + \frac{0.1}{9} + ...] = 1
$$

So, I won't do all the computation here. What is important is how we can solve the problem and not the computation in itself.

<div class="blue-color-box">
<b>17.14</b> What is the time complexity of $d$ steps of POMDP value iteration for a sensorless
environment?
</div>

**Answer** In a sensor environment the time complexity is $O(|A|^d.|E|^q)$ where $|A|$ is the number of actions and $|E|$ is
the number of observation and $d$ is the depth search. In a sensorless environment we don't have to build branches for the
observations so we would simply have a time complexity of $O(|A|^d)$

<div class="blue-color-box">
<b>17.15</b> Show that a dominant strategy equilibrium is a Nash equilibrium, but not vice versa.
</div>

To do that we need to write down the mathematical definition of a _dominant strategy equilibrium_ and a _Nash equilibrium_.<br>
A strategy _s_ for a player _p_ dominates strategy _s'_  if the outcome for s is better for _p_ than the outcome for _s'_,
for every choice of strategies by the other player(s).

So, mathematically we can say that a _dominant strategy equilibrium_ can be written:

$$
\exists s_j \in [s_1, s_2, ..., s_n] / \forall p \in Player, \forall s_i' \in [s_1', s_2', ... s_n'], \forall str \in \text{[other strategies of the opponents]}, outcome(s_j,str) > outcome(s_j', str)\tag{1}
$$

where $str$ is a strategy among all the possible combination of strategies of all the opponents.
While, in a Nash equilibrium, we only require that the strategy $s_i$ is optimal for the current combination of the opponents' strategies:

$$
\exists s_j \in [s_1, s_2, ..., s_n] / \forall p \in Player, \forall s_i' \in [s_1', s_2', ... s_n'], outcome(s_j,s_{-j}) > outcome(s_j', s_{-j})\tag{2}
$$

So (1) => (2).

Yet, (2) does not neccessarily imply (1) as it is depicted by the BluRay/DVD example of the book.

<div class="blue-color-box">
<b>17.17</b> In the children’s game of rock–paper–scissors each player reveals at the same time
a choice of rock, paper, or scissors. Paper wraps rock, rock blunts scissors, and scissors cut
paper. In the extended version rock–paper–scissors–fire–water, fire beats rock, paper, and
scissors; rock, paper, and scissors beat water; and water beats fire. Write out the payoff
matrix and find a mixed-strategy solution to this game.
</div>

**Answer** We will create a table for this game with reward +1 for the winner and -1 for the loser (reward 0 for a draw).
The table is straightforward and it is antisymmetric:
<div class="centered-img">
<img src="../images/rl_exercices/figure-17-13.png" width="400px" alt="Table for rock paper scissors fire water" />
<div class="legend">Figure 17.17.1: table for rock-paper-scissors-fire-water.</div>
</div>

To find the mixed-strategy solution to this game we will need to find the probability of playing Rock (r), Paper (p), Scissors (s), Fire (f), Water (w),
such that $r + p + s + f + w = 1$. To do so we will firstly focus on what we need to compute if Player _A_ plays Rock. To answer this question we can draw
a graph like this:

<div class="centered-img">
<img src="../images/rl_exercices/figure-17-17.png" width="400px" alt="Outcome if A plays rock" />
<div class="legend">Figure 17.17.2: B's actions with their outcome if A plays Rock.</div>
</div>

According to Figure 17.17.2, if _A_ chooses Rock then the payoff is : $+1 \times p + (-1) \times s + 1 \times f + (-1) \times w + 0 \times r$
We do that for all choices of _A_ and we end with a system of equation:

$$
\text{A chooses R:} +p -s +f -w \\
\text{A chooses P:} -r +s +f -w \\
\text{A chooses S:} +r -p +f -w \\
\text{A chooses F:} -r -p -s +w \\
\text{A chooses W:} +r +p +s -f \\
$$

We then need to solve for the intersection of the hyperplanes. We find that $r=p=s=\frac{1}{9}$ and $f=w=\frac{1}{3}$.


<br><br>
