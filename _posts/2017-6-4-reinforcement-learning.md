---
layout: post
update: <b>Note</b>&#58; I will update this article in the coming days (grammar, add images, add other exercise)
title: Reinforcement Learning Exercise
---

In this article I present some solutions to reinforcement learning exercise. These exercise are
taken from the book "Artificial Intelligence A Modern Approach 3rd edition". I will gradually
update this post with new solutions while I'm learning about the field. I don't pretend all my
solutions are 100 % accurate but I think they make sense. If you do find some mistakes, please do
let me know.

<div class="blue-color-box">
<b>15.1</b> For the 4 × 3 world shown in Figure 17.1, calculate which squares can be reached
from (1,1) by the action sequence [Up, Up, Right, Right, Right] and with what probabilities.
Explain how this computation is related to the prediction task (see Section 15.2.1) for a hidden
Markov model.
</div>

**Answer** Well, this exercise doesn't seem difficult but we surely need lot's of time to
compute all the solutions with their probabilities. So, first of all, we are asking to compute the probabilities
of being in each square at the end of the sequence *[Up, Up, Right, Right, Right]*. So we don't
need to use the Ballman equation here, as we only want a **probability** and not the **utility**.
Then, obviously, we can do somehting a little bit smarter than just enumerate all the solutions. Indeed,
we can compute the probability of each square we reach doing only a upward movement and then reusing these
probabilities (associated with their positions) to compute the new probabilities of being in the next position
when we are doing another movement of the sequence, and so now. For the 2 first Upward movements we have:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:400px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.1-table#R3VfLcpswFP0aZtpFMkJyaLOMSdpuummm07Uw4jERiAoRk359r5CEedhTT%2By4drxBnPu%2BukI%2BHgmL9qukVfZdxIx7GMWtR%2B49jP0FDuChkReDBLcWSGUeW6UN8Jj%2FYRZEFm3ymNUjRSUEV3k1BleiLNlKjTAqpViP1RLBx1ErmrIZ8LiifI7%2BymOVGZTcoA3%2BjeVpZiMvkBVEdPWUStGUNpyHSdL9jLigzpXVrzMai%2FUAIg8eCaUQyqyKNmRct9Z1zdh92SHt05asVPsYEGPwTHnDXMYBB9OlohE0AyoSMmaykwW%2FG53X0t8sQWHd96eX627cjJRq9WJb60DFWnVFeZ6WHrnTOwkZQxzId%2BhVS5w7K8ls37eITK5XK8E5rWpmHbu3QT5Bap%2Bm1EjAnE5BOUNiPVwm40EhLvFd3qH52vIovqIBQIsKFmVUV6b4n5Xuwwzvk4guP7EdMX90A%2FH%2BHGpwPoXHmss%2Bhw9wnkP%2F41uPCrr2j%2BsO4X%2F6m1Q8a92ZGJ5yo%2FEJNvrzcd3hhTOtm2rqLRFw03VfeTG%2BpbqrFyF9Q0xdDgfR2M%2B6sCXSmQzLKaZs1wDh138p9v8I9Of79swa%2BD87r48uOaTzr64VXQeLt45xQRtBDj0CB2wE2vfwXOZGaHDbX3GNGzbicDyiFFhzCcAzVXAAfFjWSoonFtpLoRQlaC6TnPMJJJ6ZTHjHwLQUkJjWGYutG8uOmIQIOxmW3%2FM2oMNMFExJKAG1jhoaC0uENVvBI9aE8Serkg0YZU81qWWyae95Q%2BdgYRmde90wx042YOfk4S8%3D"></iframe>
<div class="legend">Figure 17.1.1: occupancy probabilities at each time step for the first 2 movements.</div>
</div>

So for example, to compute 0.24<sup>1</sup> in position **(1,2)**, we reused the probability of the first column and we multiply by the probability given in the exercise. It gives us:

$$
P((1,2) \text{ in step 2}) = P((1,1) \text{ in step 1}) \times P((1,2)|(1,1) \text{ in step 1}) + P((1,2) \text{ in step 1}) \times P((1,2)|(1,2) \text{ in step 1}) + P((2,1) \text{ in step 1}) \times P((1,2)|(2,1) \text{ in step 1}) \\
= 0.1 \times 0.8 + 0.8 \times (0.1 + 0.1) + 0.1 \times 0 = 0.24
$$

Where $(0.1 + 0.1)$ in $0.8 \times (0.1 + 0.1)$ comes from the fact we can stay in our position if, while we are willing to go Up, the agent try to go left (with probability 0.1) or
right (with probability 0.1). as there are walls in both directions, the agent will likely stay is its position.

So finally we can come up with the full table, with the asked occupancy probabilities in last column:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:520px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.1-table#R5Vhdk5owFP01zLQP24GED32sdrt96Ut3On0OEoHZQGiIu25%2FfRMJX6JIBHVsfTGee3Nzc%2B7l4hwDLpPtE0NZ9J0GmBjADLYG%2FGIAYNnAFV8SeS8Qd66AkMWBcqqB5%2FgPVqCp0E0c4LzlyCklPM7a4IqmKV7xFoYYo29ttzUl7VMzFOIO8LxCpIv%2BigMeqVuYZo1%2Fw3EYqZMdoAw%2BWr2EjG5SdZwB4Hr3KcwJKkMp%2FzxCAX1rQPDRgEtGKS9WyXaJiaS2ZK3Y9%2FWItUqb4ZQP2QCLDa%2BIbHCZsUvE1gVHviBD3IiyALOdzf29kXktrHopHN4qfiq7pMlpOeX8XVFbghxv%2BQMicZga8LOspMhYnCPybUaVljKcskSK9wOmIteHFSUEZTlWgctfjXzcUH0XV%2FWp6NN9kHWQQDZXkXHjImXix6IL8uXOSWL5DQAlmVikfp4Vl%2F%2BZSR46eJWEf%2F%2BJHTnzx64h%2Fr2AEux24VR9WeXwQTzPS%2BvjpVvF%2FGRNG84EJ%2BPVrq6G78zWcLbd%2FjSuWURwhSLOpg0HhnMNnNNnV76WN9zZmplOfxoaReyrDzyvPsOfn9LT1ejg2XCiTGjrPEfOieJOwCo4f3Tthewce%2FqCcx3iNHjztIbVdUge1bpnkzyiOo41%2FPVgeUDjIXDB5SmHt%2Bzr4XPG9OClhocLvGuQfOYr83Z9beqUR6s%2Btm33e09E%2BZ2NkhG1ss15fxO3h9B8fvG5Yt9sroxqeZ3p7GhMmZnrTfbvvY%2FyO5sy48aTBv%2BWLQWcy%2FP%2Fv4ycEYWDwDvxv%2FNgLSR4SD6TeKEgljhoyYBA6n8Cj3hCBGCJZc4ZfcFLSqiUG1OaCs%2FFOiZkD6KvmK3JTjWVVoEEKI9woMIoRRMzccJRVdSqtNZk%2B4RpgjkTVzDVBlups0q8hkolrZXOWvCNGiqw7ThKgVbqc1hFriVYsVAqbPmzVnt3toaiDh%2F%2FAg%3D%3D"></iframe>
<div class="legend">Figure 17.1.2: occupancy probabilities at each time step. Answer to the question are the probabilities in the last column</div>
</div>

This computation is related to the prediction task fro a HMM in the sense that we only need to consider the probability of the previous position to compute the
probability of being in the next position.


<div class="blue-color-box">
<b>17.2</b> Select a specific member of the set of policies that are optimal for R(s) > 0 as shown
in Figure 17.2(b), and calculate the fraction of time the agent spends in each state, in the limit,
if the policy is executed forever. (Hint: Construct the state-to-state transition probability
matrix corresponding to the policy and see Exercise 15.2.)
</div>


**Answer** According to Figure 17.2(b) from the book, we can take whatever policy we want for the squares:
(1,2), (2,1), (3,1), (1,2), (1,3), (2,3). For example we can choose to always go **right**. We
must note that again, if we choose to go right the agent will go right with 0.8 probability and it will go
down or up (in the perpendicular directions) with probability 0.1. Having say that our Transtion matrix looks like:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:481px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.1-table#R7VlLc5swEP41zLSHZECymfhY0yS99JRDzwLJwEQgR8gx6a%2BvZEk84njih4xJGl%2BAb7UPLauV8OfBqKjvOVpmvxkm1AM%2Brj340wMgmIBQXhTyopFwZoCU59gMaoGH%2FC8xoG%2FQVY5J1RsoGKMiX%2FbBhJUlSUQPQ5yzdX%2FYgtG%2B1yVKyRbwkCC6jf7Jscg0OvP9Fv9F8jQznqdWEKPkMeVsVRp3HoCLzU%2BLC2RNmfFVhjBbdyB468GIMyb0XVFHhKrU2qxpvbsd0iZsTkqxjwLUCs%2BIroiNOKRSdS5QLJMhZ8Q4JnwjC59WKq550N7KAesmP41cZWPaG1SJF5NaCwpSiytE87T04A%2F1JmXE0o%2BMt2tVSaw5I8lM3t8Q6VivEkYpWlbEGLZPnXjC1Fz1VGMm6%2FQ1yLcQrIpLR9yZiA18l3WZfKXpxFZsgW%2FyHUTB946PeBi%2F4Hi%2Fu0xC9yYn7k2qhAP3E3dsUkUJ3ZoE7k3C40wqcHtVulqnQ68t%2Fzpwa%2B7m0PexlblzKe4z1UvFNpTiUMV7QoM%2Bxu%2F%2BWTuolsG75sZf9Z9X8YBaPvumf%2FmeNUANfqIG6qB23J3uRrgD%2Btez0c5i9IoOiuvkc%2F5HWKtfHe1sm5qbD7oRNqZLxDbew917eXPUidx8eI%2B1XX%2BM%2Bh3BSdBBNbn7G2f8RfE%2FKLr7Rna08X0V1%2BUb0Aku9u1cR218CnyLZFG45pksDnpkEVAskcQzUVAJBPK2Epw9kohRpkipkpVy5HyRU%2FoKYs%2BEL%2BiGW1NSiWBUZQQbM4b3Ilx62MmdBQ0jV9T3hBVEcDkF3yhYDs9QnJZLa%2FkwcBOaIVmHK5xMp4anNBxl2lhuiTp5Y7g6%2B9hyghtZh3eFt%2F8A"></iframe>
<div class="legend">Figure 17.2.1: Transition Matrix</div>
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

<div class="blue-color-box">
<b>17.3</b> Suppose that we define the utility of a state sequence to be the maximum reward obtained
in any state in the sequence. Show that this utility function does not result in stationary
preferences between state sequences. Is it still possible to define a utility function on states
such that MEU decision making gives optimal behavior?
</div>

**Answer** To understand the problem, let's write it mathematically. The utility function can be written:

$$U(s_0, a_0, s_1, ..., a_n, s_n) = \max\limits_{i=0}^{n-1} R(s_i, a_i, s_{i+1})$$

We say that a Utility function meets the stationary property if the result of applying the Utility function to
the sequences $[s_1, s_2, ...]$ and $[s_1', s_2', ...]$ leads to the same solution **and** the result of
applying the Utility function to the (next) sequences $[s_2, s_3, ...]$ and $[s_2', s_3', ...]$ leads again to
the same solution.

Obviously if we take $[2, 1, 0, 0 ...]$ and $[2, 0, 0, 0 ...]$ then the Utility function will return the same result: **2**. While in the (next) sequences $[1, 0, 0 ...]$ and $[0, 0, 0 ...]$ the Utility function won't return the same value so this Utility function does not result in stationary preferences between state sequences.

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
If the reward depends on the action then as we want to maximize the utility (see the $\max$ in the
equation), we need to maximize our action too, so we can rewrite the Utility function as:

$$U(s) = \max\limits_{a \in A(s)}\left[R(s,a) + \gamma\sum\limits_{s'}P(s'|s,a)U(s')\right]$$

We are then asked to rewrite it using $R(s, a, s')$. This time the action depend of the action and
on the resultant state, so we both need to put this term in the max over a and in the sum over s':

$$U(s) = \max\limits_{a \in A(s)}\sum\limits_{s'}P(s'|s,a)[R(s,a, s') + \gamma U(s')]$$

+ **b**. The idea here is to define for every s, a, s' a pre-state such that $T'(s, a, pre(s,a,s'))$, i.e executing the action $a$ in state $s$ leads to the pre-state $pre(s, a, s')$ from which there is only one action that always leads to s'. Hence we can rewrite U(s) as follow:

$$
U(s) = \max\limits_{a}\left[R'(s,a) + \gamma ' \sum\limits_{s'}T(s, a, s')U(s')\right] \\
= \max\limits_{a}\left[R'(s,a) + \gamma ' \sum\limits_{s'}T(s, a, pre)(\max\limits_{b}[R'(pre,b) + \gamma ' \sum\limits_{s'}T(pre, b, s')U(s')]\right] \\
= \max\limits_{a}\sum\limits_{s'}P(s'|s,a)[R(s,a, s') + \gamma U(s')]
$$

The second equality comes from the idea of the pre-state and the fact that we expand U(s) by one recursive call
so we can write the pre-state and the state s'. The third equality is how we would like to rewrite the Utility function. So, by analyzing the second and the last relationship, we see that the equality is satisfied if we define:

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

**b**. This question isn't difficult, we just need to carefully notices that want we want to prove is that:

$$||BUi - BUi'|| \leq \gamma ||Ui - Ui'||$$

and the question ask us to compute $|(B Ui − B Ui')(s)|$ first! So let's compute this quantity. We have:

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

we can then write (without max operator):

$$
|(B U_i − B U_i')(s)| \leq \gamma|\sum\limits_{s'}T(s,a^{*},s')U_i(s') - \sum\limits_{s'}T(s,a^{*},s')U_i'(s)| \\
= \gamma|\sum\limits_{s'}T(s,a^{*},s')(U_i(s') - U_i'(s))|
$$

Finally, we can now compute the **max norm** of $ BU_i - B U_i'$ to proove that the Bellman operator is a contraction:

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
<li><b>c</b>. Consider the game described in Figure 5.17 on page 197. Draw the state space (rather
than the game tree), showing the moves by $A$ as solid lines and moves by $B$ as dashed
lines. Mark each state with $R(s)$. You will find it helpful to arrange the states $(sA, sB)$
on a two-dimensional grid, using $sA$ and $sB$ as “coordinates.”
</li>
<li><b>d</b>. Now apply two-player value iteration to solve this game, and derive the optimal policy.
</li>
</ul>
</div>

**Answer**
+ **a**. when is $A$'s turn to move, $A$ reach a new state $s'$ from s and in this new state $s'$ is $B$'s
turn to move. The the Utility function is written as:

$$U_A(s) = R(s) + \max_{a}\sum\limits_{s'}T(s,a,s')U_B(s')$$

As we want the utility $U_B$ from $A$'s point of view, $A$ will likely take into consideration that $B$ will
want to **minimize** its Utility. So we have:

$$U_B(s) = R(s) + \min_{a}\sum\limits_{s'}T(s,a,s')U_A(s')$$

+ **b**. To do two-player value iteration we simply apply the Bellman update for the two-player alternatively.
The process terminates when 2 successive utilities (for the same player) are equal or within a certain (fixed) epsilon.

+ **c**, **d**
To solve these questions we need to iteratively apply the value iteration algorithm starting from the four final states: (4,3), (4,2) and (2,1), (3,1). Note that the state (4,1) is not a final state as, if A reaches 4
then the game ends and B can not reach 1 (and vice-versa).

<div class="blue-color-box">
<b>17.8</b> Consider the 3 × 3 world shown in Figure 17.14(a). The transition model is the same
as in the 4 × 3 Figure 17.1: 80% of the time the agent goes in the direction it selects; the rest
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

**Answer** It would be to cumbersome to run the value iteration algorithm by hand on all the 4 cases. So I won't do it. Yet we can try to figure out the policy obtained in each case. I draw a figure with all the
different policy for all the different cases a, b, c, d:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:543px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.8-policies#R7V1Nc6M4EP01PuxhqhACjI9Jdnf2MlVblcOeCSg2Ndh4MUmc%2FfUrDPJHi1k0WiRVWnEqFVuCNrwnPdStlrKgD9vj1ybbb77VBasWYVAcF%2FTXRRiSKEz4n67kvS9JVkPBuimL4aBLwWP5DxsKg6H0pSzY4ebAtq6rttzfFub1bsfy9qYsa5r67faw57q6%2FdZ9tmZSwWOeVXLpX2XRbvrSOA4u5X%2Bwcr1pQcVTln9fN%2FXLbvi6RUifT6%2B%2BepsJU8Pxh01W1G9XRfS3BX1o6rrt322PD6zqoBWo9ef9%2FoPa82U3bNeqnBD2J7xm1QsTV3y6rvZdQPG2KVv2uM%2Fy7vMbp3tB7zfttuKfCH%2BbHfY9Ac%2FlkXGr94e2qb%2BfUaO85Lmsqoe6qpuTQfqc5izPz0de1TylccSR5GfUu%2FZxuIThnl5Z07LjD2%2BTnMHjbZLVW9Y27%2FyQo%2BCnP2NojXRA%2F%2B1CbTIUba5YFWXZ0JjWZ7sXRPmbAdRxgKkLgPsWNwJwcHrxmhkwJcQZqNEIqEnVDg3nBt3k75daVHw5nHTmjh%2FAb%2FR4qeTv1v3f%2B06AelP8Inpros44bXH3M0ZbcnrNRNvSGW0x2r4A9GVlD9MELaZQXyyCusQL6tIZqClaUEHvJ7E9UFdoQYXd3yaqwgvBCOvSIaxkGla2K%2B46%2F41%2FyqvscCjzW1xlGHsTrJAcuklIrm45HrllUdawKmvL11vzYzgM3%2FBnXZ4GoQLxBCAeAigP9UuTs%2BGsa18NGgonDLVZs2atZOhEy%2Fm21ZhS8AvxMwXbvC5R0M6MPCm4l%2Fh5IulcPQoampGpMZ8VPVMhBHi50mQqnTA0I1MKbio%2BptLoBl%2FdHgXMROb6k4Lji54lAvHVpIkE5nhS8KXR8xRCfHV5MjiSUHDP8fEUBhMdQZUoyZA55aNOQn7TczbDVf0f55S6m1KgeGN%2B1F3En46pyudMjRpvEXXHG964ItQYixFw8UzAiKq7eYVIIar4QVGFCmAT1RAtqlABbIbAI7z5IFACrMKqEF37oLBCDbAKq8IwAJ9TGIHwsrb3Dg0ZdN9jhaEFfqa042GSIXMBsdjJE8CK%2By5lWVpMLcP7AJDzLC3COjYX8um%2F62ZaWiQOb14gVJnEIqp4g4SSytiEFW9qoKQBNmHFG8QDGhCF9lBNFEbaHxRVqAFWYcUbxYMaYBVWL9MkwgAgvtSe1gWGKDA0n1%2BYfCZKdLH8eRIlKCR8Rp68TJSAPOl2KGjHXH9aOnlOO0qTsLg0Cu9zWs6TsAhrOALrZ6BFN1HCInGIJ%2FQchgSWeOO5kszYhBXvEmxJBGzCijfaCkXApvO6xBtuhSJgFVa84VYoAlZh9TNZAqzwoisApapnSNMJQ%2FO5hmJxk%2BdMQd9bmylzTnw65hbhZwoAHIk8rJ9Oa4kmDM3IlJcrkaNkpvglNGQwgJkquFb4mKLRTExBQyaZ8nIlssSUbqxZMmTuObXyckRBpzYkUWZqaouUGZnyU%2F2mNrrQZsrcjhkrL6euaZqCrkD0mIrCCUMzMuXl1DV0g2LdQXr433Zm5MnLqWuofbq7OkHpM7eYY%2BVlJAn2J%2B29gmCHMrlXUODlwI%2FAuIKuMwUNGXSmSKCQuYCPKrgxiTZV0JBRqkIfqZJ6la7jKxky5%2FiSwMu4XzJTKJ0E1kLpRPRfv5iCWzzSUHNUATedlAzNSZWXgT84btOe9oAjSYPTHiTwM0ox1RmUqZrqnnNSNRam%2BMkcy3A8x7K38iQKsl%2BuMi6fLocFDf892Q9OnoNaVmbLju1YgxE5FLt6x0DCxVCUVeV617U63moYL7%2FvkivKPKvuhoptWRTd19yP5XsMaM24TisU6RlXrfS8s%2Bt1M4Xr%2BHXyMUgww%2BZjanw%2FTfP9hXpAN0wWs8z3WNjEFd8e9u5YuJZTbMNBrhbbo%2F%2BjwRXbPnZuy3TL3gvBhC7cc4jIyhmNjOfgRKketvIYVh7EfmBsoQ9oF1z5qURRgZs6BFe0U6yqICT2PJMhS645bOXIHi5VANiOPM7MYSs%2FzXCJwi22I1sWmoNWjp2hkoTzwmYnmoB8pCCBa1UU5PgUKlGA4FpVBTkWhFoVosgmuPIQF7Uq0NQiuMJ79kUV6MiSPXPgyjkhqGSBOFQFKvtmqFSBuBQF2TdDJQoQW6uagDzSCKdcrQZsqOydoRIFmHpgF1zZO0OlCnA5q11wZe8MmSzQ27FCmlgEV3bPkMmCS3CRz0BEkUNwR%2F6nFypZAPOSUbq0iK3snaFSBTgvaRdc2T1DpQpwXtIguPxjU3c5JOe6r%2Fw2Nt%2FqgnVH%2FAs%3D"></iframe>
<div class="legend">Figure 17.2.1: policy for each value of r. The red square are the square were the reward is equal to <b>r</b>. The white squares have reward equal <b>-1</b>, the gray square is the final square with reward <b>+10</b></div>
</div>

**a**. If the reward in the red square is 100, the agent will likely want to stay in this square forever and
hence avoid to go to the final state (in gray). As we are dealing with a stochastic environment (we go in the
direction we want with probability 0.8 and in the perpendicular directions with probability 0.1), the arrow around the final gray state need to point in the opposite direction to avoid going into the final state.

**b**. If the reward in the red square is -3, then, as the reward of the white squares are -1 and the reward in the final square is +10, the agent we likely want to avoid the red square and go as fast as possible to the
gray square. However we don't have a down arrow in (1,2) because if we were to put a down arrow in (1,2) the agent will likely make a detour that cost more than -3 points.

**c**. Here the reward for the red square is 0, so, as the rewards in the white squares are -1, the agent will want to go through the red square before reaching the final gray square. It won't want to stay in the red square as the final square offer a +10 reward. That explains the sense of the arrows.

**d**. Here r = 3, so again the agent will want to stay in the red square indefinitely (same explanations as in **a**).

<div class="blue-color-box">
<b>17.9</b> Consider the 101 × 3 world shown in Figure 17.14(b). In the start state the agent has
a choice of two deterministic actions, Up or Down, but in the other states the agent has one
deterministic action, Right. Assuming a discounted reward function, for what values of the
discount $\gamma$ should the agent choose Up and for which Down? Compute the utility of each
action as a function of $\gamma$. (Note that this simple example actually reflects many real-world
situations in which one must weigh the value of an immediate action versus the potential
continual long-term consequences, such as choosing to dump pollutants into a lake.)
</div>

**Answer**: This exercise is quite straightforward. We need to apply the Bellman equation in the two different
situations. Let's assume first that we want to go **UP**. We have:

$$
U(s) = 0 + \gamma (\max_{a}\sum\limits_{s_{13}} P(s_{13}|s_{12},a) U(s_{13})) \\
$$

If we go **UP** we can only reach the state $s_{13}$ with probability 1, so 

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

The last relation comes from the fact that \gamma is within [0,1].

We use the Bellman equation to compute the Utility if the agent go down. We obtain:
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
**a**. If the agent is in state 1 it should do action **b** to reach the terminal state (state 3) with reward 0. If the agent is in state 2, it might prefer to do action $a$ in order to reach state $1$ and then action $b$ from state 1 to reach the terminal state. Indeed, if the agent do action $b$ in state 2, he has 0.1 chance to end in state 0 and 0.9 chance to stay in state 2 with reward -2, while, if he is in state 1 and fails to go to stay 0 it will cost the agent -1 at each attempt. So there is a trade-off to compute!

**b**. We apply policy to find out what is the best policy in each state.
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

As the action **has changed** for state 2, we need to continue the policy iteration algorithm


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

As the action for both state hasn't changed, we stop the policy iteration here.
So finally, if we are in state 1 we will choose action $b$ and if we are in state 2 we will choose action $a$.
This result match the analysis from part $a)$.

**c**. If the initial policy has action $a$ in both states then the problem is unsolvable. We only need to
write the initialization to see that the equations are inconsistent:
+ u_1 = -1 + 0.2 u_1 + 0.8 u_2
+ u_2 = -2 + 0.8 u_1 + 0.2 u_2
+ u_3 = 0
<br>
discounting may help actually because it allows us to bound the penalty. Indeed, intuitively if $\gamma$ is
near 0 than the cost incurs in the distant future plays are negligeble. Hence, the action of the agent depend on the choice of the value of $\gamma$


<br><br>
