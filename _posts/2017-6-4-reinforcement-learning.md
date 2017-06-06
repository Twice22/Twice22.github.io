---
layout: post
update: <b>Note</b>&#58; I will update this article in the coming days with more exercises
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
<iframe frameborder="0" style="width:100%;height:343px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=Figure-17-1#R7Ztdb5swFIZ%2FTaTtohXGQNPLJOu2m0nTUmnXDjiA6uAMnCbdr58Bmw%2FjLigCUkq5CT42NrzP4eCDyQyudqdvMdoHP6iHycw0vNMMfpmZJjBtwH9Sy0tucYCVG%2Fw49ESj0rAO%2F2JhNIT1EHo4qTVklBIW7utGl0YRdlnNhuKYHuvNtpTUR90jHzcMaxeRpvV36LFAXIVhlPbvOPQDMTK0RcUGuU9%2BTA%2BRGG5mwm225dU7JLsS7ZMAefRYMcGHGVzFlLJ8b3daYZJKK1XLj%2Fv6Sm1x2jGOWJsDzPyAZ0QOWJ5xdl7sRUpxDEKG13vkpuUjxz2Dy4DtCC8BvpuwmD4VIkFu2YaErCihMS9HNOKHLcUoOGb49OqZguL6uVthusMsfuFNTlL7%2FAjhUFAUjyWduTAFFTDShoQ%2F%2BEW%2FpSh8R%2Bii1wiORiNgXU0kazQiFffe8CLZoxEJGlcTyRmNSEpIAmA4ke56Fok%2Ft%2B6zrWhZqXGyrZ%2BQNaSI89F4mhqzhlTpfjQqqUFrSJXkUBWZ1o%2BLX49vVSs1dt0PKBUYjUc1otOQMo1nbt4IT0PKNJ7peSM%2BDSmTboLuEJZeLOWnX9XL%2BXOgsuImyXLzBW%2FAL%2FVUVvI9P%2F9dAtkTP4m8M1nV8ZREpNKaKYmRbR1hAnYNk203KNkaSrALSroMoQtKN%2B%2BcEYADQtJlKF1Aas2I68Z0WJSQpYliiIR%2BxIsu1xlz%2BzKlELqILETFLvS8dJilzg86ADevh8CiXOFmabhZXXDTJU1dcDMnwA04VwSnS9S6AAcnAM60rghOlzt2Ac6aADhoXg%2BcHPrjEXdJqFS4ad5C9MZNl1p%2FPOIu4gY077x746bL9T%2BecBdxswfENp5VL8tWvNtoJkx9vXtos%2B6FI2%2BRrtunjkRQkoTuOZ3yLrDXWMg%2FK8mZHFHaYkwQC5%2Fr3et0ECP8pGF2q0rF7xXF7xQpE3qIXSyOKtU835GpdMRQ7GPW6CjDUlx2O1ItFt%2BSAO3T3S3BJ4FseY7e%2F5eWqlQbbw2KtpIaAPmA4qMR%2B9axxugNKsRLvcHpzxlaLDJe2xkMxRcsOzfIXoA5Jd%2Bw1SDfY6RosXj61pzDVp1jWoEDDuccfeX9xu18AhNa604h1Zyq9TWjlT7SA7kppP5QvVnlit8Q6PrK%2FaeBzrbh9dD1lf7LXjbS8Al9rrDcqO3eMV9T%2FaKx5Wu5LlYMoe4bhX74bqbKV31oQs0Ljo748mL5hX4%2BPSr%2FBQEf%2FgE%3D"></iframe>
<div class="legend">Figure 17.1.1: (a) A simple 4 x 3 environment that presents the agent with a sequential
decision problem. (b) Illustration of the transition model of the environment: the "intented" outcome occurs
with probability 0.8, but with probability 0.2 the agent moves at right angles to the intended direction. A collision with a wall results in no movement.<br>
The two terminal states have reward +1 and -1, respectively, and all other states have a reward of -0.04.</div>
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
<iframe frameborder="0" style="width:100%;height:400px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.1-table#R3VfLcpswFP0aZtpFMkJyaLOMSdpuummm07Uw4jERiAoRk359r5CEedhTT%2By4drxBnPu%2BukI%2BHgmL9qukVfZdxIx7GMWtR%2B49jP0FDuChkReDBLcWSGUeW6UN8Jj%2FYRZEFm3ymNUjRSUEV3k1BleiLNlKjTAqpViP1RLBx1ErmrIZ8LiifI7%2BymOVGZTcoA3%2BjeVpZiMvkBVEdPWUStGUNpyHSdL9jLigzpXVrzMai%2FUAIg8eCaUQyqyKNmRct9Z1zdh92SHt05asVPsYEGPwTHnDXMYBB9OlohE0AyoSMmaykwW%2FG53X0t8sQWHd96eX627cjJRq9WJb60DFWnVFeZ6WHrnTOwkZQxzId%2BhVS5w7K8ls37eITK5XK8E5rWpmHbu3QT5Bap%2Bm1EjAnE5BOUNiPVwm40EhLvFd3qH52vIovqIBQIsKFmVUV6b4n5Xuwwzvk4guP7EdMX90A%2FH%2BHGpwPoXHmss%2Bhw9wnkP%2F41uPCrr2j%2BsO4X%2F6m1Q8a92ZGJ5yo%2FEJNvrzcd3hhTOtm2rqLRFw03VfeTG%2BpbqrFyF9Q0xdDgfR2M%2B6sCXSmQzLKaZs1wDh138p9v8I9Of79swa%2BD87r48uOaTzr64VXQeLt45xQRtBDj0CB2wE2vfwXOZGaHDbX3GNGzbicDyiFFhzCcAzVXAAfFjWSoonFtpLoRQlaC6TnPMJJJ6ZTHjHwLQUkJjWGYutG8uOmIQIOxmW3%2FM2oMNMFExJKAG1jhoaC0uENVvBI9aE8Serkg0YZU81qWWyae95Q%2BdgYRmde90wx042YOfk4S8%3D"></iframe>
<div class="legend">Figure 17.1.2: occupancy probabilities at each time step for the first 2 movements.</div>
</div>

So for example, to compute 0.24<sup>1</sup> in position **(1,2)**, we reused the probabilities of the first column and we multiplied by the probability given in Figure 17.1.

$$
P((1,2) \text{ in step 2}) = P((1,1) \text{ in step 1}) \times P((1,2)|(1,1) \text{ in step 1}) + P((1,2) \text{ in step 1}) \times P((1,2)|(1,2) \text{ in step 1}) + P((2,1) \text{ in step 1}) \times P((1,2)|(2,1) \text{ in step 1}) \\
= 0.1 \times 0.8 + 0.8 \times (0.1 + 0.1) + 0.1 \times 0 = 0.24
$$

So finally we can come up with the full table, with the asked occupancy probabilities in the last column:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:520px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.1-table#R5Vhdk5owFP01zLQP24GED32sdrt96Ut3On0OEoHZQGiIu25%2FfRMJX6JIBHVsfTGee3Nzc%2B7l4hwDLpPtE0NZ9J0GmBjADLYG%2FGIAYNnAFV8SeS8Qd66AkMWBcqqB5%2FgPVqCp0E0c4LzlyCklPM7a4IqmKV7xFoYYo29ttzUl7VMzFOIO8LxCpIv%2BigMeqVuYZo1%2Fw3EYqZMdoAw%2BWr2EjG5SdZwB4Hr3KcwJKkMp%2FzxCAX1rQPDRgEtGKS9WyXaJiaS2ZK3Y9%2FWItUqb4ZQP2QCLDa%2BIbHCZsUvE1gVHviBD3IiyALOdzf29kXktrHopHN4qfiq7pMlpOeX8XVFbghxv%2BQMicZga8LOspMhYnCPybUaVljKcskSK9wOmIteHFSUEZTlWgctfjXzcUH0XV%2FWp6NN9kHWQQDZXkXHjImXix6IL8uXOSWL5DQAlmVikfp4Vl%2F%2BZSR46eJWEf%2F%2BJHTnzx64h%2Fr2AEux24VR9WeXwQTzPS%2BvjpVvF%2FGRNG84EJ%2BPVrq6G78zWcLbd%2FjSuWURwhSLOpg0HhnMNnNNnV76WN9zZmplOfxoaReyrDzyvPsOfn9LT1ejg2XCiTGjrPEfOieJOwCo4f3Tthewce%2FqCcx3iNHjztIbVdUge1bpnkzyiOo41%2FPVgeUDjIXDB5SmHt%2Bzr4XPG9OClhocLvGuQfOYr83Z9beqUR6s%2Btm33e09E%2BZ2NkhG1ss15fxO3h9B8fvG5Yt9sroxqeZ3p7GhMmZnrTfbvvY%2FyO5sy48aTBv%2BWLQWcy%2FP%2Fv4ycEYWDwDvxv%2FNgLSR4SD6TeKEgljhoyYBA6n8Cj3hCBGCJZc4ZfcFLSqiUG1OaCs%2FFOiZkD6KvmK3JTjWVVoEEKI9woMIoRRMzccJRVdSqtNZk%2B4RpgjkTVzDVBlups0q8hkolrZXOWvCNGiqw7ThKgVbqc1hFriVYsVAqbPmzVnt3toaiDh%2F%2FAg%3D%3D"></iframe>
<div class="legend">Figure 17.1.3: Occupancy probabilities at each time step. The probabilities in the last column are the answer to the question.</div>
</div>

This computation is related to the prediction task for a HMM in the sense that we only need to consider the probabilities of the previous positions to compute the probabilities of being in the next positions.


<div class="blue-color-box">
<b>17.2</b> Select a specific member of the set of policies that are optimal for $R(s) > 0$ as shown
in Figure 17.2.1, and calculate the fraction of time the agent spends in each state, in the limit,
if the policy is executed forever. (Hint: Construct the state-to-state transition probability
matrix corresponding to the policy and see Exercise 15.2.)
</div>
<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:346px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=Figure-17-2#R7ZxBk5owFMc%2FDTPtYWeEAK7H1W7bS2c63UPPESIyi8RCXLWfvgESEIhu6kAwKx524SUk8P89X%2FJCRgMsNodvCdyuf2AfRYY18Q8G%2BGJYFphO6d%2FMcCwMzswtDEES%2BoXJrAwv4V%2FEjBNm3YU%2BSmsVCcYRCbd1o4fjGHmkZoNJgvf1aisc1XvdwgC1DC8ejNrW36FP1oXVdiaV%2FTsKgzXrGfCCJfRegwTvYtadYYFV%2FimKN5A3xeqna%2Bjj%2FYkJPBtgkWBMiqPNYYGiTFmuWnHd1zOl5W0nKCYyF1jFBW8w2iF%2Bx%2Fl9kSOXYr8OCXrZQi8731PaBpivySaiZyY9TEmCX0uRALWswiha4Agn9DzGMb1sznpBCUGHs3dqls9PvQrhDSLJkVZhFzwyxZhDAXa6r%2BjwGusTMNwGmT8EZbuVKPSA6SLWCGijkekOJpKtjUiWPZhIjjYiAWswkVxtRGqEJNNUJ9K0Z5HouDXLP2XNkxI3%2F%2FQTslSK%2BKiNpzVjlkqVZtqo1AxaKlXiXWkgUzNszRSqZGqjUiswqZRJn2l5KzKplEmfmXkrNKmUSTQ3dyOSPSymt3%2Bql%2Ftnh3nBQ5qn5U%2B0An3UQ1VIj4Li%2F9zkLdGbKBrjRR3PRlgWLZiNTPJPR5iAU8PkOC1KjoAS6IKSKDnogtLDB2dkAoWQRMlJF5CkGVHdiAhLI2QJohiMwiCmpx7VGVH7PKMQejB6YgWb0PezbuYiP%2BhiRJ3UY6A1bYOzBeDsLsCJEqYuwFn3AO5xQHCiJK0LcOAOwFnugOBEeWMX4Ow7AAfs4cDxmeo4xl0DrsFNsALRGzdRbj0OcVdxMwXr3b1xEyX74wh3FTdHITbR4kM3WXXRypIbfn1KPxtZVHbhJpON1cpm0yXfZfPqD8zcnNYTQMDXgU6ol1%2FgrjNAmRedKPafsj0ImUgRTNPQu6T0aXos%2F1Km6BT5rW0M76r4TprMbQmKIAnf6s2LlGM9%2FMRh7vXlF7MOadrQPsW7xEPsokr%2BVjvlEsyZdghMAkRa7eQYy4eWIyvxdnYk2yZr2rNu0LYa6pCtxEvlkS1l20wYOev%2FZdtqyHH6YgtE%2BUqDbbYla3t26GGbyOCSV59IYyldualbeyLiCrC5HYxIQOKdmtC3U8qA3ILLg4lYXFXxrIauiUTa43sbmIAoQRgBvw%2F4KCYjS%2FTMZroeAEu8x%2Bw7gpnTASPYtXPq23FwMEawS4CvnVrfO2BtIpjE%2FLrvCDarP6xpKQxgEtsxb9y%2F3TGAXQIssVV0BKxzAJPY5ao4gNnq4hfvSmP3no3x6xJg%2FdcIhgGsS%2FyyJdYIVKeQKgOYRAZ92%2F5dABwD2DnA2i8RDARYmwAmsUSgeAKmcgnMvvYF1e34tzMGsEuAtV8iGAjwjQYwelr9SkNRvfohDPD8Dw%3D%3D"></iframe>
<div class="legend">Figure 17.2.1: Optimal policies for $R(s) > 0$. In the squares where there are 4 arrows, the agent can decide to go in any of these directions.</div>
</div>


**Answer** According to Figure 17.2.1, we can take whatever policy we want for the squares:
(1,2), (2,1), (3,1), (1,2), (1,3), (2,3). For example, we can choose to always go **right**. We
must note that, again, if we choose to go right the agent will go right with probability 0.8 and it will go
down or up with probability 0.1. Having said that our Transition matrix looks like:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:481px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.1-table#R7VlLc5swEP41zLSHZECymfhY0yS99JRDzwLJwEQgR8gx6a%2BvZEk84njih4xJGl%2BAb7UPLauV8OfBqKjvOVpmvxkm1AM%2Brj340wMgmIBQXhTyopFwZoCU59gMaoGH%2FC8xoG%2FQVY5J1RsoGKMiX%2FbBhJUlSUQPQ5yzdX%2FYgtG%2B1yVKyRbwkCC6jf7Jscg0OvP9Fv9F8jQznqdWEKPkMeVsVRp3HoCLzU%2BLC2RNmfFVhjBbdyB468GIMyb0XVFHhKrU2qxpvbsd0iZsTkqxjwLUCs%2BIroiNOKRSdS5QLJMhZ8Q4JnwjC59WKq550N7KAesmP41cZWPaG1SJF5NaCwpSiytE87T04A%2F1JmXE0o%2BMt2tVSaw5I8lM3t8Q6VivEkYpWlbEGLZPnXjC1Fz1VGMm6%2FQ1yLcQrIpLR9yZiA18l3WZfKXpxFZsgW%2FyHUTB946PeBi%2F4Hi%2Fu0xC9yYn7k2qhAP3E3dsUkUJ3ZoE7k3C40wqcHtVulqnQ68t%2Fzpwa%2B7m0PexlblzKe4z1UvFNpTiUMV7QoM%2Bxu%2F%2BWTuolsG75sZf9Z9X8YBaPvumf%2FmeNUANfqIG6qB23J3uRrgD%2Btez0c5i9IoOiuvkc%2F5HWKtfHe1sm5qbD7oRNqZLxDbew917eXPUidx8eI%2B1XX%2BM%2Bh3BSdBBNbn7G2f8RfE%2FKLr7Rna08X0V1%2BUb0Aku9u1cR218CnyLZFG45pksDnpkEVAskcQzUVAJBPK2Epw9kohRpkipkpVy5HyRU%2FoKYs%2BEL%2BiGW1NSiWBUZQQbM4b3Ilx62MmdBQ0jV9T3hBVEcDkF3yhYDs9QnJZLa%2FkwcBOaIVmHK5xMp4anNBxl2lhuiTp5Y7g6%2B9hyghtZh3eFt%2F8A"></iframe>
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
<iframe frameborder="0" style="width:100%;height:143px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=Figure-17-7#R5ZhNs5sgFIZ%2FjcvOCKhXl4nNbTddZdG1UaLMRUkJqaa%2FvkcFPxIzt52aeidxI7zn8OH7MApaJMyrLzI6ZN9EQrmF7aSyyGcLY4RdBLdaObeKh5xWSCVLdFIvbNkvqkVbqyeW0OMoUQnBFTuMxVgUBY3VSIukFOU4bS%2F4eNRDlNIrYRtH%2FFr9zhKVtapj273%2BlbI0MyO7OrCL4rdUilOhh7Mw2TdXG84j05XOP2ZRIsqBRDYWCaUQqi3lVUh5ba1xrW33eiPaTVvSQv1JA9w2%2BBnxEzUzbualzsaKMmOKbg9RXNdLwG2RdaZyDjUExaOS4q0ziYCyZ5yHggvZtDfPbzIHEbu5IKInQaWi1c0HQZ09sOqoyKmSZ0ipDJq2hV5vRFfLHp6vpWzA7UVrkV4uaddv7xkUtG3TFpJHsRDhxTx0HsVDbC%2FmofswHvqLeehNeOhxVTshYPJDM70fJ2ECn47NB2wFCcg%2FVH0QSml9R6YbmEDbU6tfEQLX1BQUY3UhCnrBRUsRZ2kB1RhcpqCvawYMvmgrHchZktTDrKdWwQzYvDG17m1SDr%2Bf19jwDNhe7oQNPwE25CzHzb8TN%2FIE3DBejltwJ27OE3Aj9nLczOnqH8BhbwwOTmBmE9CL3ZnnErAZbWeE1YD47jLrvVUADwfnwVus%2FmZj84qCMAimNjY733XcmTY27sW%2Bxr0C702A9%2BYAjz4a%2BPXHAL8KQmezngLvxT7d7Wd6Vwf%2FizxU%2B58HTWzwg4ZsfgM%3D"></iframe>
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
<iframe frameborder="0" style="width:100%;height:334px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=Figure-17-8#R7ZnBlpowFIafhuWcQxLEcTnaabvpykXXESLkTCA2xgH79L2YAKLxnFkYnI6ygfy5ucH%2FCwmRgCyK%2Boeim%2FyXTJkIcJjWAfkWYIzwBMGpUfZGiVFkhEzx1Ab1wpL%2FZVYMrbrjKdsOArWUQvPNUExkWbJEDzSqlKyGYWsphr1uaMbOhGVCxbn6m6c6NyqZhL3%2Bk%2FEs1ycVK5q8ZUruSttdgMn6cJjqgrapbPw2p6msjiTyGpCFklKbq6JeMNFY27pm2n2%2FUNvdtmKl%2FkgDbBq8U7Fj7R3HAprO1xIywA3qvfUk%2FrOTbcXT9kDsBQKgt7qvhKusOas2DXRtMhnd%2FuouKa5yrtlyQ5OmXMFggqBcFwJKCC63Wsm3DgFpuudCLKSQCsqlLFmX9J0pzeqLPqDOXRi0TBZMqz2E2Abx1LSwwzWyCaqe%2FbMllh9hbzVqR1vW5e0thwvrupsA8UTgCf1nCFB0MwaRg8HnNAnjm5k0eQxU51yB8HgM4i%2FBwAS1IiyR4eHwM42MiWf6JfB4mKbGhPD8gOCcp3A4HoPZg4F7NhoTQruXelA4nY5GpYA8UfgwBPBNu3w%2F8dlhPRU8K6GYgM8M9HlDgcMu%2BcVWFDxNm27mLtBXADcLh9xm59wiB7foGtx8bY3xHXBD0xuC87WjJncADk9uCO7yDnPVe30NkgGeN4vjBZqdvPr06xoZrmuTc1ixA1Z8DVi%2BtqL3sKzhk4esfVMb4yHz9Wp%2BD8vaCTfk%2BPfNFzfs62X%2BLla1IbfYGzYo9t9uDnVH38fI6z8%3D"></iframe>
<div class="legend">Figure 17.8.1: The reward for each state is indicated. The upper right square is a terminal state.</div>
</div>

**Answer** It would be too cumbersome to run the value iteration algorithm on all the 4 cases by hand. And the question asked to implement value iteration, so I think we should write a program to solve this question. I won't do it. Yet we can try to figure out the policy obtained in each case. I draw a figure with all the
different policies for all the different cases _a_, _b_, _c_, _d_:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:543px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=17.8-policies#R7V1Nc6M4EP01PuxhqhACjI9Jdnf2MlVblcOeCSg2Ndh4MUmc%2FfUrDPJHi1k0WiRVWnEqFVuCNrwnPdStlrKgD9vj1ybbb77VBasWYVAcF%2FTXRRiSKEz4n67kvS9JVkPBuimL4aBLwWP5DxsKg6H0pSzY4ebAtq6rttzfFub1bsfy9qYsa5r67faw57q6%2FdZ9tmZSwWOeVXLpX2XRbvrSOA4u5X%2Bwcr1pQcVTln9fN%2FXLbvi6RUifT6%2B%2BepsJU8Pxh01W1G9XRfS3BX1o6rrt322PD6zqoBWo9ef9%2FoPa82U3bNeqnBD2J7xm1QsTV3y6rvZdQPG2KVv2uM%2Fy7vMbp3tB7zfttuKfCH%2BbHfY9Ac%2FlkXGr94e2qb%2BfUaO85Lmsqoe6qpuTQfqc5izPz0de1TylccSR5GfUu%2FZxuIThnl5Z07LjD2%2BTnMHjbZLVW9Y27%2FyQo%2BCnP2NojXRA%2F%2B1CbTIUba5YFWXZ0JjWZ7sXRPmbAdRxgKkLgPsWNwJwcHrxmhkwJcQZqNEIqEnVDg3nBt3k75daVHw5nHTmjh%2FAb%2FR4qeTv1v3f%2B06AelP8Inpros44bXH3M0ZbcnrNRNvSGW0x2r4A9GVlD9MELaZQXyyCusQL6tIZqClaUEHvJ7E9UFdoQYXd3yaqwgvBCOvSIaxkGla2K%2B46%2F41%2FyqvscCjzW1xlGHsTrJAcuklIrm45HrllUdawKmvL11vzYzgM3%2FBnXZ4GoQLxBCAeAigP9UuTs%2BGsa18NGgonDLVZs2atZOhEy%2Fm21ZhS8AvxMwXbvC5R0M6MPCm4l%2Fh5IulcPQoampGpMZ8VPVMhBHi50mQqnTA0I1MKbio%2BptLoBl%2FdHgXMROb6k4Lji54lAvHVpIkE5nhS8KXR8xRCfHV5MjiSUHDP8fEUBhMdQZUoyZA55aNOQn7TczbDVf0f55S6m1KgeGN%2B1F3En46pyudMjRpvEXXHG964ItQYixFw8UzAiKq7eYVIIar4QVGFCmAT1RAtqlABbIbAI7z5IFACrMKqEF37oLBCDbAKq8IwAJ9TGIHwsrb3Dg0ZdN9jhaEFfqa042GSIXMBsdjJE8CK%2By5lWVpMLcP7AJDzLC3COjYX8um%2F62ZaWiQOb14gVJnEIqp4g4SSytiEFW9qoKQBNmHFG8QDGhCF9lBNFEbaHxRVqAFWYcUbxYMaYBVWL9MkwgAgvtSe1gWGKDA0n1%2BYfCZKdLH8eRIlKCR8Rp68TJSAPOl2KGjHXH9aOnlOO0qTsLg0Cu9zWs6TsAhrOALrZ6BFN1HCInGIJ%2FQchgSWeOO5kszYhBXvEmxJBGzCijfaCkXApvO6xBtuhSJgFVa84VYoAlZh9TNZAqzwoisApapnSNMJQ%2FO5hmJxk%2BdMQd9bmylzTnw65hbhZwoAHIk8rJ9Oa4kmDM3IlJcrkaNkpvglNGQwgJkquFb4mKLRTExBQyaZ8nIlssSUbqxZMmTuObXyckRBpzYkUWZqaouUGZnyU%2F2mNrrQZsrcjhkrL6euaZqCrkD0mIrCCUMzMuXl1DV0g2LdQXr433Zm5MnLqWuofbq7OkHpM7eYY%2BVlJAn2J%2B29gmCHMrlXUODlwI%2FAuIKuMwUNGXSmSKCQuYCPKrgxiTZV0JBRqkIfqZJ6la7jKxky5%2FiSwMu4XzJTKJ0E1kLpRPRfv5iCWzzSUHNUATedlAzNSZWXgT84btOe9oAjSYPTHiTwM0ox1RmUqZrqnnNSNRam%2BMkcy3A8x7K38iQKsl%2BuMi6fLocFDf892Q9OnoNaVmbLju1YgxE5FLt6x0DCxVCUVeV617U63moYL7%2FvkivKPKvuhoptWRTd19yP5XsMaM24TisU6RlXrfS8s%2Bt1M4Xr%2BHXyMUgww%2BZjanw%2FTfP9hXpAN0wWs8z3WNjEFd8e9u5YuJZTbMNBrhbbo%2F%2BjwRXbPnZuy3TL3gvBhC7cc4jIyhmNjOfgRKketvIYVh7EfmBsoQ9oF1z5qURRgZs6BFe0U6yqICT2PJMhS645bOXIHi5VANiOPM7MYSs%2FzXCJwi22I1sWmoNWjp2hkoTzwmYnmoB8pCCBa1UU5PgUKlGA4FpVBTkWhFoVosgmuPIQF7Uq0NQiuMJ79kUV6MiSPXPgyjkhqGSBOFQFKvtmqFSBuBQF2TdDJQoQW6uagDzSCKdcrQZsqOydoRIFmHpgF1zZO0OlCnA5q11wZe8MmSzQ27FCmlgEV3bPkMmCS3CRz0BEkUNwR%2F6nFypZAPOSUbq0iK3snaFSBTgvaRdc2T1DpQpwXtIguPxjU3c5JOe6r%2Fw2Nt%2FqgnVH%2FAs%3D"></iframe>
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
<iframe frameborder="0" style="width:100%;height:323px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=Figure-17-9#R7ZxRb5swEIB%2FTR47YRtI8thk3fayl6XSngm4gEpwRkiT7NfPBAMBXCnqOLsOQaqCz8ZG913O57PTCVlujt8zbxv9ZAFNJtgKjhPydYIxwg7iH4XkVEpcZJeCMIsD0agRrOK%2FVAgtId3HAd21GuaMJXm8bQt9lqbUz1syL8vYod3shSXtUbdeSHuCle8lfenvOMijUjq3rEb%2Bg8ZhJEYmjqhYe%2F5rmLF9KoabYPJyvsrqjVd1JdrvIi9ghwsReZqQZcZYXt5tjkuaFKqttFY%2B9%2B2d2vq1M5rm1zyAywfevGRPqzd2E%2F7o4oXxHvgL5iehE%2FfPnlUVD7szsUfegI92bCr5XVh%2BLgqVlF3x4cveqjrc6hgfojinq63nF%2BUDNyjeKMo3CS8hfrvLM%2FZaYyDFK8RJsmQJy3g5ZSmtO32jWU6P7%2BoC1RrmhkvZhubZiTcRD7iCiTBZIoqHhv9MiKIL9JXMExYX1v02auc3QvNyCgSIwgMyDAGytTGw7wyES8DaGDgSBp9TSdOZNiW5UIZqusNGc3UQpsCWyuOG%2BfmqW17UuOcLxuMipE6J1VD%2FYcrIlpny6vnx17PhxqySAwJyKTwKNG32630fFDoVBBiNm8ahG4Uo5SCLxz%2BnmrpxiFI1QYXMysy1WZ1LZlnrfA2Eae60MGHk9DA5EkxkCEyyqNqglY02SI5CRlBB%2FdWIuNpyGZWOy5J4MS%2BJw5QXfa5myuWLAkLse8mjqNjEQVAMs5CZwQDcOi6wLl9wsyXc7CG4ydYBQ3DDI%2BCGXI3gZkDgyAjAYVsjuDkQOHsE4AjWB64a%2Bj7FfcRVdrhJkgNg3KCSA6OY4jqrM0maGIwbVDJhDDNch5ujENt9H6qarKw2BIVbLGB7LAYm4roYVGaYbmebpZfOVJjex7K1jpFa7NmiQi0SqPnUOMc81eeYCdRJGQMdcxeDSsdMZFGKkS6lp0WVLsXwzPyAm9%2F6zhyRe6z3LgalLuVmYr2eFlW6FKi8tnEuxdF3lJRA5agNdCldDCpdSjW0%2BS6lp0WFLsUeYOHjyq257GVdCb7wpuKvtvF1t7Uhdk%2BuOJhTb4AOjmyAZZIpyNQlH7pEJd9BOKIDpIVNIQr3JZTEAHDEoH%2FMoG6JfIXhg00%2BcKszCLs3eMvLvuJsgIzyEJteNtQxqvlsBOQcjeeobKj1Jndrt0%2FO1XiQyoZapKIio3rz6KYaj1JVxxEA0I3hMNXMUoaOF5vf8p%2FrLv5fAnn6Bw%3D%3D"></iframe>
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
<b>17.14</b> Show that a dominant strategy equilibrium is a Nash equilibrium, but not vice versa.
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
<iframe frameborder="0" style="width:100%;height:278px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=Figure-17-13#R5VhRb5swEP41PCKBabLnhKVdNU2twqQ%2BG3AA1djMOAndr5%2BNDYkLUUiUpKzNS8x357vz57NBn%2BX5efXAYJH%2BojHCFnDiyvK%2BWwC4YOKKP4m8KWTq3ikgYVmsnXZAkP1FGnQ0us5iVBqOnFLMs8IEI0oIiriBQcbo1nRbUWxmLWCCOkAQQdxFX7KYpwqdOM4O%2F4GyJNWZvcYQwug1YXRNdDoLeKv6p8w5bEJp%2FzKFMd3uQd7C8nxGKVejvPIRltQ2rKl59wesbdkMET5kAlATNhCvUVPxFIupcw5DQYZYEWUxYrVt%2Bmct65q7u6Fw2Lb8tHbJxsRwKvmbprYB9bSZMDb%2BjvJNNbE9JlWMHVGMYVEi5dM%2B7SWcJvpfrSWkohHfg6yDxLJ7cJYQo9JIcInY4ehhJ5DgW8a6TPQGWD75P%2FcShDdJ%2Bjx7XizPymoicbbp7QKOKm7r%2Buq9HFpY4D8GwdMyOFJbnXcoTxcs7%2F5xuRBjmBfigYRlYRYwGhpfZr%2BP7u9QDiXYPVMXO2W3PAfi0vGdSwa0xZ3pu9dsQZnAvmqGUazhRk12%2Fr13QlZzudd5jQzs5DF3xX%2FXd7d%2FzV2Z81Nac8w32Cc4DONtVf3Jc9M2Pf8GHXObfo2D8EFtOuSL91Ncp1%2FkxXz1k9DbpxLsExUkroSTBgdGeUDWJfCU51gArhiWnNFX5FNMpcpCKBGe81WG8TuIbhBb4VosktY2uIBF0IP6j9uqSnn1gGiOOBNVO1UjXKkZWqaTUgswNB1wd6dd0j29C0w0CLXOlrSRd2KTGGi9qV978g5qT%2FKAOBGGZWns6awoMLIzuZFRCkmCbIK2OCO9ys9R3i%2FAnjs16XO%2FOR3%2BvB76vNPZE487VbC27Smv3uIf"></iframe>
<div class="legend">Figure 17.17.1: table for rock-paper-scissors-fire-water.</div>
</div>

To find the mixed-strategy solution to this game we will need to find the probability of playing Rock (r), Paper (p), Scissors (s), Fire (f), Water (w),
such that $r + p + s + f + w = 1$. To do so we will firstly focus on what we need to compute if Player _A_ plays Rock. To answer this question we can draw
a graph like this:

<div class="centered-img">
<iframe frameborder="0" style="width:100%;height:357px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=Figure-17-17#R7ZrPk5owFMf%2FGo%2B7AwmwclS72146szN7aHuMEDGzkdgQV%2B1f3wAJSAyz1EXZjnoRXn6%2F7yfhPXQEZ6vdV47Wy%2B8sxnQEnHg3gl9GALjAd%2BVXbtmXlsD1SkPCSawq1YYX8gcro6OsGxLjrFFRMEYFWTeNEUtTHImGDXHOts1qC0abo65Rgo8MLxGix9YfJBZLtQrHqe3fMEmWamQ4VgVzFL0mnG1SNdwIwEXxKYtXSHel6mdLFLPtgQk%2BjuCMMybKq9VuhmnuWu21st1TS2k1bY5T0aUBKBu8IbrBesYBlU2nCyZ7yB1MGS9Kgt%2BbfFbTgyXVJicTe%2BU7bcw7uMsKZSeygpzV7rBFkKjvYrS5Nky0Rc55btaStnJa2gwaIwO5NgmIvJlul0TglzWK8pKtZFTalmJF5Z0rL1G2LqlZkB2WrphmgrPXSmqYT59QOqsWD5%2FccBaGVc2DkvnY93ynms4b5gLvWuVwK5Hl3sFshQXfyyqqAXiAZRO1bTQm25pBTdryAD9tQ4r6pOq4ll5eKPXtJPitJBzJoND4Z8m7wlSO1q66DZDLkTAJZ97j1EZCEI3xfHEeEtzgciiMLSiY%2Fk3jSX7MyruIoiwjUdOt1bnm2BzlFJ8uzg6Lj7QXZ2ouUNEj3hHxM7%2B%2B99XdLzWwdC%2Ffl0VhEGhDXnrn3LuBpy3PmBPpGMxVj%2BUKcXz0WDAEk15gGx7hxqYRiCdY14J2XQ908y26aRvHFAny1pyFTUw1wjMjxWZU2EC3iY1n4FBOXjU6fC4Y%2Fbi6oeoHAqOjcslHHRVoVavuRFv4CWlr58vgsKbtwT%2BErV%2BkvCGRAqHfRGF8GlLARMo9G1K65%2F%2BdqZ6QAsdI%2BYMeUsB4tpkkdEUKGki5%2FvmQcm9IvXNKfSqmTBI6IwUud0rZkq%2BhkeoaZ30Am%2BFio6a0JwdHnnOx4EiHc%2B0J%2BvtZWGBPvKdu17RaZjHCwK5BWspSbCClTIiSJM3ZlVDkofY0z4lIhOhEFaxIHNO2NG308TQqbJ4KEB5nUZ6FOFPQU7Io%2Fcard%2BnurkE4dzygcu3vQm6brsu7iyF3XXDbdacrV73HHkK5hzMp51yBcN6Awumw9RbHDpPqnJzr%2BJeLY0EPcaz1B6T1FWxu18hJQXjBzd1DFGsVLrsC4YADhxOuhyDWKtziGoQzf4e9pHA9RLBW4bZXIByEAwrXQwBrFY5fgXCe%2BXLufMLJ2%2FoPNmUsU%2F%2BJCT7%2BBQ%3D%3D"></iframe>
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
