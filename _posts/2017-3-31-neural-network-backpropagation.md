---
layout: post
title: Backpropagation in Neural Network
---

In this article, I will detail how one can compute the gradient of the **ReLu**, the **bias** and the
**weight matrix** in a fully connected neural network. I wanted to write this article because one half of the
articles available online focus on functions with 1 dimension. The other half don't detail the calculations. I've based my article on the work I've accomplished in the first assignment of the CS231n class from Stanford University.

## Forward pass
Before dealing with the backward pass and the computation of the gradient in higher dimension, let's first compute the forward pass. Using the notations of the assignment, we have:

$$
y_1 = XW_1 + b_1 \\
h_1 = max(0,y_1) \\
y_2 = h_1W_2 + b_2 \\
\mathcal{L}_i = \frac{e^{y_{2_{y_i}}}}{\sum\limits_{j}{e^{y_{2_j}}}} \\
\mathcal{L} = \frac{\sum\limits_s {\mathcal{L}_i}}{N} 
$$

with:
+ $X \in \mathbb{R}^{N \times D}$, $W_1 \in \mathbb{R}^{D \times H}$, $b_1 \in \mathbb{R}^{H}$
+ $h_1 \in \mathbb{R}^{N \times H}$, $W_2 \in \mathbb{R}^{H \times C}$
+ $y_2 \in \mathbb{R}^{N \times C}$, $b_2 \in \mathbb{R}^{C}$


In python code we can compute the forward pass using the following code:
```python
y1 = X.dot(W1) + b1 #(N,H) + (H)
h1 = np.maximum(0, y1)
y2 = h1.dot(W2) + b2
scores = y2

# correspond to e^y2 in maths
exp_scores = np.exp(scores)

# correspond to e^y2/∑e^(y2)[j] in maths
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

# correspond to -log[ (e^y2/∑e^(y2)[j])[yi] ] = Li in maths
correct_logprobs = -np.log(probs[range(N), y])

# correspond to L
loss = np.sum(correct_logprobs) / N
```

## Backpropagation pass
### Gradient of the Softmax
We've already seen (previous article) that the gradient of the softmax function is given by:

$$\frac{\partial \mathcal{L}_i}{\partial f_k} = (p_k \ - \ 1(k \ = \ y_i))$$

where:

$$p_{y_i} = \frac{e^{f_{y_i}}}{\sum\limits_{j}{e^{f_j}}}$$

So actually the gradient of the loss $\mathcal{L}$ with respect to $y_2$ is just the $probs$ matrix (see python code of the forward pass) in which we substract 1 only in the $y_i^{th}$ column. We need to do that for each row of the $probs$ matrix (because each row corresponds to a sample). So in python we can write:

```python
dy2 = probs
dy2[range(N),y] -= 1
dy2 /= N
```

Note : We divide by $N$ because the total loss is averaged over the $N$ samples (see forward pass code).

### Gradient of the fully connected network (weight matrix)
Let's calculate $\frac{d \mathcal{L}}{dW_2}$. To do so, we will use the chain rule. Note that to avoid complex notations, I rewrite $W_2$ as $W$ with $w_{ij}$ being the coefficient of $W$ ($b_2$ is replaced by $b$, $y_2$ by $y$ and $h_1$ by $h$). As $\mathcal{L}$ is a scalar we can compute $\frac{\partial \mathcal{L}}{\partial w_{ij}}$ directly. To do so, we will use the chain rule in higher dimension. Let's recall first that, with our simplified notations, we have:

$$y = hW + b$$
where:
+ $h$ is a ($N$, $H$) matrix, $W$ is a ($H$, $C$) matrix
+ $b$ is a ($C$, 1) column vector

\begin{equation}
\frac{d \mathcal{L}}{d w_{ij}} = \sum\limits_{p,q} \frac{d \mathcal{L}}{d y_{pq}}\frac{d y_{pq}}{d w_{ij}}
\end{equation}

We already know all the $\frac{d \mathcal{L}}{d y_{pq}}$ (this is the term of the $\frac{\partial \mathcal{L}}{\partial y_2}$ we computed in the previous paragraph) so we only need to focus on the calculation of $\frac{d y_{pq}}{d w_{ij}}$:

\begin{equation}
\begin{gathered}
\frac{d y_{pq}}{d w_{ij}} = \frac{d}{d w_{ij}}\left(\sum\limits_{u=1}^H h_{pu}w_{uq} + b_q\right) 
= 1(q=j)h_{pi}
\end{gathered}
\end{equation}

Finally replacing (2) in (1) we have:


$$
\frac{d \mathcal{L}}{d w_{ij}} = \sum\limits_{p,q}\frac{d \mathcal{L}}{d y_{pq}}1(q=j)h_{pi}
= \sum\limits_{p} \frac{d \mathcal{L}}{d y_{pj}}h_{pi}
= \sum\limits_{p} h_{p,i}\frac{d \mathcal{L}}{d y_{pj}}
= \sum\limits_{p} h^{\intercal}_{i,p}\left(\frac{d L}{d y}\right)_{p,j} 
$$

I used the fact the $h_{pi}$ and $\frac{d \mathcal{L}}{d y_{pj}}$ are scalars and $\times$ is a commutative operation for scalars.
Finally we see that:

$$\left(\frac{\partial \mathcal{L}}{\partial W}\right)_{i,j} = \sum\limits_{p} h^{\intercal}_{i,p}\left(\frac{\partial \mathcal{L}}{\partial y}\right)_{p,j}$$

We recognize the product of two matrices: $h^{\intercal}$ and $\frac{\partial \mathcal{L}}{\partial y}$.
Using the assignment notations we have:

$$\frac{\partial \mathcal{L}}{\partial W_2} = h^{\intercal}_1\frac{\partial \mathcal{L}}{\partial y_2}$$

In python we denote $dx$ as $\frac{\partial \mathcal{L}}{\partial x}$, so we can write:

```python
dW2 = h1.T.dot(dy2)
```

### Gradient of the fully connected network (bias)
Let's define $Y=Wx + b$ with $x$ being a column vector. With the notation of the assignment we have:

$$
\begin{bmatrix}
	w_{11} & w_{12} & \ldots & w_{1C}\\
	w_{21} & w_{22} & \ldots & w_{2C}\\
	\vdots & \ddots & \ddots & \vdots\\
	w_{H1} & w_{H2} & \ldots & w_{HC}\\
\end{bmatrix}
\begin{bmatrix}
	x_1\\
	x_2\\
	\vdots \\
	x_C\\
\end{bmatrix}
+
\begin{bmatrix}
	b_1\\
	b_2\\
	\vdots\\
	b_H\\
\end{bmatrix}
=
\begin{bmatrix}
	y_1\\
	y_2\\
	\vdots\\
	y_H\\
\end{bmatrix} \tag{4}
$$

We've already computed $\frac{\partial \mathcal{L}}{\partial y_2}$ (gradient of the softmax), so according to the chain rule we want to compute:

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial y_2}\frac{\partial y_2}{\partial b}$$

Note that here I rewrite $y_2$ as $y$ to simplify the notations.

According to (4), we see that:

$$
\text{$\forall i \neq j$,  } \frac{d y_i}{d b_j} = \frac{d}{d b_j}\left(\sum\limits_{k=1}^{C}w_{ik}x_k + b_i\right)=0 \\
\text{if $i=j$,  } \frac{d y_i}{d b_i} = \frac{d}{d b_i}\left(\sum\limits_{k=1}^{C}w_{ik}x_k + b_i\right)=1
$$

hence we have that $\frac{\partial y_2}{\partial b}$ is the identity matrix (1 on the diagonal and 0 elsewhere). Noting this matrix $I_{HC}$, we have:

$$\frac{\partial \mathcal{L}}{\partial b}=\frac{\partial \mathcal{L}}{\partial y_2}I_{HC}=\frac{\partial \mathcal{L}}{\partial y_2}$$

<div class="centered-img">
	<img src="../images/neural_net/neural_net.png" width="400px" alt="Backpropagation in neural network" />
	<div class="legend">
	Figure 1 : Gradient of the bias. We see that b receives n incoming gradients (red arrows).<br>
	So we have to add all those incoming gradients to get the gradient with respect to the bias</div>
</div>

Now, if $X$ is a matrix we can simply noticed that the gradient is the
sum of all local gradient in $y_i$ (see Figure 1). So we have:

$$\frac{\partial \mathcal{L}}{\partial b} = \sum\limits_{i=1}^{n}\frac{\partial \mathcal{L}}{\partial y_i}\frac{\partial y_i}{\partial b}=\sum\limits_{i=1}^{n}\frac{\partial \mathcal{L}}{\partial y_i}$$

In python, we can achieve the gradient with the following code: 
```python
db2 = np.sum(dy2, axis=0)
```

### Gradient of ReLu
I won't enter into too many details as we understand how it works now. We use the chain rule and the local gradient. Here again I will focus on computing the gradient of $x$, $x$ being a vector. In reality $X$ is actually a matrix as in practice we use mini-batch and vectorized implementation to speed up the computation. For the local gradient we have:

$$
\frac{\partial}{\partial x}\left(ReLu(x)\right) = \frac{\partial}{\partial x}max(0, x) =
\begin{bmatrix}
	\frac{\partial}{\partial x_1}max(0,x_1) & \frac{\partial}{\partial x_2}max(0,x_1) & \ldots & \frac{\partial}{\partial x_H}max(0,x_1)\\
	\frac{\partial}{\partial x_1}max(0,x_2) & \frac{\partial}{\partial x_2}max(0,x_2) & \ldots & \frac{\partial}{\partial x_H}max(0,x_2)\\
	\vdots & \ddots & \ddots & \vdots \\
	\frac{\partial}{\partial x_1}max(0,x_H) & \frac{\partial}{\partial x_2}max(0,x_H) & \ldots & \frac{\partial}{\partial x_H}max(0,x_H)\\
\end{bmatrix} \\
=
\begin{bmatrix}
	1(x_1>0) & 0 & \ldots & 0\\
	0 & 1(x_2>0) & \ldots & 0\\
	\vdots & \ddots & \ddots & \vdots \\
	0 & 0 & \ldots & 1(x_H>0)\\
\end{bmatrix}
$$

and then using the chain rule we have what we want:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial x} = \frac{\partial L}{\partial y}\frac{\partial}{\partial x}\left(ReLu(x)\right) = \frac{\partial L}{\partial y}
\begin{bmatrix}
	1(x_1>0) & 0 & \ldots & 0\\
	0 & 1(x_2>0) & \ldots & 0\\
	\vdots & \ddots & \ddots & \vdots \\
	0 & 0 & \ldots & 1(x_H>0)\\
\end{bmatrix}
$$

In python ($dy1$ being $\frac{\partial L}{\partial x}$ and $dh1$ being $\frac{\partial L}{\partial y}$), we can write:
```python
dy1 = dh1 * (y1 >= 0)
```
Note : I let the reader calculate the gradient of the Relu if $X$ is a matrix. It isn't difficult. We just need to use the chain rule in higher dimension (like I did for the computation of the Gradient w.r.t the weight matrix). I preferred to use $x$ as a vector to be able to visualize the Jacobian matrix of the Relu function.

## Conclusion
In this article we've learned to understand how to compute the gradient in higher dimension. We hence deepen our understanding of what's happening behind the python code and we are ready to compute the gradient of other activation functions or other kind of layers (not necessarily fully connected for example).
<br><br>