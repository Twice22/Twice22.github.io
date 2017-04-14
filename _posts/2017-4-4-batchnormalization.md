---
layout: post
title: Implementing Batch Normalization
---

Batch normalization is a *recent* technique introduced by [Ioffe et al, 2015](https://arxiv.org/abs/1502.03167 "Batchnormalization research paper"). In this article, I will describe how the gradient flow through the batch normalization layer. I based my work on the course given at Stanford in 2016 (CS231n class about Convolutional Neural Network for Visual Recognition). Actually, one part of the 2nd assignment consists in implementing the batch normalization procedure. I will derive the python code associated with each part. Note that the full code is in layers.py of the 2ns assignment. Finally I will also implement a faster way of computing the backward pass.

## Backward pass: Naive implementation

### 1.1 Batch normalization flowchart
<div class="centered-img">
<iframe frameborder="0" style="width:920px;height:600px;" src="https://www.draw.io/?lightbox=1&highlight=0000ff&layers=1&nav=1&title=flow.xml#R7Vxbc5s4FP41nmkf2uF%2BeWyctH3odjKTh%2B0%2BZbBRbLaAXJATe3%2F9SoAwuuDIRLZxa2diw0EIcb5PR%2BccCSb2NNt8KaLV8i8Yg3RiGfFmYt9OLMt0HA%2F%2FEMm2lvieUQsWRRI3hXaCh%2BQ%2F0AhpsXUSg5IpiCBMUbJihXOY52COGFlUFPCFLfYEU%2Faqq2gBBMHDPEpF6d9JjJaN1DKM3YGvIFksm0sH9MAsmv9cFHCdN9ebWPZT9akPZ1FbVy0ol1EMX2pRVYV9N7GnBYSo3so2U5AS3VK11df53HO0bXcBcqRyglWf8Byla0BbXLULbaku8AlY7Xjn5mWZIPCwiubkyAtGHsuWKEvxnok3n2COGijNEO%2BLbWma9wwKBDYdUdO2LwBmABVbXKQ5ajdqbWjU3Izx0sGEMmvZgaMVRg0PFm3NO13gjUYdctXY41aNFdrn040zbt14lns%2B3bjj1k3oc33KO6FuvAvTjXNK4vjjVo5pGGyvciTaCSXKCTXoJrgw3bjB6XQTjls3nhecjTbUn7wU3ZySNqaK75fHn4g7jffmaVSWyXygSkq4LuaA8TlRVCwAYnwtEDNuuai2jlpciVqorABphJJn1pmX6aq5wj1McHtbVFx2FPBsTtv1zTQndZ1rrh7LDpmKAq6eWgNCPRVw7U2rYanidLyCJYas2P7AOwbd%2BYfsfHTJ7iZBP2g5vL07MogBtsgAZ1QMcDhHgEdOlQFmwNbjHpECKr6VMgU%2BGqbN0sDxbAnc%2FcSoTr0HRYLvARQNrZTY4YjscEfFDt9gUfXDYewInP31aCSHim%2Bpy9aPBCXT4Oy4YwztxUJNzvG6sYqnq96NXaYTD7XXrtgjvVFhHXKJIt4%2FUkfa31%2BRRqBV3PYDuyS1xrtB%2Bx8FSqhRQDJmB6PigGk4bEbMGTxq8zUdcdimqakL8dxCkQX%2BuFjARVQ%2Bb6qVnXeTrcg7nilo48bL4AANqhkHbVwssDgWeNSpPJQFYfhKRRpZoCEe7x8ABiE9MgOvq2sL%2FpzPp281ojryqRKTnyo5acpbabJkmOU7yD1S6g2WaPZGlrYwfRZMc6jZszl%2FWqhIYwfRm7ew%2FIAdAEPH6ctGSMfFwVlNSdpiXGlNK%2BTYMTSvaQfc1LN5tMSFdHIDa9rZYOHEnSJY%2Fz4V0Xzi35gT%2Fxb%2FfCc%2F7rRcZ%2Fg7TbIElY9YnFTiu%2B%2Bbx6SuRCAaNpqIpVaJCvgTTGEKCWtymBMz%2FZSkKSeK0mSRE35ijhCC3RATnMyj9FNzIEviOO2z8YRxn6MsSYlGv4L0GZBzj2TxPZNFLxQtfiBhoZY1FTIfx0tRc6cMGN6vNaQHPpSVDj7hAma42lSKoMfx1qL6dafvNvjrPS707vvEwo0xbt%2FT6nHL6is0hf9E5D3O6zEF3GVzTW6oAXfZYpoBuEtBx%2F%2FZmgKPK5iaV9SZNVTcogbqdHVw9yW4O64G3GXeb2O%2Bp7XlxtDtLPnmQyP5Y60zj5bvC2BJ3XG64vJNaMlmCXT10g0Zgefky6oG6duuqb7a6S4HnJDzvk3JGi1TRoI9PqEyCWQTEFV3xA5Vssgi7EFZrPNV%2FioQxpUtQHbBqkxSmBOwL7BHB3rQDD3W7aZZjC6YMvvr6bC%2FMvdZ47i7w5sZfq%2B9uYN%2FYPGxtC0SwDpSb6bZHHEAfpwPiqAISnUURU7zbxqznlCbfmddYlf3dTnY3MohsatbMruto6s7mqb4lbqFymyD8dEjumKnIfH2oDUhktnHcc1AO%2FyjCrqWDPmK804Y2GjbKbYiBUr1Btv8kyLCori95fFG3YKhmR5H5ndodT5Rtrq6nXsXMXs%2BA7Fri5kh07SPNFId01UpUXxNEPTj7rL5XDN0TuehuH0eCsGsShL0uSpEXjkrpORlhhi6nltxuEm8QOJ4GEdyPFzZ6oUDO26wJ6eL%2FxdRRsKM364Da4owPZ97wsJXy%2FBZGjL6rnV09GcAXcHvB5%2BbjPNMEXxZwlAL%2BP1p%2FXIV5Srg24Z0zKYZJ4RDhZJa9hb0uvJLBP13eBbP7U8TvxX1myvIe0A%2B5dNhrqZwrM%2BuwzW6RmJ7PHJ%2B8akleuSyTBI9703Qa4rEpNBTw9748HgLawolEWnAt9qdZ2SEJ1hq38Zkix8I%2FnimBFzI7gWhSBQ60a%2BbKP4J5gvNa8pmD%2Fxu2JN17MIvC%2Fx0RO7%2BUSeXKjuAbwMsIzK5uLlOGr82ZgQO6xIaIhdk67p0UIFe%2BhAqpEkOPtCGVGQgqSBXNmbUNaFohqvghTMYbzXVXwiSmJfMeIEq0y1rVS1ZJGcahzQtygiT8lm5qs%2B936IlzCtdFwnRSEmqFcuJImFf3ndasXCvWCZq5Ow6uo9WuGcy%2BtBzZ0QoUkKBJIfdrwLmlevz2k2pQENbRl%2B4Rq%2FtdJuh1JesHsJi072tjfTh7X2zwvfeVq92N485LLITKpjJbnFDXEd1M%2BHOL0KZaHUeXQpzfBesxWru4ww6bOfJLlh3eBw4V1%2FmlkNdsBLJGtzz6JAu5b5g5VUTFmdS326yZO%2F1pXpVVCAR1l63KJe56IdHZxCHPU9ptU6JxGRYUr1SFZBog7iH7VtTyc6CrGRqtucwI2ua6kJMiDexbK%2F6tEfoi12r19p0Ij%2FyylaX%2FE0UAziyyInGiSl4IveSRjOQ3sAiBgUXUNZH2rfEckd1xIEm9xYEycyAY%2FhiHKjlAZ%2FAFeAlC7Qeml1Y4OBlAfMovdtJOXCVHmfue59Ffaxd5daoFUs%2BVzyqLvAvQGjbgBetESSEa9v1DcKVNEcwIUvzyUdOIFWi9CLMPHcteZaQyl5d76a8kK0HU7y7e%2F9wvZBr95Jn%2B%2B5%2F"></iframe>
<div class="legend">Figure 1: Computational Graph of Batch Normalization Layer</div>
</div>
The Forward pass of the Batch normalization is straightforward. We just have to look at **Figure 1** and
implement the code in Python so I will directly focus on the backward pass. Let's first define some notations:

+ $\mathcal{L}$ design the loss (the quantity computed at the end of all the layers in a neural network)
+ $\frac{\partial \mathcal{L}}{\partial y}$ correspond to the gradient of the loss $\mathcal{L}$ relatively to the last quantity computed during the forward pass of the batch normalization procedure. Note that in python we write $dout$ to design such derivative ($dout$ is then the gradient of $\mathcal{L}$ w.r.t $y$)
+ to make it clear each time I write $dx$ (python notation) it will correspond to the gradient of the loss $\mathcal{L}$ w.r.t to $x$, hence $dx = \frac{\partial \mathcal{L}}{\partial x}$
+ $x$ is a $N \times D$ matrix. Where N is the size of the mini-batch.

Now that we have defined our notations. Let's define the problem. What do we want? Actually during the backward pass we want the gradient of $\mathcal{L}$ w.r.t to all the inputs we used to compute $y$. By looking at **Figure 1**, we see that we want 3 different gradients:

+ $\frac{\partial \mathcal{L}}{\partial \beta} = \frac{\partial \mathcal{L}}{\partial y}\frac{\partial y}{\partial \beta}$ (in python notation $dbeta$)
+ $\frac{\partial \mathcal{L}}{\partial \gamma} = \frac{\partial \mathcal{L}}{\partial y}\frac{\partial y}{\partial \gamma}$ (in python notation $dgamma$)
+ $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}\frac{\partial y}{\partial x}$ (in python notation $dx$)

Since we already know $dout$ ($\frac{\partial \mathcal{L}}{\partial y}$), we only need to compute the partial derivatives of $y$ w.r.t the inputs $\beta$, $\gamma$, $x$. Let's start to compute the backward pass through each step of the Figure 1.


## 1.2 Computation of dbeta
We want to compute:

$$\frac{\partial \mathcal{L}}{\partial \beta}$$

Using the chain rule we can write:

 $$\frac{\partial \mathcal{L}}{\partial \beta}=\frac{\partial \mathcal{L}}{\partial y}\frac{\partial y}{\partial \beta}$$

As we already know $\frac{\partial \mathcal{L}}{\partial y}$ ($dout$), we only need to compute $\frac{\partial y}{\partial \beta}$. However we notice that $y$ is a (N,D) matrix and $\beta$ is a (N,1) vector so we can't compute $\frac{\partial y}{\partial \beta}$ directly. We will instead focus on computing $\forall i \in [1,D]$, $\frac{\partial y}{\partial \beta_i}$. To do so we will use the extended version of the chain rule for higher dimensions.

But let's first see what the $y$ matrix looks like. Indeed, we need to pay attention to the fact that $y$ is obtained using **row-wise summation/multiplication**:

$$y = \gamma \odot \widehat{x} + \beta$$

where I used $\odot$ to highlight the fact that in this relation we are dealing with a row-wise multiplication. Having that in mind we can write:

$$
y =
\begin{bmatrix}
	\gamma_1\\
	\gamma_2\\
	\vdots\\
	\gamma_D\\
\end{bmatrix}
\odot
\begin{bmatrix}
	x_{11} & x_{12} & \ldots & x_{1D}\\
	x_{21} & x_{22} & \ldots & x_{2D}\\
	\vdots & \ddots & \ddots & \vdots\\
	x_{N1} & x_{N2} & \ldots & x_{ND}\\
\end{bmatrix}
+
\begin{bmatrix}
	\beta_1\\
	\beta_2\\
	\vdots\\
	\beta_D\\
\end{bmatrix} \\
=
\begin{bmatrix}
	\gamma_1 x_{11}+\beta_1 & \gamma_2 x_{12}+\beta_2 & \ldots & \gamma_D x_{1D}+\beta_D\\
	\gamma_1 x_{21}+\beta_1 & \gamma_2 x_{22}+\beta_2 & \ldots & \gamma_D x_{2D}+\beta_D\\
	\vdots & \ddots & \ddots & \vdots\\
	\gamma_1 x_{k1}+\beta_1 & \gamma_2 x_{k2}+\beta_2 & \ldots & \gamma_D x_{kD}+\beta_D\\
	\vdots & \ddots & \ddots & \vdots\\
	\gamma_1 x_{N1}+\beta_1 & \gamma_2 x_{N2}+\beta_2 & \ldots & \gamma_D x_{ND}+\beta_D\\
\end{bmatrix}\tag{1.1}
$$

We can easily notice that:

$$\forall i \in [1,D] \, \ \frac{d y_{kl}}{d {\beta}_i}=\frac{d ({\gamma}_l {\widehat{x}}_{kl} + {\beta}_{l})}{d \beta_i}=\frac{d \beta_l}{d \beta_i}=1\{i=l\}$$

We can now use the chain rule for higher dimensions to compute $\frac{\partial \mathcal{L}}{\partial {\beta}_i}$:

$$
\frac{d \mathcal{L}}{d {\beta}_i}= \sum\limits_{k,l}\frac{d \mathcal{L}}{d y_{kl}}\frac{d y_{kl}}{d {\beta}_i} \\
= \sum\limits_{k,l}\frac{d \mathcal{L}}{d y_{kl}}1\{i=l\}
= \sum\limits_{k}\frac{d \mathcal{L}}{d y_{ki}}\tag{1.2}
$$

In python we can compute this quantity using this piece of code:
```python
# Gradient flowing along beta axis
dbeta = np.sum(dout, axis=0)

# Gradient flowing along x_tmp axis
dx_tmp = dout
```

We can retain that:

+ The first gate being an additive gate we only need to multiply the output gradient ($y$) by 1 to get the gradient that flows through $x_{tmp}$ axis.
+ If we are doing a **row-wise summation** during the forward pass, we will need to sum up the flowing gradient over **all columns** during the backward pass.


### 1.3 Computation of dgamma
We want to compute

$$\frac{\partial \mathcal{L}}{\partial \gamma}$$

Once again we use the chain rule: 

$$\frac{\partial \mathcal{L}}{\partial \gamma}=\frac{\partial \mathcal{L}}{\partial x_{tmp}}\frac{\partial x_{tmp}}{\partial \gamma}$$

We already know $\frac{\partial \mathcal{L}}{\partial x_{tmp}}=\frac{\partial \mathcal{L}}{\partial y} (= dout)$ according to the previous paragraph. So we only need to compute:

$$\frac{\partial x_{tmp}}{\partial \gamma}=\frac{\partial y}{\partial \gamma}$$

As $y$ is a (N,D) and $\gamma$ is a (D,1) vector we use the chain rule for higher dimensions:

$$
\frac{d \mathcal{L}}{d {\gamma}_i}= \sum\limits_{k,l}\frac{d \mathcal{L}}{d y_{kl}}\frac{d y_{kl}}{d {\gamma}_i} \\
= \sum\limits_{k,l}\frac{d \mathcal{L}}{d y_{kl}}\frac{d ({\gamma}_l {\widehat{x}}_{kl} + {\beta}_{l})}{d {\gamma}_i} \\
= \sum\limits_{k,l}\frac{d \mathcal{L}}{d y_{kl}}{\widehat{x}}_{kl}1\{i=l\}
= \sum\limits_{k}\frac{d \mathcal{L}}{d y_{ki}}{\widehat{x}}_{ki}\tag{1.3}
$$

Finally $\frac{\partial \mathcal{L}}{\partial \gamma}$ is a (D,1) vector (same shape as $\gamma$) that has on each cell the sum of the row of the $\frac{\partial \mathcal{L}}{\partial y} \widehat{x}$ matrix.
In python we can compute this quantity using this piece of code:
```python
# Gradient flowing along gamma axis
dgamma = np.sum(dout * x_norm, axis=0)

# Gradient flowing along x_norm axis
dx_norm = gamma * dout
```

### 1.4 Computation of dx
To get the gradient of $\mathcal{L}$ w.r.t $x$ we need to backpropgate the gradient through each gate of the Figure 1

#### 1.4.1 First we need to compute $\frac{\partial \mathcal{L}}{\partial x_{c_1}} = dxc1$
$\frac{\partial \mathcal{L}}{\partial x_{c_1}} = \frac{\partial \mathcal{L}}{\partial {\widehat{x}}}\frac{\partial {\widehat{x}}}{\partial x_{c_1}}$. we already know according to step 2 that $\frac{\partial \mathcal{L}}{\partial {\widehat{x}}} = dx\\_norm = gamma \times dout$, so we have:

+ $\frac{\partial {\widehat{x}}}{\partial x_{c_1}} = std^{-1}$ and then the gradient that flows along $x_{c_1}$ axis is $dxc1 = dx\\_norm \times std^{-1}$
+ $\frac{\partial {\widehat{x}}}{\partial std} = \sum\limits_{i=1}^N {x_c} \times std^{-2}$ and the gradient that flows along $std$ axis is:

$$dstd = -dx\_norm \times \sum\limits_{i=1}^N {x_c} \times std^{-2}$$


Why do we have a summation over N for the gradient that flows along $std$ axes?
For the same reason as previously we need to use the chain rule for higher dimensions:

$$
\frac{d \mathcal{L}}{d std_i}= \sum\limits_{k,l}\frac{d \mathcal{L}}{d {\widehat{x}}_{kl}}\frac{d {\widehat{x}}_{kl}}{d std_i} \\
= \sum\limits_{k,l}\frac{d \mathcal{L}}{d {\widehat{x}}_{kl}}\frac{d \frac{x_{c_{kl}}}{std_{k}}}{d std_i} 
= \sum\limits_{k,l}\frac{d \mathcal{L}}{d {\widehat{x}}_{kl}}x_{c_{kl}}\frac{d }{d std_i}\left(\frac{1}{std_{k}}\right) \\
= -\sum\limits_{k,l}\frac{d \mathcal{L}}{d {\widehat{x}}_{kl}}x_{c_{kl}}1\{k = i\}{std_{l}^{-2}}
= \sum\limits_{l}\frac{d \mathcal{L}}{d {\widehat{x}}_{il}}x_{c_{il}}std_{i}^{-2}\tag{1.4}
$$


In python we can implement this gradient using:
```python
# Gradient flowing along std axis
dstd = -np.sum(dx_norm * xc * (std ** -2), axis=0)

# Gradient flowing along xc1 axis
dxc1 = dx_norm * (std ** -1)
```

Note that we could have divided the $x_c, std \to \frac{x_c}{std}$ gate into a **multiply** and a **reverse** gate.


#### 1.4.2 Then we compute $\frac{\partial \mathcal{L}}{\partial \sigma^2} = dvar$
Again we apply the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \sigma^2} = \frac{\partial \mathcal{L}}{\partial std}\frac{\partial std}{\partial \sigma^2}$$

We already know $\frac{\partial \mathcal{L}}{\partial std}$ via the previous computation. Let's then compute: $\frac{\partial std}{\partial \sigma^2}$:

$$\frac{\partial std}{\partial \sigma^2} = \frac{\partial}{\partial \sigma^2}\left(\sqrt{\sigma^2+\epsilon}\right) = 1/2 \times (\sigma^2+ \epsilon)^{-1} = 1/2 \times std^{-1}$$

so finally we have:

$$\frac{\partial \mathcal{L}}{\partial \sigma^2} = 1/2 \times dstd \times std^{-1}$$

and in python we can write:
```python
# Gradient flowing along var axis
dvar = 0.5 * dstd * (std ** -1)
```

#### 1.4.3 We also need to compute $\frac{\partial \mathcal{L}}{\partial x_{c_2}} = dxc2$
By the chain rule we have:

$$\frac{\partial \mathcal{L}}{\partial x_{c_2}} = \frac{\partial \mathcal{L}}{\partial \sigma^2}\frac{\partial \sigma^2}{\partial x_{c_2}}$$

So we just need to compute $\frac{\partial \sigma^2}{\partial x_{c_2}}$. But here $\sigma^2$ is a vector and $x_{c_2}$ is a matrix so we will instead calculate $\frac{\partial \mathcal{L}}{\partial {x_{c2_{kl}}}}$ $\forall k \in [1, N]$, $\forall l \in [1, D]$:

$$
\frac{d \mathcal{L}}{d {x_{c2_{kl}}}} = \sum\limits_{i}\frac{d \mathcal{L}}{d \sigma^2_i}\frac{d \sigma^2_i}{d {x_{c2_{kl}}}} \\
= \sum\limits_{i}\frac{d \mathcal{L}}{d \sigma^2_i} \frac{1}{N}\frac{d}{d {x_{c2_{kl}}}}\left(\sum\limits_{p=1}^N {x^2_{c2_{pi}}} \right)
= \sum\limits_{i}\frac{d \mathcal{L}}{d \sigma^2_i} \frac{2}{N}1\{l=i\}{x_{c2_{kl}}} \\
= \frac{2}{N}\frac{d \mathcal{L}}{d \sigma^2_l}{x_{c2_{kl}}}\tag{1.5}
$$

So finally we can easily see that in term of matrix multiplication we have :

$$\frac{\partial \mathcal{L}}{\partial x_{c_2}} = dvar * \frac{2}{N}x_c$$

In python we can write:
```python
# Gradient flowing along xc2 axis
# very important 2.0 / N and not 2 / N
# because we are using python 2.7
dxc2 = (2.0 / N) * dvar * xc 
```

#### 1.4.4 Again we need $\frac{\partial \mathcal{L}}{\partial x_c} = dmu$
here we have two different gradients that are coming to the $\mu \to x-\mu$ gate so we have to add those two different gradients:

$$\frac{\partial \mathcal{L}}{\partial x_c} = \frac{\partial \mathcal{L}}{\partial x_{c_{1}}} + \frac{\partial \mathcal{L}}{\partial x_{c_{2}}} = \frac{\partial \mathcal{L}}{\partial \widehat{x}} \times std^{-1} + \frac{2}{N}\frac{\partial \mathcal{L}}{\partial var} \times x_c$$

In python we have:
```python
# dxc = dxc1 + dxc2 (two incoming gradients)
dxc = dxc1 + dxc2 # (= dx_norm*std**-1 + (2 / N) * dvar * xc)
```

Also, using the same procedure as in step 1 and 2, the gradient that flows to  $\mu$ is the sum over N of the incoming gradient:

$$\frac{\partial \mathcal{L}}{\partial \mu} = -\sum\limits_{i=1}^N \frac{\partial \mathcal{L}}{\partial x_{c_{ij}}}$$

Hence in python we have:
```python
# Gradient flowing along mu axes
dmu = -np.sum(dxc, axis=0)
```

#### 1.4.5 Finally we are able to compute $\frac{\partial \mathcal{L}}{\partial x} = dx$
Finally we can recover $\frac{\partial \mathcal{L}}{\partial x}$. Again using the chain rule and the fact that the last gate receives 2 incoming gradients, we have:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \mu}\frac{\partial \mu}{\partial x} + \frac{\partial \mathcal{L}}{\partial x_{c}}\frac{\partial x_{c}}{\partial x}$$

Let's compute $\frac{\partial \mu}{\partial x}$ first. As $\mu$ is a vector and $x$ is a matrix we will instead compute $\frac{\partial \mathcal{L}}{\partial x_{k,l}}$ using the chain rule for higher dimensions. (Note that here I'm only computing $\frac{\partial \mathcal{L}}{\partial \mu}\frac{\partial \mu}{\partial x}$):

$$
\frac{d \mathcal{L}}{d x_{kl}} = \sum\limits_{i}\frac{d \mathcal{L}}{d \mu_i}\frac{d \mu_i}{d x_{kl}} \\
= \sum\limits_{i}\frac{d \mathcal{L}}{d \mu_i} \frac{1}{N}\frac{d}{d x_{kl}}\left(\sum\limits_{p=1}^N {x_{pi}} \right)
= \sum\limits_{i}\frac{d \mathcal{L}}{d \mu_i} \frac{1}{N}1\{l=i\} \\
= \frac{1}{N}\frac{d \mathcal{L}}{d \mu_l}\tag{1.6}
$$

So finally using matrix notations we have:

$$\frac{\partial \mathcal{L}}{\partial \mu}\frac{\partial \mu}{\partial x} = \frac{1}{N}\frac{\partial \mathcal{L}}{\partial \mu}$$

Now, let's compute $\frac{\partial \mathcal{L}}{\partial x_{c}}\frac{\partial x_{c}}{\partial x}$:

$$
\frac{\partial \mathcal{L}}{\partial x_{c}}\frac{\partial x_{c}}{\partial x} = \frac{\partial \mathcal{L}}{\partial x_{c}}\frac{\partial}{\partial x}\left(x-\mu \right)=\frac{\partial \mathcal{L}}{\partial x_{c}}I_{ND} = \frac{\partial \mathcal{L}}{\partial x_{c}}\tag{1.7}
$$

Here $I_{ND}$ is the $N \times D$ identity matrix. Finally we have:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{N}\frac{\partial \mathcal{L}}{\partial \mu} + \frac{\partial \mathcal{L}}{\partial x_c}$$

In python we can write:
```python
#final gradient dL/dx
dx = dxc + dmu / N
```

## Backward pass: Faster implementation
In this part I will derive a faster implementation of the backward pass using the chain rule in higher dimension. I'll first define the problem correctly using the notations from the CS231n assignment.

### Goal
Our objective didn't change. We still want to compute $\frac{\partial \mathcal{L}}{\partial x}$, $\frac{\partial \mathcal{L}}{\partial \gamma}$ and $\frac{\partial \mathcal{L}}{\partial \beta}$.
We already saw in the first part how to compute $\frac{\partial \mathcal{L}}{\partial \gamma}$ and $\frac{\partial \mathcal{L}}{\partial \beta}$ directly. We will therefore only focus on the calculation of $\frac{\partial \mathcal{L}}{\partial x}$.

### Problem
Before attacking the problem, let's define it correctly:
We have :

$$
X = 
\begin{bmatrix}
	x_{11} & x_{12} & \ldots & x_{1l} & \ldots & x_{1D}\\[10pt]
	x_{21} & x_{22} & \ldots & x_{2l} & \ldots & x_{2D}\\[10pt]
	\vdots & \ddots & \ddots & \ddots & \vdots \\[10pt]
	x_{k1} & x_{k2} & \ldots & x_{kl} & \ldots & x_{kD}\\[10pt]
	\vdots & \ddots & \ddots & \ddots & \vdots \\[10pt]
	x_{N1} & x_{N2} & \ldots & x_{Nl} & \ldots & x_{ND}\\[10pt]
\end{bmatrix}
\mu =
\begin{bmatrix}
	\mu_{1}\\[10pt]
	\mu_{2}\\[10pt]
	\vdots\\[10pt]
	\mu_{k}\\[10pt]
	\vdots\\[10pt]
	\mu_{D}\\[10pt]
\end{bmatrix}
\sigma^{2} = 
\begin{bmatrix}
	{\sigma_{1}}^{2}\\[10pt]
	{\sigma_{2}}^{2}\\[10pt]
	\vdots\\[10pt]
	{\sigma_{k}}^{2}\\[10pt]
	\vdots\\[10pt]
	{\sigma_{D}}^{2}\\[10pt]\nonumber
\end{bmatrix}
$$

so actually when we write

$${\widehat{x}} = \frac{x-\mu}{\sqrt{\sigma^{2} + \epsilon}}$$

it actually means that $\forall k \in [1,N]$, $\forall l \in [1,D]$

$${\widehat{x}_{kl}} = (x_{kl}-\mu_{l})({\sigma^2}_l + \epsilon)^{-1/2}$$

We want to compute $\frac{\partial \mathcal{L}}{\partial x}$. To do so we will use the chain rule for higher dimensions:

$$\frac{d \mathcal{L}}{d x_{ij}} = \sum\limits_{\substack{k \in [1,N] \\ l \in [1,D]}} \frac{d \mathcal{L}}{d {\widehat{x}_{kl}}}\frac{d {\widehat{x}_{kl}}}{d {x_{ij}}}$$

We don't know the derivatives in the summation and we don't know how to compute $\frac{d \mathcal{L}}{d {\widehat{x}_{kl}}}$ because we don't have access to $\mathcal{L}$ directly. Yet we have access to $\frac{d \mathcal{L}}{d {y}}$ (that is our $dout$ in Python notation). So we will introduce this term in the chain rule and it give us:

$$
\frac{d \mathcal{L}}{d x_{ij}} = \sum\limits_{\substack{k \in [1,N] \\ l \in [1,D]}} \frac{d \mathcal{L}}{d {y_{kl}}}\frac{d y_{kl}}{d {\widehat{x}_{kl}}}\frac{d {\widehat{x}_{kl}}}{d {x_{ij}}}\tag{2.1}
$$

So now we need to compute $$\frac{d y_{kl}}{d {\widehat{x}_{kl}}}$$ and $$\frac{d {\widehat{x}_{kl}}}{d {x_{ij}}}$$. Because we have access to the expression of both $y$ and $\widehat{x}$, we can compute these derivatives, so let's do it:

$$
\frac{d y_{kl}}{d {\widehat{x}_{kl}}} = \frac{d \gamma_{l}{\widehat{x}_{kl}} + \beta_{l}}{d {\widehat{x}_{kl}}} = \gamma_{l}\tag{2.2}
$$

This one was straightforward!
Now let's calculate the other derivative:

$$
\frac{d {\widehat{x}_{kl}}}{d {x_{ij}}} = \frac{d (x_{kl}-\mu_{l})*({\sigma^2}_l + \epsilon)^{-1/2}}{d {x_{ij}}} \\
= \frac{d (x_{kl}-\mu_{l})}{d x_{ij}}({\sigma^{2}}_l + \epsilon)^{-1/2} + (x_{kl}-\mu_{l})\frac{d ({\sigma^{2}_{l}} + \epsilon)^{-1/2}}{d x_{ij}}\tag{2.3}
$$

Again we calculate the first derivative of (2.3):

$$
\frac{d (x_{kl}-\mu_{l})}{d x_{ij}} = \frac{d x_{kl}}{d x_{ij}} - \frac{d \mu_{l}}{d x_{ij}} = 1\{i = k,\ j=l\}-\frac{d}{d x_{ij}}\left(\frac{1}{N}\sum\limits_{i=1}^N x_{il}\right) \\
= 1\{i = k,\ j=l\} - \frac{1}{N}1\{j = l\}\tag{2.4}
$$

This one was quite straightforward, let's handle the other derivative:

$$
\frac{d ({\sigma^{2}_{l}} + \epsilon)^{-1/2}}{d x_{ij}} = -\frac{1}{2}\frac{d ({\sigma^{2}_{l}} + \epsilon)}{d x_{ij}}({\sigma^{2}_{l}} + \epsilon)^{-3/2}\tag{2.5}
$$

So we need to compute $$\frac{d (\sigma^2_l + \epsilon)}{d x_{ij}}$$:

$$
\frac{d ({\sigma^{2}_{l}} + \epsilon)}{d x_{ij}} = \frac{d}{x_{ij}}\left(\frac{1}{N}\sum\limits_{q=1}^N (x_{ql}- \mu_{l})^{2}\right) \\
= \frac{2}{N}\sum\limits_{q=1}^N (x_{ql}-\mu_{l})\frac{d}{d x_{ij}}(x_{ql}-\mu_{l})\tag{2.6}
$$

Using equation (2.4) we have:

$$
\frac{d ({\sigma^{2}_{l}} + \epsilon)}{d x_{ij}} = \frac{2}{N}\sum\limits_{q=1}^N (x_{ql}-\mu_{l})(1\{i=q, \ j=l \} - \frac{1}{N}1\{j=l\}) \\
= \frac{2}{N}\left[\sum\limits_{q=1}^N (x_{ql}-\mu_{l})1\{i=q, \ j=l \} - \frac{1}{N}\sum\limits_{q=1}^N (x_{ql}-\mu_{l})1\{j=l\})\right] \\
= \frac{2}{N}\left[(x_{il}-\mu_{l})1\{j = l\} - \frac{1}{N}1\{j = l\}\left(\sum\limits_{q=1}^N x_{ql}-\mu_{l}\right)\right]\tag{2.7}
$$

To simplify even more this last expression, let's focus on the sum:

$$
\sum\limits_{q=1}^N x_{ql}-\mu_{l} = \sum\limits_{q=1}^N x_{ql}-\sum\limits_{q=1}^N \mu_{l} \\
\triangleq N\mu_{l} - \mu_{l}\sum\limits_{q=1}^N 1 = N\mu_{l} - N\mu_{l} = 0\tag{2.8}
$$

So finally, the second term in (2.7) disappear and we have:

$$
\frac{d ({\sigma^{2}_{l}} + \epsilon)}{d x_{ij}} = \frac{2}{N}(x_{il}-\mu_{l})1\{j = l\}\tag{2.9}
$$

Combining (2.3), (2.5) and (2.9) we finally have:

$$
\frac{d ({\sigma^{2}_l} + \epsilon)^{-1/2}}{d x_{ij}} = \left(1\{i=k, \ j=l\} - \frac{1}{N}1\{j = l\}\right)({\sigma^2_l}+\epsilon)^{-1/2}-\frac{1}{N}(x_{kl}+\mu_l)({\sigma^2_l}+\epsilon)^{-3/2}(x_{il}-\mu_l)1\{j=l\}\tag{2.10}
$$

Finally using (2.1), (2.2), (2.10) we can determine a beautiful expression for $\frac{\partial \mathcal{L}}{\partial x}$:

$$
\frac{d \mathcal{L}}{d x_{ij}} = \sum\limits_{\substack{k \in [1,N] \\ l \in [1,D]}} \frac{d \mathcal{L}}{d {y_{kl}}}\frac{d y_{kl}}{d {\widehat{x}_{kl}}}\frac{d {\widehat{x}_{kl}}}{d {x_{ij}}} \\
= \sum\limits_{\substack{k \in [1,N] \\ l \in [1,D]}} \frac{d \mathcal{L}}{d {y_{kl}}}\gamma_l\left(\left[1\{i=k,\ j=l\} - \frac{1}{N}1\{j = l\}\right]({\sigma^2_l}+\epsilon)^{-1/2}-\frac{1}{N}(x_{kl}+\mu_{l})({\sigma^2_l}+\epsilon)^{-3/2}(x_{il}-\mu_{l})1\{j=l\}\right) \\
= \sum\limits_{\substack{k \in [1,N] \\ l \in [1,D]}} \frac{d \mathcal{L}}{d {y_{kl}}}\gamma_l\left[1\{i=k,\ j=l\} - \frac{1}{N}1\{j = l\}\right]({\sigma^2_l}+\epsilon)^{-1/2} \\ -
\sum\limits_{\substack{k \in [1,N] \\ l \in [1,D]}} \frac{d \mathcal{L}}{d {y_{kl}}}\gamma_l\frac{1}{N}(x_{kl}+\mu_{l})({\sigma^2_l}+\epsilon)^{-3/2}(x_{ij}-\mu_{l})1\{j=l\} \\
= \frac{1}{N}({\sigma^2_l}+\epsilon)^{-1/2}\gamma_j\sum\limits_{k=1}^N \frac{d \mathcal{L}}{d {y_{kl}}}(1\{i=k\}N - 1) - \frac{1}{N}({\sigma^2_l} + \epsilon)^{-3/2}(x_{ij}-\mu_{j})\sum\limits_{k=1}^N \frac{d \mathcal{L}}{d {y_{kj}}}\gamma_j(x_{kj}-\mu_{j}) \\
= \frac{1}{N}({\sigma^2_j}+\epsilon)^{-1/2}\gamma_j\left(\left[N\sum\limits_{k=1}^N\frac{d \mathcal{L}}{d {y_{kj}}}1\{i=k\} - \sum\limits_{k=1}^N\frac{d \mathcal{L}}{d {y_{kj}}}\right] -
({\sigma^2_j}+\epsilon)^{-1}(x_{ij}-\mu_{j})\sum\limits_{k=1}^N \frac{d \mathcal{L}}{d {y_{kj}}}(x_{kj}- \mu_j)\right) \\
= \frac{1}{N}({\sigma^2_j}+\epsilon)^{-1/2}\gamma_j\left(N\frac{d \mathcal{L}}{d {y_{ij}}} - \sum\limits_{k=1}^N\frac{d \mathcal{L}}{d {y_{kj}}} -
({\sigma^2_j}+\epsilon)^{-1}(x_{ij}-\mu_{j})\sum\limits_{k=1}^N \frac{d \mathcal{L}}{d {y_{kj}}}(x_{kj}- \mu_j)\right)
$$

We finally have an expression for $\frac{d \mathcal{L}}{d x_{ij}}$.
We just need to recall that $\frac{d \mathcal{L}}{d x}$ is a (N,D) matrix (same shape as $x$) that looks like:

$$
\begin{bmatrix}
	\frac{d \mathcal{L}}{d x_{11}} & \frac{d \mathcal{L}}{d x_{12}} & \ldots & \frac{d \mathcal{L}}{d x_{1l}} & \ldots & \frac{d \mathcal{L}}{d x_{1D}}\\
	\frac{d \mathcal{L}}{d x_{21}} & \frac{d \mathcal{L}}{d x_{22}} & \ldots & \frac{d \mathcal{L}}{d x_{2l}} & \ldots & \frac{d \mathcal{L}}{d x_{2D}}\\
	\vdots & \ddots & \ddots & \ddots & \vdots \\
	\frac{d \mathcal{L}}{d x_{k1}} & \frac{d \mathcal{L}}{d x_{k2}} & \ldots & \frac{d \mathcal{L}}{d x_{kl}} & \ldots & \frac{d \mathcal{L}}{d x_{kD}}\\
	\vdots & \ddots & \ddots & \ddots & \vdots \\
	\frac{d \mathcal{L}}{d x_{N1}} & \frac{d \mathcal{L}}{d x_{N2}} & \ldots & \frac{d \mathcal{L}}{d x_{Nl}} & \ldots & \frac{d \mathcal{L}}{d x_{ND}}\\
\end{bmatrix}
$$


Having this in mind we can actually come up with the python implementation that looks like:
```python
  N = dout.shape[0]
  dx = (1. / N) * (var + eps)**(-1./2) * gamma \
  		* (N * dout - np.sum(dout, axis=0)\
  		- (var + eps)**(-1.0) * (x - mu.T) \
  		* np.sum(dout * (x - mu.T), axis=0))
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * x_norm, axis=0)
```

## Conclusion
In this article we've learned how we can implement batch normalization in Python. To do so we have drawn the computational graph of the batch normalization layer. The backward pass can then be computed directly using this graph. The most important thing to remember is to use **the chain rule for higher dimensions**.
<br><br>