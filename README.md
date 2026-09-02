# Concept: Spectral Operator Chain (SOC) or Fourier Neural Operators

<img width="1290" height="784" alt="image" src="https://github.com/user-attachments/assets/eb1974c2-5d91-46da-848b-1f32c4f215c2" />  

*early testing*

# Comparison

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/2394ecb1-c542-405b-9a97-e8d0b499782d" />  
*Generated Illustration*  
  
  
| Feature / Dimension | Fourier Spectral Approach (SOC/FNO) | Standard CNNs (ResNet, YOLO, etc.) |
| --- | --- | --- |
| **Input Representation** | 1D closed contour ($z(t)$ or Fourier vector $\mathbf{S}$) | 2D Pixel Grid ($H \times W \times C$) |
| **Invariances** | **Exact by design** (rotation, scale, position are strictly decoupled) | **Learned / Approximate** (requires data augmentation like rotating images) |
| **Reversibility** | Lossless (Fourier pairs are 100% invertible; clear taxonomy of loss) | Lossy (Pooling and activations discard spatial info irreversibly) |
| **Data Efficiency** | High (Zero training needed for raw metric comparison $\eta$) | Low (Requires thousands of labeled images to generalize) |
| **Compute / Speed** | Fast ($O(N \log N)$ FFT-based calculation) | Heavy (Millions of parameter multiplies per image frame) |
| **Occlusion Handling** | **Poor** (A broken or hidden contour corrupts all Fourier frequencies) | **Strong** (Local kernels can recognize a car even if 50% is hidden) |
| **Resolution Dependence** | Resolution-invariant (can sample $z(t)$ at any continuous point) | Fixed grid-size (sensitive to image pixel resolution) |

## 1. Object

The model operates on **finite-dimensional complex vectors representing a closed curve in a Fourier basis**:  

$$\mathbf{S} \in \mathbb{C}^{2N+1}, \quad \mathbf{S} = (C_{-N}, \dots, C_0, \dots, C_N)$$

*with spatial reconstruction*

$$z(t) = \sum_{k=-N}^{N} C_k \, e^{i2\pi k t}, \quad t \in [0,1)$$

A closed, periodic, band-limited complex-valued function reconstructed linearly from $\mathbf{S}$.  

The model acts as:  
- An Encoder: It takes a 2D closed shape (boundary contour) and encodes it into a compact list of complex numbers ($\mathbf{S}$),  
  breaking the shape down into basic circular harmonics.  
- A Feature Comparator: It compares two encoded shapes using a metric ($\eta$) that focuses purely on geometry while ignoring position, scale, and start-point phase.
- A Decoder: It converts those complex numbers back into the original spatial boundary ($z(t)$) without losing information.

**Worth stating plainly:**  
We establish strict algebraic rules for how to manipulate and compare those descriptors,  
$\mathbf{S}$ and $z(t)$ are related by a discrete Fourier transform pair.  
Every operation below can equivalently be described as acting on $\mathbf{S}$ (coefficient space) or on $z(t)$ (reconstructed space),  
they are the same operation viewed through two lossless, invertible representations.  
This is the one guarantee the whole model rests on.  

---

## 2. Verified invariants

These are not design choices, they are consequences of the definitions, confirmed both analytically,  
and numerically (θ swept 0→π, gain swept 0.1→7.5, exact agreement to floating-point precision):  

| Transform applied to $\mathbf{S}$ | Effect (*) |
|---|---|
| Global phase rotation, $\mathbf{S} \to \mathbf{S}\,e^{i\theta}$ | **None.** $\eta$ invariant for all $\theta$. |
| Uniform gain, $\mathbf{S} \to g\mathbf{S},\ g \in \mathbb{R}_{>0}$ | **None.** $\eta$ invariant for all $g$. |
| Modifying $C_0$ alone | **None.** $\eta$ is defined to exclude $k=0$ entirely. |

**Effect** on η(S, T) = |⟨S, T⟩ₖ≠₀| / (‖S‖ₖ≠₀ · ‖T‖ₖ≠₀)  

**Consequence:**  
$\eta$ does not compare vectors in $\mathbb{C}^{2N+1}$. It compares **rays** in the $2N$-dimensional subspace $\{k \neq 0\}$,  
i.e. points in $\mathbb{CP}^{2N-1}$. Two states that differ only by phase, uniform scale,  
or their $C_0$ term are indistinguishable to $\eta$, by construction, not by approximation.  

This means $C_0$ carries information ("where," "how much bias," "what baseline")  
that is **structurally decoupled** from everything $\eta$ measures ("what shape," "what relative pattern").  
Any downstream quantity that is supposed to represent *match quality* must not depend on $C_0$,  
or it silently reintroduces a variable the metric was built to ignore. (See §5 for the concrete failure mode this produces.)  

---

## 3. Operator taxonomy (by reversibility)

Every operator in the model was checked against one property:  
**is it unitary** (norm-preserving, therefore invertible with zero information loss)?  
This produces three non-overlapping classes, not a spectrum:  

### Class A, Always unitary
- **Phase rotation** $R_\theta$: $C_{k,out} = C_{k,in}\,e^{i\theta}$.
  Diagonal, unit-modulus, invertible by $R_{-\theta}$ for any $\theta$, always.

### Class B, Conditionally unitary
- **Index shift** $H_m$: $C_{k,out} = C_{k-m,in}$.
  Unitary **iff** no nonzero coefficient is shifted past the boundary $\pm N$.
  Verified numerically: at $m=8$ (with $N=8$), 0.538 of 0.551 total energy  
  (≈98%) is truncated and unrecoverable. The operator is a lossless permutation  
  in-bounds and an irreversible projection out-of-bounds, the same formula,  
  two different behaviors, depending only on how far the shift pushes energy  
  toward the boundary.  

### Class C, Never unitary (once active)
- **Magnitude filtering** $W$ (any $w_k \neq 1$ for any $k$), **attenuation**,  
  **noise injection**, **norm clamping**. Each of these either discards energy  
  (filter, clamp) or injects new, non-recoverable degrees of freedom (noise).  
  None has a general inverse.  

**Why this split matters structurally, independent of application:**  
Class A operators can be composed freely with no accounting needed, order and repetition don't cost anything.  
Class B operators require a boundary/bandwidth check before composition, or the operation silently becomes lossy.  
Class C operators require explicit energy bookkeeping every time,  
they are the only place in the model where "how much was lost" is a well-formed, necessary question.

---

## 4. Verified frequency-dependent degradation

Given attenuation of the form $C_{k,out} = C_{k,in}\,e^{-\alpha|k|d}$ (decay
proportional to distance $d$ and harmonic index $|k|$):

- Low-$|k|$ (coarse-structure) components decay slower than high-$|k|$
  (fine-structure) components, for any $\alpha, d > 0$.
- Because $\eta$ depends on *ratios* between harmonics (not absolute magnitudes, 
  see §2), differential decay rates change those ratios over $d$, which measurably
  degrades $\eta$ even before any noise term is added.
- **Consequence, generally stated:** any state whose energy is concentrated in
  low-$|k|$ modes is more robust to this class of degradation than one with
  energy spread into high-$|k|$ modes, independent of what the modes represent.
  This is a property of the exponential-in-$|k|$ decay law, not of any specific
  use of the model.

---

## 5. A verified failure mode

Given a readout of the form:

$$y = \gamma \cdot \|\mathbf{S}\|^2_{\text{full}} \cdot \eta(\mathbf{S}, \mathbf{T})$$

where $\|\mathbf{S}\|^2_{\text{full}}$ includes $C_0$ and $\eta$ excludes it (§2):  

**Numerically confirmed:**  
boosting $|C_0|$ by 20× while holding the $k\neq0$ subspace fixed moved $y$ from 0.73 → 73.5 (100×)  
while $\eta$ stayed exactly 1.000000 and the $C_0$-excluding variant of $y$ stayed exactly constant.  

This is not a bug in arithmetic, it is what necessarily happens whenever a scalar readout is built  
by multiplying a metric that is invariant to some subspace by a norm that is *not* invariant to that same subspace.  
The general lesson, stated without reference to any application; if a similarity metric explicitly excludes a component,  
any downstream formula multiplying that metric by a magnitude must exclude the same component, or the exclusion is undone at the next step.  
This is worth keeping as a named finding, call it the **decoupled-term leak**, because it will recur in any pipeline built from this model,  
regardless of what $C_0$ ends up meaning in a given use.  

---

## 6. Correspondences checked against external formalisms

Each of the following was tested against precise definitions, not vocabulary
overlap. Two survived, three did not.

**Survives, exact structural match:**

- **η ≡ quantum state fidelity.** For normalized vectors,
  $\eta = |\langle \mathbf{A}, \mathbf{B}\rangle| / (\|\mathbf{A}\|\|\mathbf{B}\|)$
  is the same formula as $|\langle\psi|\phi\rangle|$, the Born-rule overlap
  between two pure quantum states. Not an analogy, the same expression.

- **Class A/B/C taxonomy (§3) ≡ unitary vs. non-unitary quantum operators.**
  A phase rotation is unitary in exactly the sense a quantum gate must be;
  a lossy filter is precisely what a quantum channel with decoherence looks
  like. This correspondence is exact at the level of the linear-algebra
  definitions, independent of whether "quantum" means anything else here.

**Partially survives, same primitive, missing structure:**

- **Per-mode multiplicative filtering ($C_{out,k} = w_k C_{in,k}$) ≡ core
  operation of spectral neural operators** (e.g. Fourier Neural Operators,
  Global Filter Networks). The elementwise-multiply-in-frequency-domain
  primitive is identical. What is *not* present in this model: complex-valued
  weights (here $w_k$ is real, so it can only rescale magnitude, never rotate
  phase), multi-channel mixing (no analog of a per-mode matrix), stacked
  nonlinear layers, and any optimization loop fitting $w_k$ to data. The
  primitive is shared; the system built from it is not.

**Ruled out, vocabulary overlap only, mechanism does not match:**

- ~~Index shift $H_m$ ≡ attention.~~ Attention weights are computed from
  content (learned query–key similarity); $H_m$ is a fixed structural
  permutation independent of coefficient values. No shared mechanism.
- ~~Uniform gain invariance ≡ batch normalization.~~ BatchNorm is a
  data-dependent statistic computed per training batch to stabilize
  gradients; gain invariance here is a static algebraic property of a
  cosine-similarity formula. Both involve "scale," nothing else.
- ~~$C_0$-exclusion ≡ CNN translation invariance.~~ $C_0$-exclusion discards
  one specific coordinate (global offset). CNN translation invariance comes
  from weight-sharing across sliding spatial windows, a different
  mechanism producing a superficially similar-sounding invariance.

---

## 7. Open parameters (the model does not decide these, implementation must)

- **$H_m$ boundary behavior:** wrap (circular, information-preserving) vs.
  truncate (linear, lossy) at $\pm N$. Both are valid readings of
  "$C_{k,out} = C_{k-m,in}$"; the spec as written does not disambiguate,
  and the two choices put $H_m$ in different rows of §3's taxonomy.
- **Readout formula:** whether a scalar output should scale with
  $\|\mathbf{S}\|^2_{\text{full}}$ or $\|\mathbf{S}\|^2_{\text{shape}}$
  (§5) is a decision about what the $C_0$ term is allowed to control
  downstream, the math doesn't pick a side, but the two choices produce
  materially different systems.
- **Threshold sharpness:** the model as specified uses a hard cutoff on
  $\eta$ (step function). Nothing in the derivation requires this, a
  continuous/soft readout is equally consistent with everything above,
  and changes the nature of the boundary between "no match" and "match."

---

## 8. Reason

**The verified structure is:**  
A lossless dual representation (§1), a metric with a provable invariance subspace (§2), a strict three-class operator taxonomy by reversibility (§3),  
a derived degradation asymmetry by frequency (§4), one concretely reproducible failure mode from combining components incorrectly (§5),  
two exact and one partial correspondence to established formalisms (§6), and three explicitly undecided parameters (§7).  

