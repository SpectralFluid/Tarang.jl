# AMD numerical range and backend validation

The AMD viscosity forms cubic velocity-gradient products divided by a squared
gradient norm. Diffusivity forms products involving two scalar gradients before
dividing by their squared norm. Finite Float32/Float64 inputs can therefore
produce zero or NaN even when the final answer is representable.

Normalize the velocity gradient by its largest absolute component at each cell.
For diffusivity, independently normalize the scalar gradient. Evaluate the
existing contractions on normalized components, then restore the single linear
velocity scale. Scalar-gradient scale cancels. Keep the calculation in the
model's dtype and the existing fused broadcast, shared by CPU and CUDA.
Zero gradients return zero; nonfinite gradients remain visible as NaN.

Widening Float32 alone would leave Float64 vulnerable and make the CUDA path
more expensive. An absolute epsilon guard would reintroduce dependence on units.
Normalization addresses the reported failures without either change.

Use analytic incompressible gradients to test small/large velocity and scalar
scales, both clipping modes, 2D/3D, and both precisions. Run the same cases on CPU,
JLArrays with scalar indexing disabled, and CUDA when available. Retain the
independent random tensor-contraction tests. Register the LES file in GPU CI.

Correct the manual to apply the divergence of the symmetric SGS stress for
spatially varying eddy viscosity; the closure remains an array-level utility.
