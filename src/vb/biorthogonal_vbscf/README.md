## Biorthogonal VBSCF Module

This directory is the staging area for the first biorthogonal VBSCF prototype.

The implementation is intentionally split into thirteen layers:

1. `biorthogonal_orbital_frame.*`
   Builds the right-orbital / left-dual-orbital frame
   $$
   \widetilde C = C (C^T S C)^{-1},
   $$
   and validates the basic biorthogonality residual
   $$
   \| \widetilde C^T S C - I \|.
   $$

2. `biorthogonal_selected_structure_space.*`
   Builds the fixed selected-structure metric
   $$
   M^{(0)} = T^T T,
   $$
   together with its reusable Cholesky factor and inverse factor.

3. `biorthogonal_projected_structure_problem.*`
   Forms the selected-space projected Hamiltonian
   $$
   H^{\mathrm{bi}} = T^T h^{\mathrm{bi}} T,
   $$
   and its metric-orthogonalized form
   $$
   \widehat H = L^{-1} H^{\mathrm{bi}} L^{-T}.
   $$

4. `biorthogonal_orbital_integrals.*`
   Reuses the existing right/right active-space `HHO/GGO` layer and the dual
   transform
   $$
   X^{-1} = (C^T S C)^{-1}
   $$
   to produce the left/right one-electron matrix
   $$
   h^{LR} = X^{-1} h^{RR},
   $$
   and to evaluate left/right two-electron integrals on demand.

5. `biorthogonal_determinant_hamiltonian.*`
   Evaluates determinant matrix elements
   $$
   h^{\mathrm{bi}}_{JI} = \langle \widetilde D_J | \hat H | D_I \rangle
   $$
   with biorthogonal Slater-Condon rules and assembles determinant Hamiltonian
   matrices over canonical alpha/beta occupied-orbital lists.

6. `biorthogonal_projected_solver.*`
   Solves the fixed-metric left/right selected-space coefficient problem
   $$
   H^{\mathrm{bi}} r_n = E_n M^{(0)} r_n,
   \qquad
   l_n^T H^{\mathrm{bi}} = E_n l_n^T M^{(0)},
   $$
   by diagonalizing the orthogonalized non-Hermitian operator
   $$
   \widehat H = L^{-1} H^{\mathrm{bi}} L^{-T}
   $$
   and mapping the left/right eigenvectors back to the selected-structure basis.

7. `biorthogonal_structure_expansion.*`
   Repackages the existing full-determinant VB expansion into the determinant
   list and selected-structure coefficient map
   $$
   |\Phi_K\rangle = \sum_I T_{I K} |D_I\rangle.
   $$

8. `biorthogonal_structure_hamiltonian_builder.*`
   Reorganizes the determinant expansion into unique alpha/beta spin strings
   and assembles the ordered projected Hamiltonian directly by local
   block-contraction, without materializing the full determinant matrix
   $$
   H^{\mathrm{bi}} = T^T h^{\mathrm{bi}} T.
   $$

9. `biorthogonal_prepared_input.*`
   Centralizes the conversion
   $$
   \texttt{CppVbInput}
   \rightarrow
   (\text{prepared active space}, \text{full determinant topology})
   $$
   so repeated biorthogonal evaluations do not rebuild the active-space layer
   at every selected-structure query.

10. `biorthogonal_forward_evaluator.*`
   Closes the first end-to-end forward chain
   $$
   \texttt{FullDeterminantStructureData}
   \rightarrow (D, T)
   \rightarrow \text{unique-spin block contraction}
   \rightarrow H^{\mathrm{bi}}
   \rightarrow (E_n, l_n, r_n).
   $$

11. `biorthogonal_exact_selected_structure.*`
   Builds the exact selected-subspace projection
   $$
   U_{\mathrm{sel}} = S_{\mathrm{det}}^{(\mathrm{full})} T_{\mathrm{sel}},
   \qquad
   Y_{\mathrm{sel}} = h_{\mathrm{det}}^{(\mathrm{bi},\mathrm{full})} T_{\mathrm{sel}},
   $$
   and then solves
   $$
   H_{\mathrm{str}} C = S_{\mathrm{str}} C E,
   \qquad
   S_{\mathrm{str}} = U_{\mathrm{sel}}^T T_{\mathrm{sel}},
   \qquad
   H_{\mathrm{str}} = U_{\mathrm{sel}}^T Y_{\mathrm{sel}}.
   $$
   This is the preferred API for truncated selected-structure calculations
   because it reproduces the original nonorthogonal VB selected-space pencil
   exactly.

12. `biorthogonal_exact_selected_structure_scf.*`
   Wraps the exact selected-subspace projection in SCF-style bookkeeping:
   selected-state weights, state-averaged electronic energy, total energy, and
   an exported structure-matrix view for higher-level callers.

13. `biorthogonal_exact_selected_structure_gradient.*`
   Reuses the existing nonorthogonal selected-subspace gradient evaluators on
   the matching selected-structure input, and validates that their exported
   physical structure matrices
   $$
   S_{\mathrm{str}},\ H_{\mathrm{str}}
   $$
   agree with the exact selected-subspace forward path before returning the
   exact biorthogonal SCF bookkeeping. This is the current exact selected-space
   gradient API for both the active-space and orbital layers.

The current prototype now reaches the end-to-end forward selected-space
coefficient solve from explicit full-determinant structure data. The current
gradient support is limited to the exact selected-space wrapper described
above; the module still does not implement a dedicated determinant-level
biorthogonal backward, structure-space root tracking, or any second-order
optimizer.
