# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Euclidean projections."""

from typing import Any, Callable, Tuple

import chex
import jax
from jax import flatten_util
import jax.numpy as jnp
import optax.tree


def projection_non_negative(tree: Any) -> Any:
  r"""Projection onto the non-negative orthant.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad
    \textrm{subject to} \quad p \ge 0

  where :math:`x` is the input tree.

  Args:
    tree: tree to project.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  return jax.tree.map(jax.nn.relu, tree)


def projection_box(tree: Any, lower: Any, upper: Any) -> Any:
  r"""Projection onto box constraints.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad \textrm{subject to} \quad
    \text{lower} \le p \le \text{upper}

  where :math:`x` is the input tree.

  Args:
    tree: tree to project.
    lower:  lower bound, a scalar or tree with the same structure as
      ``tree``.
    upper:  upper bound, a scalar or tree with the same structure as
      ``tree``.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  return jax.tree.map(jnp.clip, tree, lower, upper)


def projection_hypercube(tree: Any, scale: Any = 1) -> Any:
  r"""Projection onto the (unit) hypercube.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad \textrm{subject to} \quad
    0 \le p \le \text{scale}

  where :math:`x` is the input tree.

  By default, we project to the unit hypercube (`scale=1`).

  This is a convenience wrapper around
  :func:`projection_box <optax.projections.projection_box>`.

  Args:
    tree: tree to project.
    scale: scale of the hypercube, a scalar or a tree (default: 1).

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  return projection_box(tree, lower=0, upper=scale)


@jax.custom_jvp
def _projection_unit_simplex(values: chex.Array) -> chex.Array:
  """Projection onto the unit simplex."""
  s = 1
  n_features = values.shape[0]
  u = jnp.sort(values)[::-1]
  cumsum_u = jnp.cumsum(u)
  ind = jnp.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  return jax.nn.relu(s / idx + (values - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(
    primals: list[chex.Array], tangents: list[chex.Array]
) -> tuple[chex.Array, chex.Array]:
  (values,) = primals
  (values_dot,) = tangents
  primal_out = _projection_unit_simplex(values)
  supp = primal_out > 0
  card = jnp.count_nonzero(supp)
  tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
  return primal_out, tangent_out


def projection_simplex(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto a simplex.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{p}{\text{argmin}} ~ \|x - p\|_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{scale}

  By default, the projection is onto the probability simplex (unit simplex).

  Args:
    tree: tree to project.
    scale: value the projected tree should sum to (default: 1).

  Returns:
    projected tree, a tree with the same structure as ``tree``.

  Example:

    Here is an example using a tree::

      >>> import jax.numpy as jnp
      >>> from optax import tree, projections
      >>> data = {"w": jnp.array([2.5, 3.2]), "b": 0.5}
      >>> print(tree.sum(data))
      6.2
      >>> new_data = projections.projection_simplex(data)
      >>> print(tree.sum(new_data))
      1.0000002

  .. versionadded:: 0.2.3
  """
  values, unravel_fn = flatten_util.ravel_pytree(tree)
  new_values = scale * _projection_unit_simplex(values / scale)
  return unravel_fn(new_values)


def projection_l1_sphere(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l1 sphere.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_1 = \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the sphere.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  tree_abs = jax.tree.map(jnp.abs, tree)
  tree_sign = jax.tree.map(jnp.sign, tree)
  tree_abs_proj = projection_simplex(tree_abs, scale)
  return optax.tree.mul(tree_sign, tree_abs_proj)


def projection_l1_ball(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l1 ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_1 \le \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the ball.

  Returns:
    projected tree, with the same structure as ``tree``.

  Example:

      >>> import jax.numpy as jnp
      >>> from optax import tree, projections
      >>> data = {"w": jnp.array([2.5, 3.2]), "b": 0.5}
      >>> print(tree.norm(data, ord=1))
      6.2
      >>> new_data = projections.projection_l1_ball(data)
      >>> print(tree.norm(new_data, ord=1))
      1.0000002

  .. versionadded:: 0.2.4
  """
  l1_norm = optax.tree.norm(tree, ord=1)
  return jax.lax.cond(
      l1_norm <= scale,
      lambda tree: tree,
      lambda tree: projection_l1_sphere(tree, scale),
      operand=tree,
  )


def projection_l2_sphere(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l2 sphere.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_2 = \text{value}

  Args:
    tree: tree to project.
    scale: radius of the sphere.

  Returns:
    projected tree, with the same structure as ``tree``.

  .. versionadded:: 0.2.4
  """
  factor = scale / optax.tree.norm(tree)
  return optax.tree.scale(factor, tree)


def projection_l2_ball(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l2 ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_2 \le \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the ball.

  Returns:
    projected tree, with the same structure as ``tree``.

  .. versionadded:: 0.2.4
  """
  squared_norm = optax.tree.norm(tree, squared=True)
  positive = squared_norm > 0
  valid_squared_norm = jnp.where(positive, squared_norm, 1.)
  norm = jnp.where(positive, jnp.sqrt(valid_squared_norm), 0.)
  factor = scale / jnp.maximum(norm, scale)
  return optax.tree.scale(factor, tree)


def projection_linf_ball(tree: Any, scale: chex.Numeric = 1) -> Any:
  r"""Projection onto the l-infinity ball.

  This function solves the following constrained optimization problem,
  where ``x`` is the input tree.

  .. math::

    \underset{y}{\text{argmin}} ~ \|x - y\|_2^2 \quad \textrm{subject to} \quad
    \|y\|_{\infty} \le \text{scale}

  Args:
    tree: tree to project.
    scale: radius of the ball.

  Returns:
    projected tree, with the same structure as ``tree``.
  """
  lower = optax.tree.full_like(tree, -scale)
  upper = optax.tree.full_like(tree, scale)
  return projection_box(tree, lower=lower, upper=upper)


def _max_l2(x, marginal_b, gamma):
  scale = gamma * marginal_b
  x_scale = x / scale
  p = _projection_unit_simplex(x_scale)
  # From Danskin's theorem, we do not need to backpropagate
  # through projection_simplex.
  p = jax.lax.stop_gradient(p)
  return jnp.dot(x, p) - 0.5 * scale * jnp.dot(p, p)


def _max_ent(x, marginal_b, gamma):
  return gamma * jax.nn.logsumexp(x / gamma) - gamma * jnp.log(marginal_b)


_max_l2_vmap = jax.vmap(_max_l2, in_axes=(1, 0, None))
_max_l2_grad_vmap = jax.vmap(jax.grad(_max_l2), in_axes=(1, 0, None))

_max_ent_vmap = jax.vmap(_max_ent, in_axes=(1, 0, None))
_max_ent_grad_vmap = jax.vmap(jax.grad(_max_ent), in_axes=(1, 0, None))


def _delta_l2(x, gamma=1.0):
  # Solution to Eqn. (6) in https://arxiv.org/abs/1710.06276 with squared l2
  # regularization (see Table 1 in the paper).
  z = (0.5 / gamma) * jnp.dot(jax.nn.relu(x), jax.nn.relu(x))
  return z


def _delta_ent(x, gamma):
  # Solution to Eqn. (6) in https://arxiv.org/abs/1710.06276 with negative
  # entropy regularization.
  return gamma * jnp.exp((x / gamma) - 1).sum()

_delta_l2_vmap = jax.vmap(_delta_l2, in_axes=(1, None))
_delta_l2_grad_vmap = jax.vmap(jax.grad(_delta_l2), in_axes=(1, None))

_delta_ent_vmap = jax.vmap(_delta_ent, in_axes=(1, None))
_delta_ent_grad_vmap = jax.vmap(jax.grad(_delta_ent), in_axes=(1, None))


def _make_semi_dual(max_vmap, gamma=1.0):
  # Semi-dual objective, see equation (10) in
  # https://arxiv.org/abs/1710.06276
  def fun(alpha, cost_matrix, marginals_a, marginals_b):
    X = alpha[:, jnp.newaxis] - cost_matrix
    ret = jnp.dot(marginals_b, max_vmap(X, marginals_b, gamma))
    ret -= jnp.dot(alpha, marginals_a)
    return ret
  return fun


def _make_dual(delta_vmap, gamma):
  r"""Make the objective function of dual variables.

  Args:
    delta_vmap: The smoothed version of delta function, acting on each column of
      its matrix-valued input.
    gamma: A regularization constant.
  Returns:
    A cost function of dual variables. Cf. Equation (7) in
      https://arxiv.org/abs/1710.06276
  """

  def fun(alpha_beta, cost_matrix, marginals_a, marginals_b):
    alpha, beta = alpha_beta
    alpha_column = alpha[:, jnp.newaxis]
    beta_row = beta[jnp.newaxis, :]
    # Make a dual constraint matrix, whose (i,j)-th entry is
    # alpha[i] + beta[j] - c[i,j]. JAXopt solvers minimize functions hence
    # the sign is the opposite of Eqn (7)"
    dual_constraint_matrix = alpha_column + beta_row - cost_matrix
    delta_dual_constraints = delta_vmap(dual_constraint_matrix, gamma)
    dual_loss = delta_dual_constraints.sum() - jnp.dot(
        alpha, marginals_a) - jnp.dot(beta, marginals_b)
    return dual_loss
  return fun


def _regularized_transport_semi_dual(cost_matrix,
                           marginals_a,
                           marginals_b,
                           make_solver,
                           max_vmap,
                           max_grad_vmap,
                           gamma=1.0):

  r"""Regularized transport in the semi-dual formulation.

  Args:
    cost_matrix: The cost matrix of size (m, n).
    marginals_a: The marginals of size (m,)
    marginals_b: The marginals of size (n,)
    make_solver: A function that makes the optimization algorithm
    max_vmap:  A function that computes the regularized max on columns of
      its matrix-valued input
    max_grad_vmap:  A function that computes gradient of regularized max
      on columns of its matrix-valued input
    gamma: A parameter that controls the strength of regularization.

  Returns:
    The optimized plan. See the text under Eqn. (10) of
      https://arxiv.org/abs/1710.06276
  """
  size_a, size_b = cost_matrix.shape

  if len(marginals_a.shape) >= 2:
    raise ValueError("marginals_a should be a vector.")

  if len(marginals_b.shape) >= 2:
    raise ValueError("marginals_b should be a vector.")

  if size_a != marginals_a.shape[0] or size_b != marginals_b.shape[0]:
    raise ValueError("cost_matrix and marginals must have matching shapes.")

  if make_solver is None:
    # Default solver - this would need to be replaced with actual LBFGS implementation
    raise NotImplementedError("Default LBFGS solver not implemented. Please provide make_solver.")

  semi_dual = _make_semi_dual(max_vmap, gamma=gamma)
  solver = make_solver(semi_dual)
  alpha_init = jnp.zeros(size_a)

  # Optimal dual potentials.
  alpha = solver.run(alpha_init,
                     cost_matrix=cost_matrix,
                     marginals_a=marginals_a,
                     marginals_b=marginals_b).params

  # Optimal primal transportation plan.
  X = alpha[:, jnp.newaxis] - cost_matrix
  P = max_grad_vmap(X, marginals_b, gamma).T * marginals_b

  return P


def _regularized_transport_dual(cost_matrix,
                                marginals_a,
                                marginals_b,
                                make_solver,
                                delta_vmap,
                                delta_grad_vmap,
                                gamma=1.0):
  r"""Regularized transport in the dual formulation.

  Args:
    cost_matrix: The cost matrix of size (m, n).
    marginals_a: The marginals of size (m,)
    marginals_b: The marginals of size (n,)
    make_solver: A function that makes the optimization algorithm
    delta_vmap:  A function that computes the regularized delta on columns of
      its matrix-valued input
    delta_grad_vmap:  A function that computes gradient of regularized delta
      on columns of its matrix-valued input
    gamma: A parameter that controls the strength of regularization.

  Returns:
    The optimized plan. See the text under Eqn. (7) of
      https://arxiv.org/abs/1710.06276

  """

  size_a, size_b = cost_matrix.shape

  if len(marginals_a.shape) >= 2:
    raise ValueError("marginals_a should be a vector.")

  if len(marginals_b.shape) >= 2:
    raise ValueError("marginals_b should be a vector.")

  if size_a != marginals_a.shape[0] or size_b != marginals_b.shape[0]:
    raise ValueError("cost_matrix and marginals must have matching shapes.")

  if make_solver is None:
    # Default solver - this would need to be replaced with actual LBFGS implementation
    raise NotImplementedError("Default LBFGS solver not implemented. Please provide make_solver.")

  dual = _make_dual(delta_vmap, gamma=gamma)
  solver = make_solver(dual)
  alpha_beta_init = (jnp.zeros(size_a), jnp.zeros(size_b))

  # Optimal dual potentials.
  alpha_beta = solver.run(init_params=alpha_beta_init,
                          cost_matrix=cost_matrix,
                          marginals_a=marginals_a,
                          marginals_b=marginals_b).params

  # Optimal primal transportation plan.
  alpha, beta = alpha_beta
  alpha_column = alpha[:, jnp.newaxis]
  beta_row = beta[jnp.newaxis, :]
  # The (i,j)-th entry of dual_constraint_matrix is alpha[i] + beta[j] - c[i,j].
  dual_constraint_matrix = alpha_column + beta_row - cost_matrix
  plan = delta_grad_vmap(dual_constraint_matrix, gamma).T
  return plan


def projection_transport(sim_matrix: jnp.ndarray,
                         marginals: Tuple,
                         make_solver: Callable = None,
                         use_semi_dual: bool = True):
  r"""Projection onto the transportation polytope.

  We solve

  .. math::

    \underset{P \ge 0}{\text{argmin}} ~ ||P - S||_2^2 \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  or equivalently

  .. math::

    \underset{P \ge 0}{\text{argmin}} ~ \langle P, C \rangle
    + \frac{1}{2} \|P\|_2^2 \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  where :math:`S` is a similarity matrix, :math:`C` is a cost matrix
  and :math:`S = -C`.

  This implementation solves the semi-dual (see equation 10 in reference below)
  using LBFGS but the solver can be overidden using the ``make_solver`` option.

  For an entropy-regularized version, see
  :func:`kl_projection_transport <optax.projections.kl_projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size_a, size_b).
    marginals: a tuple (marginals_a, marginals_b),
      where marginals_a has shape=(size_a,) and
      marginals_b has shape=(size_b,).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).

  Returns:
    plan: transportation matrix, shape=(size_a, size_b).
  References:
    Smooth and Sparse Optimal Transport.
    Mathieu Blondel, Vivien Seguy, Antoine Rolet.
    In Proceedings of Artificial Intelligence and Statistics (AISTATS), 2018.
    https://arxiv.org/abs/1710.06276
  """
  marginals_a, marginals_b = marginals

  if use_semi_dual:
    plan = _regularized_transport_semi_dual(cost_matrix=-sim_matrix,
                                  marginals_a=marginals_a,
                                  marginals_b=marginals_b,
                                  make_solver=make_solver,
                                  max_vmap=_max_l2_vmap,
                                  max_grad_vmap=_max_l2_grad_vmap)
  else:
    plan = _regularized_transport_dual(cost_matrix=-sim_matrix,
                                       marginals_a=marginals_a,
                                       marginals_b=marginals_b,
                                       make_solver=make_solver,
                                       delta_vmap=_delta_l2_vmap,
                                       delta_grad_vmap=_delta_l2_grad_vmap)
  return plan


def kl_projection_transport(sim_matrix: jnp.ndarray,
                            marginals: Tuple,
                            make_solver: Callable = None,
                            use_semi_dual: bool = True):

  r"""Kullback-Leibler projection onto the transportation polytope.

  We solve

  .. math::
    \underset{P > 0}{\text{argmin}} ~ \text{KL}(P, \exp(S)) \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  or equivalently

  .. math::

    \underset{P > 0}{\text{argmin}} ~ \langle P, C \rangle
    + \langle P, \log P \rangle \quad
    \textrm{subject to} \quad P^\top \mathbf{1} = a, P \mathbf{1} = b

  where :math:`S` is a similarity matrix, :math:`C` is a cost matrix
  and :math:`S = -C`.

  This implementation solves the semi-dual (see equation 10 in reference below)
  using LBFGS but the solver can be overidden using the ``make_solver`` option.

  For an l2-regularized version, see
  :func:`projection_transport <optax.projections.projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size_a, size_b).
    marginals: a tuple (marginals_a, marginals_b),
      where marginals_a has shape=(size_a,) and
      marginals_b has shape=(size_b,).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).
  Returns:
    plan: transportation matrix, shape=(size_a, size_b).
  References:
    Smooth and Sparse Optimal Transport.
    Mathieu Blondel, Vivien Seguy, Antoine Rolet.
    In Proceedings of Artificial Intelligence and Statistics (AISTATS), 2018.
    https://arxiv.org/abs/1710.06276
  """
  marginals_a, marginals_b = marginals

  if use_semi_dual:
    plan = _regularized_transport_semi_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        max_vmap=_max_ent_vmap,
        max_grad_vmap=_max_ent_grad_vmap)
  else:
    plan = _regularized_transport_dual(
        cost_matrix=-sim_matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
        make_solver=make_solver,
        delta_vmap=_delta_ent_vmap,
        delta_grad_vmap=_delta_ent_grad_vmap)
  return plan


def projection_birkhoff(sim_matrix: jnp.ndarray,
                        make_solver: Callable = None,
                        use_semi_dual: bool = True):

  r"""Projection onto the Birkhoff polytope, the set of doubly stochastic
  matrices.

  This function is a special case of
  :func:`projection_transport <optax.projections.projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size, size).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).
  Returns:
    P: doubly-stochastic matrix, shape=(size, size).
  """
  marginals_a = jnp.ones(sim_matrix.shape[0])
  marginals_b = jnp.ones(sim_matrix.shape[1])
  return projection_transport(sim_matrix=sim_matrix,
                              marginals=(marginals_a, marginals_b),
                              make_solver=make_solver,
                              use_semi_dual=use_semi_dual)


def kl_projection_birkhoff(sim_matrix: jnp.ndarray,
                           make_solver: Callable = None,
                           use_semi_dual: bool = True):

  r"""Kullback-Leibler projection onto the Birkhoff polytope,
  the set of doubly stochastic matrices.

  This function is a special case of
  :func:`kl_projection_transport <optax.projections.kl_projection_transport>`.

  Args:
    sim_matrix: similarity matrix, shape=(size, size).
    make_solver: a function of the form make_solver(fun),
      for creating an iterative solver to minimize fun.
    use_semi_dual: if true, use the semi-dual formulation in
      Equation (10) of https://arxiv.org/abs/1710.06276. Otherwise, use
      the dual-formulation in Equation (7).
  Returns:
    P: doubly-stochastic matrix, shape=(size, size).
  """
  marginals_a = jnp.ones(sim_matrix.shape[0])
  marginals_b = jnp.ones(sim_matrix.shape[1])
  return kl_projection_transport(sim_matrix=sim_matrix,
                                 marginals=(marginals_a, marginals_b),
                                 make_solver=make_solver,
                                 use_semi_dual=use_semi_dual)
