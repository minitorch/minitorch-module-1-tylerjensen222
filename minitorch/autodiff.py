from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)

    vals[arg] += epsilon
    f_plus = f(*vals)

    vals[arg] -= 2 * epsilon
    f_minus = f(*vals)

    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()  # To keep track of visited nodes
    topological_order = []  # To store the topologically sorted variables

    def visit(var):
        var_id = var.unique_id

        # check if variable was alreaady visited
        if (var_id in visited) or var.is_constant():
            return

        else:
            visited.add(var_id)

            # recursively visit all parent variables
            for parent in var.parents:
                visit(parent)

            # add to top order
            topological_order.append(var)

    # start recursing
    visit(variable)

    # return reversed
    return reversed(topological_order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # get topological sort, init {var: gradient} map
    topo_order = topological_sort(variable)
    var_gradient_map = {variable.unique_id: deriv}

    for back_var in topo_order:
        # get gradient
        d_output = var_gradient_map[back_var.unique_id]

        # if leaf, accumulate
        if back_var.is_leaf():
            back_var.accumulate_derivative(d_output)

        # else, chain rule, update map for parents
        else:
            vars_to_grads = back_var.chain_rule(d_output)
            for parent_var, grad in vars_to_grads:
                if parent_var.unique_id in var_gradient_map:
                    var_gradient_map[parent_var.unique_id] += grad
                else:
                    var_gradient_map[parent_var.unique_id] = grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
