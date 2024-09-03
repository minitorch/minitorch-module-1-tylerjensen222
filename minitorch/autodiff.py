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
        if (var.unique_id in visited) or var.is_constant:
            return
        # If the variable is already visited, we don't process it again
        else:
            visited.add(var_id)

            # Recursively visit all variables that this variable depends on
            for parent in var.parents:
                visit(parent)

            # After visiting all the dependencies, add the variable to the order
            topological_order.append(var)

    # Start the visit from the given variable
    visit(variable)

    # Return the variables in reverse order to get the correct topological order
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
    # Step 1: Compute the topological order of the computation graph
    topo_order = topological_sort(variable)

    # Step 2: Initialize the derivative of the final output with respect to itself
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)

    # Step 3: Backpropagate the derivatives through the graph
    for var in topo_order:
        if var.history is not None and var.history.last_fn is not None:
            # Get the current accumulated derivative
            d_output = var.derivative

            # Use the chain rule to get the local derivatives with respect to inputs
            local_derivatives = var.chain_rule(d_output)

            # Accumulate the derivatives for each input variable
            for input_var, d_input in local_derivatives:
                input_var.accumulate_derivative(d_input)


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
