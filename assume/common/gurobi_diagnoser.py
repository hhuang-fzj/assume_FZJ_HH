from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


def optimize_and_diagnose(
    model: gp.Model,
    output_directory: str = "gurobi_diagnostics",
    file_prefix: str = "infeasible_model",
    compute_iis: bool = True,
    retry_inf_or_unbd: bool = True,
) -> int:
    """
    Optimize a Gurobi model and automatically diagnose infeasibility.

    Parameters
    ----------
    model:
        Fully constructed Gurobi model.
    output_directory:
        Directory in which diagnostic LP and ILP files are written.
    file_prefix:
        Base name for exported diagnostic files.
    compute_iis:
        Whether to compute and print an IIS if the model is infeasible.
    retry_inf_or_unbd:
        If the status is INF_OR_UNBD, rerun with DualReductions=0
        to distinguish infeasibility from unboundedness.

    Returns
    -------
    int
        Final Gurobi model status.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Starting Gurobi optimization")
    print("=" * 70)

    model.optimize()

    # Distinguish infeasible from unbounded if presolve returned INF_OR_UNBD.
    if model.Status == GRB.INF_OR_UNBD and retry_inf_or_unbd:
        print("\nStatus is INF_OR_UNBD.")
        print("Rerunning with DualReductions = 0...")

        model.Params.DualReductions = 0
        model.reset()
        model.optimize()

    status_names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.WORK_LIMIT: "WORK_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }

    status_name = status_names.get(
        model.Status,
        f"UNKNOWN_STATUS_{model.Status}",
    )

    print("\n" + "=" * 70)
    print("Optimization result")
    print("=" * 70)
    print(f"Status:         {status_name} ({model.Status})")
    print(f"Solution count: {model.SolCount}")
    print(f"Runtime:        {model.Runtime:.4f} seconds")
    print(f"Variables:      {model.NumVars}")
    print(f"Constraints:    {model.NumConstrs}")
    print(f"Quadratic cons: {model.NumQConstrs}")
    print(f"General cons:   {model.NumGenConstrs}")

    if model.SolCount > 0:
        print(f"Objective:      {model.ObjVal}")

    if model.Status == GRB.OPTIMAL:
        print("\nThe model is feasible and solved to optimality.")
        return model.Status

    if model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            print(
                "\nThe time limit was reached, but a feasible "
                "incumbent solution exists."
            )
        else:
            print(
                "\nThe time limit was reached without finding a "
                "feasible solution. This does not prove infeasibility."
            )

        return model.Status

    if model.Status == GRB.UNBOUNDED:
        print(
            "\nThe model is unbounded. This is not the same as "
            "infeasibility."
        )
        return model.Status

    if model.Status != GRB.INFEASIBLE:
        print(
            "\nNo infeasibility diagnostics were generated because "
            "the final status is not INFEASIBLE."
        )
        return model.Status

    # Infeasibility-specific output
    print("\n" + "=" * 70)
    print("Model is infeasible")
    print("=" * 70)

    print("\nModel statistics:")
    model.printStats()

    lp_file = output_path / f"{file_prefix}.lp"
    model.write(str(lp_file))
    print(f"\nFull model written to:\n{lp_file.resolve()}")

    print_impossible_bounds(model)

    if compute_iis:
        print("\nComputing IIS...")
        model.computeIIS()

        ilp_file = output_path / f"{file_prefix}.ilp"
        model.write(str(ilp_file))

        print(f"IIS written to:\n{ilp_file.resolve()}")

        print_iis(model)

    return model.Status


def print_impossible_bounds(model: gp.Model) -> None:
    """
    Print variables with directly contradictory or suspicious bounds.
    """

    contradictions = []
    suspicious_binary_bounds = []

    for variable in model.getVars():
        if variable.LB > variable.UB:
            contradictions.append(variable)

        if variable.VType == GRB.BINARY:
            if variable.LB > 1 or variable.UB < 0:
                suspicious_binary_bounds.append(variable)

    print("\nDirect bound check:")

    if not contradictions:
        print("  No variables with LB > UB were found.")
    else:
        for variable in contradictions:
            print(
                f"  CONTRADICTION: {variable.VarName}: "
                f"LB={variable.LB}, UB={variable.UB}"
            )

    if suspicious_binary_bounds:
        print("\nSuspicious binary bounds:")
        for variable in suspicious_binary_bounds:
            print(
                f"  {variable.VarName}: "
                f"LB={variable.LB}, UB={variable.UB}"
            )


def print_iis(model: gp.Model) -> None:
    """
    Print all linear IIS constraints and IIS variable bounds.
    """

    print("\n" + "=" * 70)
    print("IIS constraints")
    print("=" * 70)

    iis_constraint_count = 0

    for constraint in model.getConstrs():
        if not constraint.IISConstr:
            continue

        iis_constraint_count += 1
        print_linear_constraint(model, constraint)

    if iis_constraint_count == 0:
        print("No linear constraints were marked as part of the IIS.")

    print("\n" + "=" * 70)
    print("IIS variable bounds")
    print("=" * 70)

    iis_bound_count = 0

    for variable in model.getVars():
        if variable.IISLB:
            iis_bound_count += 1
            print(
                f"Lower bound: {variable.VarName} >= {variable.LB}"
            )

        if variable.IISUB:
            iis_bound_count += 1
            print(
                f"Upper bound: {variable.VarName} <= {variable.UB}"
            )

    if iis_bound_count == 0:
        print("No variable bounds were marked as part of the IIS.")


def print_linear_constraint(
    model: gp.Model,
    constraint: gp.Constr,
) -> None:
    """
    Print one linear constraint in readable algebraic form.
    """

    row = model.getRow(constraint)
    terms = []

    for index in range(row.size()):
        coefficient = row.getCoeff(index)
        variable = row.getVar(index)

        terms.append(
            f"{coefficient:+g} * {variable.VarName}"
            f"[LB={variable.LB:g}, UB={variable.UB:g}, "
            f"type={variable.VType}]"
        )

    expression = " ".join(terms) if terms else "0"

    print(
        f"\n{constraint.ConstrName}:\n"
        f"  {expression}\n"
        f"  {constraint.Sense} {constraint.RHS:g}"
    )