"""
Alpha model: split the forecast residual load of every time step between LEM and WM.

Written by Yihan Shu (course WP2-WP4, file course/wp4_multi_period.py); moved here
unchanged except for this header, so that the ASSUME strategy can import it.
Pure Pyomo, does not import ASSUME.

Model (one MILP, separable per time step t):
    min  sum_t  p_lem[t] * a_hat[t] + p_wm[t] * (1 - alpha[t]) * L_hat[t]
               + r_hat[t] * (L_hat[t] - a_hat[t] - (1 - alpha[t]) * L_hat[t])
    a_hat[t] = sigma[t] * s[t],  s[t] = min(alpha[t] * |L_hat[t]|, S_hat[t])
    (min() via binary z[t] and big-M = max(|L_hat|, S_hat))

Input  df: index = time steps; columns L_hat [MWh, + buy / - sell], p_lem, p_wm,
           r_hat [EUR/MWh], S_hat [MWh, >= 0]. Only values known when deciding.
Output df: input + alpha, a_hat [MWh], cost_lem, cost_wm, cost_rebap, cost_total [EUR].
Rows with |L_hat| < 1e-9: no bid, alpha = 0 [OWN rule].

Sources (full citations: reference list in the course document)
    Expected imbalance cost as a term of the day-ahead objective: [Iria2019] Sect. 3.1,
    Eq. (1). Accepted volume depends on the market outcome: [Ottesen2018] p. 124,
    Eq. (4)(11). Upper bound min(alpha |L_hat|, S_hat) and its MILP form: [OWN].
"""

import pyomo.environ as pyo
import pandas as pd

def solve_alpha(df):
    result = df.copy()

    # 先把所有行的六个结果列设为零
    output_columns = [
        "alpha",
        "a_hat",
        "cost_lem",
        "cost_wm",
        "cost_rebap",
        "cost_total",
    ]
    result[output_columns] = 0.0

    # 只优化非零电量的行
    active = result.loc[
        result["L_hat"].abs() >= 1e-9
    ].copy()

    # 如果没有需要优化的行，直接返回零结果
    if active.empty:
        return result

        # 计算每个时段的派生输入
    active["L_abs"] = active["L_hat"].abs()

    active["sigma"] = 1
    active.loc[active["L_hat"] < 0, "sigma"] = -1

    active["big_M"] = active[["L_abs", "S_hat"]].max(axis=1)

# 1. 建立模型
    m = pyo.ConcreteModel()

    m.T = pyo.Set(initialize=list(active.index))

    # 2. 五个带时间索引的预测输入
    m.L_dach = pyo.Param(
        m.T, initialize=active["L_hat"].to_dict()
    )
    m.p_dach_LEM = pyo.Param(
        m.T, initialize=active["p_lem"].to_dict()
    )
    m.p_dach_WM = pyo.Param(
        m.T, initialize=active["p_wm"].to_dict()
    )
    m.r_dach = pyo.Param(
        m.T, initialize=active["r_hat"].to_dict()
    )
    m.S_dach = pyo.Param(
        m.T, initialize=active["S_hat"].to_dict()
    )

    m.sigma = pyo.Param(
        m.T, initialize=active["sigma"].to_dict()
    )

    m.L_abs = pyo.Param(
        m.T, initialize=active["L_abs"].to_dict()
    )

    m.big_M = pyo.Param(
        m.T, initialize=active["big_M"].to_dict()
    )    


    # 3. 两个决策变量
    m.alpha = pyo.Var(m.T, bounds=(0, 1))

    m.s = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    m.z = pyo.Var(m.T, domain=pyo.Binary)

    # 每个时段的带符号成交量
    def a_dach_rule(model, t):
        return model.sigma[t] * model.s[t]

    m.a_dach = pyo.Expression(
        m.T,
        rule=a_dach_rule,
    )    
    # 4. 两条上界约束：买卖方向通用

    # 成交量绝对值不超过报量绝对值
    def lem_bid_limit_rule(model, t):
        return model.s[t] <= model.alpha[t] * model.L_abs[t]

    m.LEM_bid_limit = pyo.Constraint(
        m.T,
        rule=lem_bid_limit_rule,
    )

    # 成交量绝对值不超过预测接纳上限
    def lem_capacity_limit_rule(model, t):
        return model.s[t] <= model.S_dach[t]

    m.LEM_capacity_limit = pyo.Constraint(
        m.T,
        rule=lem_capacity_limit_rule,
    )



    # 5. 每个时段的三项预测成本，单位 EUR

    # ① LEM 成本
    def lem_cost_rule(model, t):
        return model.p_dach_LEM[t] * model.a_dach[t]

    m.LEM_dach = pyo.Expression(
        m.T,
        rule=lem_cost_rule,
    )

    # ② WM 成本
    def wm_cost_rule(model, t):
        return (
            model.p_dach_WM[t]
            * (1 - model.alpha[t])
            * model.L_dach[t]
        )

    m.WM_dach = pyo.Expression(
        m.T,
        rule=wm_cost_rule,
    )

    # ③ 预测偏差成本
    def rebap_cost_rule(model, t):
        return model.r_dach[t] * (
            model.L_dach[t]
            - model.a_dach[t]
            - (1 - model.alpha[t]) * model.L_dach[t]
        )

    m.reBAP_dach = pyo.Expression(
        m.T,
        rule=rebap_cost_rule,
    )
    # 6. 目标函数：最小化所有时段的总成本
    m.obj = pyo.Objective(
        expr=pyo.quicksum(
            m.LEM_dach[t]
            + m.WM_dach[t]
            + m.reBAP_dach[t]
            for t in m.T
        ),
        sense=pyo.minimize,
    )
    # 两条 big-M 下界约束

    def lem_bid_lower_rule(model, t):
        return model.s[t] >= (
            model.alpha[t] * model.L_abs[t]
            - model.big_M[t] * model.z[t]
        )

    m.LEM_bid_lower = pyo.Constraint(
        m.T,
        rule=lem_bid_lower_rule,
    )

    def lem_capacity_lower_rule(model, t):
        return model.s[t] >= (
            model.S_dach[t]
            - model.big_M[t] * (1 - model.z[t])
        )

    m.LEM_capacity_lower = pyo.Constraint(
        m.T,
        rule=lem_capacity_lower_rule,
    )

    # 7. 建立求解器并求解一次
    opt = pyo.SolverFactory("appsi_highs")

    if not opt.available(exception_flag=False):
        raise RuntimeError(
            "HiGHS 不可用。请在当前 Python 环境运行："
            "python -m pip install pyomo highspy"
        )

    res = opt.solve(m)

    # 确认达到最优，再读取结果
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"求解未达到最优：{res.solver.termination_condition}"
        )
        # 8. 把每个时段的求解结果写回表格
    for t in m.T:
        result.at[t, "alpha"] = pyo.value(m.alpha[t])
        result.at[t, "a_hat"] = pyo.value(m.a_dach[t])

        result.at[t, "cost_lem"] = pyo.value(m.LEM_dach[t])
        result.at[t, "cost_wm"] = pyo.value(m.WM_dach[t])
        result.at[t, "cost_rebap"] = pyo.value(m.reBAP_dach[t])

        result.at[t, "cost_total"] = pyo.value(
            m.LEM_dach[t]
            + m.WM_dach[t]
            + m.reBAP_dach[t]
        )

    return result
