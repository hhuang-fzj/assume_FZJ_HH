import pyomo.environ as pyo

def print_result(model):
    print(f"alpha    = {pyo.value(model.alpha):.4f}")
    print(f"s        = {pyo.value(model.s):.4f} MWh")
    print(f"a_dach   = {pyo.value(model.a_dach):.4f} MWh")
    print(f"LEM成本  = {pyo.value(model.LEM_dach):.4f} EUR")
    print(f"WM成本   = {pyo.value(model.WM_dach):.4f} EUR")
    print(f"偏差成本 = {pyo.value(model.reBAP_dach):.4f} EUR")
    print(f"总成本   = {pyo.value(model.obj):.4f} EUR")


# 1. 建立模型
m = pyo.ConcreteModel()


# 2. 五个预测输入
# 电量：MWh；价格：EUR/MWh
m.L_dach = pyo.Param(initialize=-0.020, mutable=True)
m.p_dach_LEM = pyo.Param(initialize=123.1, mutable=True)
m.p_dach_WM = pyo.Param(initialize=30.0, mutable=True)
m.r_dach = pyo.Param(initialize=20.0, mutable=True)
m.S_dach = pyo.Param(initialize=0.008, mutable=True)

L = pyo.value(m.L_dach)

sigma = 1 if L > 0 else -1
L_abs = abs(L)
m.sigma = pyo.Param(initialize=sigma, mutable=True)
m.L_abs = pyo.Param(initialize=L_abs, mutable=True)


# 3. 两个决策变量
m.alpha = pyo.Var(bounds=(0, 1))

m.s = pyo.Var(domain=pyo.NonNegativeReals)
m.a_dach = pyo.Expression(expr=m.sigma * m.s) 



# 4. 两个下界约束

# 卖出的电量不超过 LEM 报量


# 卖出的电量不超过预测的 LEM 接纳上限


m.LEM_bid_limit = pyo.Constraint(
    expr=m.s <= m.alpha * m.L_abs
)

m.LEM_capacity_limit = pyo.Constraint(
    expr=m.s <= m.S_dach
)

# 5. 三项预测成本，单位都是 EUR

# ① LEM 成本
m.LEM_dach = pyo.Expression(
    expr=m.p_dach_LEM * m.a_dach
)

# ② WM 成本：本例假定 WM 报量全部成交
m.WM_dach = pyo.Expression(
    expr=m.p_dach_WM * (1 - m.alpha) * m.L_dach
)

# ③ 预测偏差成本
# 预测偏差 = 预测净电量 - 预测 LEM 成交量 - 预测 WM 成交量
m.reBAP_dach = pyo.Expression(
    expr=m.r_dach * (
        m.L_dach - m.a_dach - (1 - m.alpha) * m.L_dach
    )
)


# 6. 目标函数：最小化三项成本之和
m.obj = pyo.Objective(
    expr=m.LEM_dach + m.WM_dach + m.reBAP_dach,
    sense=pyo.minimize,
)

# big-M 参数
M = max(
    pyo.value(m.L_abs),
    pyo.value(m.S_dach),
)
m.big_M = pyo.Param(initialize=M, mutable=True)

# 二进制变量：注意是 z，sigma 仍然是方向参数
m.z = pyo.Var(domain=pyo.Binary)

# 两条新增下界，与原有上界一起强制实现 min()
m.LEM_bid_lower = pyo.Constraint(
    expr=m.s >= m.alpha * m.L_abs - m.big_M * m.z
)

m.LEM_capacity_lower = pyo.Constraint(
    expr=m.s >= m.S_dach - m.big_M * (1 - m.z)
)


# 7. 建立求解器
opt = pyo.SolverFactory("appsi_highs")

if not opt.available(exception_flag=False):
    raise RuntimeError(
        "HiGHS 不可用。请在当前 Python 环境运行："
        "python -m pip install pyomo highspy"
    )


# 每行依次是：编号、L、p_LEM、p_WM、r、S
cases = [
    (1, -0.020, 123.1, 30.0,  20.0, 0.008),
    (2, -0.020, 123.1, 30.0,  40.0, 0.008),
    (3, -0.020, 123.1, 30.0, 130.0, 0.008),
    (4, -0.020, 123.1, 30.0,  20.0, 0.000),
    (5, -0.020, 123.1, 30.0,  20.0, 0.050),
    (6,  0.020, 150.0, 30.0,  40.0, 0.008),
    (7,  0.020,  25.0, 30.0,  40.0, 0.008),
]

for number, L_dach, p_dach_LEM, p_dach_WM, r_dach, S_dach in cases:
    # 更新五个原始输入
    m.L_dach.set_value(L_dach)
    m.p_dach_LEM.set_value(p_dach_LEM)
    m.p_dach_WM.set_value(p_dach_WM)
    m.r_dach.set_value(r_dach)
    m.S_dach.set_value(S_dach)

    # 根据本组输入，更新方向和绝对值
    m.sigma.set_value(1 if L_dach > 0 else -1)
    m.L_abs.set_value(abs(L_dach))
    m.big_M.set_value(max(abs(L_dach), S_dach))

    # 求解
    res = opt.solve(m)

    print(f"\n--- 情况 {number} ---")
    print("termination condition:", res.solver.termination_condition)

    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"情况 {number} 未达到最优。")

    print_result(m)

    # 检查求出的 s 是否符合规定的成交关系
    required_s = min(
        pyo.value(m.alpha) * pyo.value(m.L_abs),
        pyo.value(m.S_dach),
    )

    print(f"按 min() 应成交 = {required_s:.4f} MWh")