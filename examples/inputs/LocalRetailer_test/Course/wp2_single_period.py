import pyomo.environ as pyo


# 1. 建立模型
m = pyo.ConcreteModel()


# 2. 五个预测输入
# 电量：MWh；价格：EUR/MWh
m.L_dach = pyo.Param(initialize=-0.020, mutable=True)
m.p_dach_LEM = pyo.Param(initialize=123.1, mutable=True)
m.p_dach_WM = pyo.Param(initialize=30.0, mutable=True)
m.r_dach = pyo.Param(initialize=20.0, mutable=True)
m.S_dach = pyo.Param(initialize=0.008, mutable=True)


# 3. 两个决策变量
m.alpha = pyo.Var(bounds=(0, 1))
m.a_dach = pyo.Var(bounds=(None, 0))


# 4. 两个下界约束

# 卖出的电量不超过 LEM 报量
m.LEM_bid_limit = pyo.Constraint(
    expr=m.a_dach >= m.alpha * m.L_dach
)

# 卖出的电量不超过预测的 LEM 接纳上限
m.LEM_capacity_limit = pyo.Constraint(
    expr=m.a_dach >= -m.S_dach
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


# 7. 建立求解器
opt = pyo.SolverFactory("appsi_highs")

if not opt.available(exception_flag=False):
    raise RuntimeError(
        "HiGHS 不可用。请在当前 Python 环境运行："
        "python -m pip install pyomo highspy"
    )


# 8. 第一次求解：r_dach = 20
res = opt.solve(m)

print("\n--- 第一次求解：r_dach = 20 EUR/MWh ---")
print("termination condition:", res.solver.termination_condition)

if res.solver.termination_condition != pyo.TerminationCondition.optimal:
    raise RuntimeError("第一次求解未达到最优，请检查求解状态。")

print(f"alpha     = {pyo.value(m.alpha):.4f}")
print(f"a_dach    = {pyo.value(m.a_dach):.4f} MWh")
print(f"LEM 成本  = {pyo.value(m.LEM_dach):.4f} EUR")
print(f"WM 成本   = {pyo.value(m.WM_dach):.4f} EUR")
print(f"偏差成本  = {pyo.value(m.reBAP_dach):.4f} EUR")
print(f"总成本    = {pyo.value(m.obj):.4f} EUR")


# 9. 修改参数，然后再次求解
m.r_dach.set_value(40.0)

res = opt.solve(m)

print("\n--- 第二次求解：r_dach = 40 EUR/MWh ---")
print("termination condition:", res.solver.termination_condition)

if res.solver.termination_condition != pyo.TerminationCondition.optimal:
    raise RuntimeError("第二次求解未达到最优，请检查求解状态。")

print(f"alpha     = {pyo.value(m.alpha):.4f}")
print(f"a_dach    = {pyo.value(m.a_dach):.4f} MWh")
print(f"LEM 成本  = {pyo.value(m.LEM_dach):.4f} EUR")
print(f"WM 成本   = {pyo.value(m.WM_dach):.4f} EUR")
print(f"偏差成本  = {pyo.value(m.reBAP_dach):.4f} EUR")
print(f"总成本    = {pyo.value(m.obj):.4f} EUR")