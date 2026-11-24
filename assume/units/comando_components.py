# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from comando.core import Component, BINARY, INTEGER
from assume.common.utils import create_pwlm

import comando
import numpy as np


class IESComponent(Component):
    """Component of an industrial energy system.

    derived from comando/examples/IES/IES_components.py

    Arguments
    ---------
    - label : str
        Unique sting that serves as an identifier of this Demand.
    - nom_ref: The nominal reference size of the component
    - c_ref: coefficient of investment costs
    - c_m: factor of investment costs corresponding to maintenance costs
    - M: (1) exponent of investment costs
    - nom_min: (0) lower bound for nominal size
    - nom_max: (None) upper bound for nominal size
    - min_part_load: (0) minimum allowed value for relative output
    - base_eff: the base efficiency of the component
    - fit_params: dict of power, coefficient items for polynomial
        approximation of efficiency
    - in_name: name of input commodity
    - out_name: name of output commodity
    - in_connector_name: name of input connector
    - out_connector_name: name of output connector
    - exists: bool
        Specifies whether the Component is currently installed; if set to
        True, its nominal size will be a user-specified parameter. If set
        to False, the nominal size of the component is a design variable.
    - optional: bool
        Specifies whether the Component may be added or removed.
        If exists is True, setting optional to True allows the Component to
        be sold for a user-specified price.
        If exists is False, setting optional to False requires the
        component to be installed with a minimal nominal size of at least
        `nom_min`, while setting it to True adds the possibility of not
        installing it.
    - use_pwlm: bool
        Specifies whether the Component uses a piecewise linear function.
        If True, either the efficiency or the input output relation are approximated using a PWLM
        If False, the efficiencies are determined using fit_params which may result in a nonlinear model.
    - pwlm_breakpoints: int, number of breakpoints for piecewise linear function
    - make_eff_pwl: bool
        Specifies whether the efficiency or the input output relation are linearised.
        If True, the efficiency is linearised.
        If False, the input output relation is linearised.

    """

    def __init__(self, label, nom_ref, c_ref, c_m, M=1, nom_min=0,
                 nom_max=None, min_part_load=0, base_eff=1,
                 fit_params_nom=None, fit_params_den=None,
                 in_name='input', out_name='output', in_connector_name='IN',
                 out_connector_name='OUT', exists=True, optional=False, use_pwlm=False, pwlm_breakpoints=4,
                 make_eff_pwl=False):
        # If no fit_params are given we assume no dependency on relative output
        if not fit_params_nom:
            fit_params_nom = {0: 1}
        if not fit_params_den:
            fit_params_den = {0: 1}

        super().__init__(label)

        # nominal (installed) output power
        nom_name = out_name + '_nom'
        available = 1
        if exists:
            if optional:  # Component exits but may be sold
                installed = self.make_parameter(nom_name)
                self.add_le_constraint(nom_min, installed,
                                       name=nom_name + '_min')
                self.add_le_constraint(installed, nom_max,
                                       name=nom_name + '_max')
                keep = self.make_design_variable('keep', BINARY, init_val=1)
                available = keep
                out_nom = keep * installed

                # IDEA: Add a discount factor, that would make more sense!
                revenue = c_ref * ((installed * (1 - keep)) / nom_ref) ** M
                # revenue from selling counts as negative investment costs
                self.add_expression('investment_costs', -revenue)
            else:  # Component exists and may NOT be sold
                out_nom = self.make_parameter(nom_name, value=nom_min)
                # self.add_le_constraint(nom_min, out_nom,
                #                        name=nom_name + '_min')
                # self.add_le_constraint(out_nom, nom_max,
                #                        name=nom_name + '_max')

            # investment cost of the component
            inv = c_ref * (out_nom / nom_ref) ** M
            # add maintenance costs to fixed costs
            self.add_expression('fixed_costs', c_m * inv)
        else:
            out_nom = self.make_design_variable(nom_name,
                                                bounds=(0, nom_max),
                                                init_val=nom_max)
            if optional:  # Component doesn't exist but may be installed or not
                exists = self.make_design_variable('exists', BINARY,
                                                   init_val=1)
                available = exists
                self.add_le_constraint(exists * nom_min, out_nom,
                                       name=nom_name + '_min')
                self.add_le_constraint(out_nom, exists * nom_max,
                                       name=nom_name + '_max')
            else:
                out_nom.lb = nom_min
            # NOTE: if optional is false, the component must be installed

            # investment cost of the component
            inv = c_ref * (out_nom / nom_ref) ** M
            # fixed cost of the component
            fc = c_m * inv

            self.add_expression('investment_costs', inv)
            self.add_expression('fixed_costs', fc)

        # relative output
        out_rel = self.make_operational_variable(out_name + '_rel',
                                                 bounds=(0, 1), init_val=1)

        # If minimal part load is considered, we need to decide whether it's
        # operational or not to determine the lower bound for the part load.
        self.add_expression('min_part_load', min_part_load)
        if min_part_load == 0:
            operating = 1
        else:
            # binary indicating whether the component is operational
            operating = self.make_operational_variable('operating',
                                                       domain=BINARY,
                                                       init_val=1)
            if optional:  # available is variable
                self.add_le_constraint(operating, available,
                                       name='availability')
            self.add_ge_constraint(out_rel, operating * min_part_load,
                                   name='min_part_load')
            self.add_le_constraint(out_rel, operating,
                                   name='max_part_load')

        out = self.make_operational_variable(out_name,
                                             bounds=(0, nom_max),
                                             init_val=nom_max)

        if use_pwlm and make_eff_pwl:
            # # use piecewise linear model (PWLM) for the efficiency
            # In oder to be able to use this, add_model function must be added to the COMANDO core manually and LinMoG
            # package must be installed.


            # create a PWLM of the part-load efficiency with out_rel as input
            MILP_model = create_pwlm(base_eff=base_eff, fit_params_nom=fit_params_nom,
                                     fit_params_den=fit_params_den,
                                     pwlm_breakpoints=pwlm_breakpoints, min_part_load=min_part_load,
                                     make_eff_pwl=make_eff_pwl)

            # create variable for the efficiency (output of PWLM)
            eff_pwl = self.make_operational_variable('eff_pwl')
            # add expression for consistency with use_pwlm = False
            self.add_expression('eff', eff_pwl)

            pwl_model_name = label + '_pwl'

            # add the PWLM to comando
            self.add_model(pwlm=MILP_model, name=pwl_model_name, input_vars=out_rel, output_var=eff_pwl,
                           op_var=operating)

            inp = self.make_operational_variable(in_name)

            self.add_eq_constraint(out, out_rel * out_nom, name='component_output')
            self.add_eq_constraint(inp * eff_pwl, out, name='input_output_relation')
        elif use_pwlm and not make_eff_pwl:
            # # use piecewise linear model (PWLM) for the input output relation
            # In oder to be able to use this, add_model function must be added to the COMANDO core manually and LinMoG
            # package must be installed.

            # create a PWLM of input output relation
            MILP_model = create_pwlm(base_eff=base_eff, fit_params_nom=fit_params_nom,
                                     fit_params_den=fit_params_den,
                                     pwlm_breakpoints=pwlm_breakpoints, min_part_load=min_part_load,
                                     make_eff_pwl=make_eff_pwl)

            # create variable for the efficiency and piecewise linear input (input of PWLM)
            inp_pwl_rel = self.make_operational_variable(in_name + 'pwl_rel')
            self.add_expression('inp_pwl', inp_pwl_rel)

            # add the PWLM to comando
            pwl_model_name = label + '_pwl'
            self.add_model(pwlm=MILP_model, name=pwl_model_name, input_vars=inp_pwl_rel, output_var=out_rel,
                           op_var=operating)

            # add expression for consistency with use_pwlm = False
            eff_pwl = out_rel / inp_pwl_rel
            self.add_expression('eff', eff_pwl)

            inp = self.make_operational_variable(in_name)

            self.add_eq_constraint(out, out_rel * out_nom, name='component_output')
            self.add_eq_constraint(inp, inp_pwl_rel * out_nom,
                                   name='input_output_relation')  # inp_pwl_rel = f(out_rel)
        else:
            # use model without piecewise linearisation

            # efficiency in terms of relative output
            numerator = sum(coeff * out_rel ** power
                            for power, coeff in
                            fit_params_nom.items())
            denominator = sum(coeff * out_rel ** power
                              for power, coeff in fit_params_den.items())
            eff = base_eff * numerator / denominator
            self.add_expression('eff', eff)

            self.add_eq_constraint(out, out_rel * out_nom, name='component_output')
            inp_max = nom_max / eff.subs(out_rel, 1)
            try:
                init_ub = float(inp_max)
            except (ValueError, TypeError):  # base_eff contains symbols!
                init_ub = None
            inp = self.make_operational_variable(in_name, bounds=(0, init_ub),
                                                 init_val=init_ub)

            # # Redundant constraints (add as relaxation only?)
            # self.add_le_constraint(inp, out_nom / eff.subs(out_rel, 1),
            #                        name='input_limit')
            # self.add_le_constraint(inp, operating * inp_max,
            #                        name='input_upper_limit')

            self.add_eq_constraint(inp * eff, out, name='input_output_relation')

        self.add_expression('input', inp)
        self.add_expression('output', out)

        # set connectors
        self.add_input(in_connector_name, inp)
        self.add_output(out_connector_name, out)

class CombinedHeatAndPower(IESComponent):
    """Combined-heat-and-power (CHP) engine
    author: n.hampel

    design variable: None
    operational variable(s): operational output (heat & elec.) for each time step
    costs: The investment costs 'c_inv' is a nonlinear function of the nominal output Qdot_nom:
        c_inv = c_inv,ref * (Qdot_nom/Qdot_ref)^M
    input-output:
        input 'Qdot_in': Linear input output relation implemented for the electical output (no use of efficiencies)
        fitted to the manufacturer data.
        second output 'heat_output': Linear relation between P_el_out and heat_output implemented fitted to the
        manufacturer data.

    investment parameters
        P_el_ref in [MW] reference nominal electric power
        c_ref in  [€], reference cost
        c_m : maintenance coefficient, (fraction of investment cost)
        M : cost exponent

    performance parameters
        P_el_nom_min in [MW] minimal nominal electrical power
        P_el_nom_max in [MW] maximal nominal electrical power
        P_el_min_part_load : minimum output part load
        eff_th_nom : nominal thermal efficiency, used to set bounds for heat output
        eff_el_nom : nominal electric efficiency, used to set bounds for heat output
    """



    def __init__(
            self,
            label,
            **kwargs,
    ):
        super().__init__(label, kwargs['max_power'], kwargs['c_ref'], kwargs['c_m'], kwargs['cost_exponent'],
                         kwargs['min_power'], kwargs['max_power'],
                         kwargs['min_part_load'], in_name='Qdot_in',
                         out_name='P_el_out', out_connector_name='POWER_OUT',
                         exists=True, optional=False)
        self.eff_th_nom = kwargs['thermal_efficiency']
        self.eff_el_nom = kwargs['efficiency']

        # direct implementation of the input output relation for both heat and electric output
        inp = self.get_expression('input')
        inp.ub = kwargs['max_power'] / self.eff_el_nom
        out = self.get_expression('output')
        operating = self['operating']
        io_relation_a, io_relation_b = map(float, kwargs['I/O_relation'].split('_'))
        pq_relation_a, pq_relation_b = map(float, kwargs['P/Q_relation'].split('_'))

        # update input output relation
        self.add_eq_constraint(io_relation_b/1000 + io_relation_a * inp, out, 'input_output_relation')


        # add second output for heat
        out_heat = self.make_operational_variable('heat_output', bounds=(0, (kwargs['max_power'] *
                                                                             self.eff_th_nom / self.eff_el_nom)))

        # implement the relation between power and heat output instead of a second input output relation
        self.add_eq_constraint(pq_relation_b/1000 + pq_relation_a * out, out_heat, 'power_heat_relation')

        self.add_le_constraint(out_heat, operating*(kwargs['max_power'] * self.eff_th_nom / self.eff_el_nom))

        self.add_expression('heat_output', out_heat)
        self.add_output('HEAT_OUT', out_heat)

class Boiler(IESComponent):
    """Boiler with a constant efficiency of 80%
    author: n.hampel

    design variable: None
    operational variable: heat output for each time step
    costs: The investment costs 'c_inv' is a nonlinear function of the nominal output Qdot_nom:
        c_inv = c_inv,ref * (Qdot_nom/Qdot_ref)^M
    input-output: The input 'Qdot_in' can be determined via the efficiency relation:
        Qdot_in =  Qdot_out / eff
    The efficiency 'eff' is determined via the product of the nominal
    efficiency and a polynomial fitting function of the load fraction 'q':
        q = Qdot_out/Qdot_nom

    Qdot_ref in [MW], reference nominal power
    c_ref in [€], reference cost
    M :cost exponent
    c_m :maintenance coefficient, (fraction of investment cost)
    Qdot_min in [kW], minimal nominal power allowed for the model
    Qdot_max in [kW], maximal nominal power allowed for the model
    qdot_min :minimum modeled thermal output part load
    """



    # constant efficiency
    #eff_nom: nominal efficiency
    fit_params_nom = {0: 1}# part load behavior
    fit_params_den = {0: 1}


    def __init__(
            self,
            label,
            exists=True,
            optional=False,
            **kwargs,
    ):

        super().__init__(label, kwargs['max_power'], kwargs['c_ref'], kwargs['c_m'], M=kwargs['cost_exponent'],
                         nom_min=kwargs['min_power'], nom_max=kwargs['max_power'],
                         min_part_load=kwargs['min_part_load'], base_eff=kwargs['thermal_efficiency'],
                         fit_params_nom=self.fit_params_nom,
                         fit_params_den=self.fit_params_den, in_name='Qdot_in',
                         out_name='Qdot_out', exists=exists, optional=optional)
        self.kwargs = kwargs

class AbsorptionChiller(IESComponent):
    """Absorption chiller (AC)
    author: n.hampel

    parameterised for YHAU-CH1900EXW4S with part load behaviour as in Sass. However, instances may be used for other
    chillers. Therefore, the following Arguments from IESComponents may be overwritten:
    exists, optional, Qdot_ref, Qdot_nom_min, c_ref

    design variable: nominal cooling power to be installed (or None)
    operational variable: operational output cooling for each time step
    costs: The investment costs 'c_inv' is a nonlinear function of the nominal output Qdot_nom:
        c_inv = c_inv,ref * (Qdot_nom/Qdot_ref)^M
    input-output: Input determined via the efficiency relation
        Qdot_in =  Qdot_out / eff
    The efficiency 'eff' is determined via the product of the nominal
    efficiency and a polynomial fitting function of the load fraction 'q':
    q = Qdot_out/Qdot_nom
    """

    fit_params_nom = {1: 1}
    fit_params_den = {0: 0.0441,
                      1: 0.81248,
                      2: 0.13334}

    def __init__(
            self,
            label,
            exists=True,
            optional=False,
            **kwargs,
    ):
        super().__init__(label, kwargs['max_power'], kwargs['c_ref'], kwargs['c_m'], M=kwargs['cost_exponent'],
                         nom_min=kwargs['min_power'], nom_max=kwargs['max_power'],
                         min_part_load=kwargs['min_part_load'], base_eff=kwargs['thermal_efficiency'],
                         fit_params_nom=self.fit_params_nom,
                         fit_params_den=self.fit_params_den, in_name='Qdot_in',
                         out_name='Qdot_out', exists=exists, optional=optional, use_pwlm=False)
        self.kwargs = kwargs

    def add_to_model(self, pwlm, name, input_vars, output_var, op_var=1):
        """ Add a model"""
        var_mapping = {}  # Mapping to ensure that no name collision happens if more than model is used in one component
        for var in pwlm["variables"]:
            var_mapping[var.name] = self.make_operational_variable(name + var.name, bounds=pwlm["bounds"][var])
        bin_var_mapping = {}  # mapping for binary variables
        for bin_var in pwlm["bin_var"]:
            bin_var_mapping[bin_var.name] = self.make_operational_variable(name + bin_var.name, domain=INTEGER,
                                                                           bounds=(0, 1))

        input_mapping = {}
        if type(pwlm["inputs"]) is list:
            for i in range(pwlm["inputs"].__len__()):
                input_mapping[pwlm["inputs"][i].name] = input_vars[i]
        else:
            input_mapping[pwlm["inputs"].name] = input_vars

        for con in pwlm["constraints"]:
            con_comando = con.subs(var_mapping)
            con_comando = con_comando.subs(bin_var_mapping)
            con_comando = con_comando.subs(input_mapping)
            con_comando = con_comando.subs(pwlm["output"], output_var)
            con_comando = con_comando.subs(pwlm["op_status"], op_var)

            if type(con) is comando.get_backend().Equality:
                name_con = f'{name}{con.args[0]} = {con.args[1]}'
                con_type = 'Eq'
            elif type(con) is comando.get_backend().LessThan:
                name_con = f'{name}{con.args[0]} <= {con.args[1]}'
                con_type = 'Le'
            elif type(con) is comando.get_backend().GreaterThan:
                name_con = f'{name}{con.args[0]} >= {con.args[1]}'
                con_type = 'Ge'
            else:
                name_con = 'UNDEFINED TYPE'
            try:
                self._handle_constraint(con_type, con_comando.lhs, con_comando.rhs, name_con)
            except AttributeError:
                if con_comando:
                    print(f'constraint {name_con} is always satisfied and is therefore skipped')
                elif not con_comando:
                    print(f'constraint {name_con} is always violated')

class Grid(Component):
    """
    A component representing a generic grid connection.
    author: n.hampel
    """

    def __init__(self, label, consumption_limit=None, feedin_limit=0, price=0,
                 compensation=0, single_connector=False, constrain_flow=False,
                 co2_factor=0):
        """Initialize the Grid.

        Arguments
        ---------
        - label : str
            Unique string that serves as an identifier of this Grid.
        - consumption_limit : None, numeric data or expression (default: None)
            Limit the amount of commodity that can be supplied by the grid.
        - feedin_limit : None, numeric data or expression (default: 0)
            Limit the amount of commodity that can be fed into the grid.
        - price : numeric data
            Price per unit of commodity that is consumed.
        - compensation : numeric data
            Price per unit of commodity that is fed into the grid.
        - constrain_flow: bool (default: False)
            Enforce a constraint on the input/output flows
        - single_connector: bool (default: False)
            Specify whether a single, bidirectional connector should be used in
            favor of two directed connectors (consumption/feed-in)
        """
        super().__init__(label)

        if consumption_limit == 0:  # We don't need to look at consumption
            consumption = price = spendings = 0
        else:
            if consumption_limit is None \
                    or isinstance(consumption_limit, (int, float)):
                consumption = self.make_operational_variable(
                    'consumption', bounds=(0, consumption_limit))
            else:  # consumption_limit is an expression
                consumption = self.make_operational_variable('consumption',
                                                             bounds=(0, None))
                self.add_le_constraint(consumption, consumption_limit,
                                       'consumption_limit')
            price = self.make_parameter('price', price)
            spendings = self.add_expression('spendings', price * consumption)

        if feedin_limit == 0:  # We don't need to look at feedin
            feedin = compensation = earnings = 0
        else:
            if feedin_limit is None \
                    or isinstance(feedin_limit, (int, float)):
                feedin = self.make_operational_variable('feedin',
                                                        bounds=(0,
                                                                feedin_limit))
            else:  # feedin_limit is an expression
                feedin = self.make_operational_variable('feedin',
                                                        bounds=(0, None))
                self.add_le_constraint(feedin, feedin_limit, 'feedin_limit')
            compensation = self.make_parameter('compensation', compensation)
            earnings = self.add_expression('earnings', compensation * feedin)

        self.add_expression('variable_costs', spendings - earnings)

        co2_factor = self.make_parameter('co2_factor', value=co2_factor)
        emissions = (consumption - feedin) * co2_factor
        self.add_expression('emissions', emissions)

        if constrain_flow and feedin != 0 and consumption != 0:
            if feedin_limit and consumption_limit:  # Neither is 0 nor None
                # Big-M formulation
                consuming = self.make_operational_variable('consuming',
                                                           domain=BINARY,
                                                           init_val=1)
                self.add_le_constraint(consumption,
                                       consuming * consumption_limit,
                                       'consumption_limit')
                self.add_le_constraint(feedin,
                                       (1 - consuming) * feedin_limit,
                                       'feeding_in')
            else:
                # Complementarity formulation
                self.add_eq_constraint(feedin * consumption, 0,
                                       'flow_complementarity')
            if single_connector:
                self.add_connector('CONSUMPTION', feedin - consumption)
                return
        if feedin != 0:
            self.add_input('FEEDIN', feedin)
        if consumption != 0:
            self.add_output('CONSUMPTION', consumption)

class CompressionChiller(IESComponent):
    """Compression-chiller (CC)
    author: n.hampel

    design variable: None
    operational variable: operational output cooling for each time step
    costs: The investment costs 'c_inv' is a nonlinear function of the nominal output Qdot_nom:
        c_inv = c_inv,ref * (Qdot_nom/Qdot_ref)^M
    input-output: The input 'P_IN' can be determined via the efficiency relation:
        P_IN =  Qdot_out / eff
    The efficiency 'eff' is determined via the product of the nominal
    efficiency and a polynomial fitting function of the load fraction 'q'
        q = Qdot_out/Qdot_nom
    """
    fit_params_nom = {0: 1}

    # maintenance coefficient, fraction of the investment cost
    def __init__(
            self,
            label,
            exists=True,
            optional=False,
            network_coupled=True,

            **kwargs,
    ):
        self.c_ref = kwargs.get('c_ref', 0) * kwargs.get('max_power', 0)
        super().__init__(label, kwargs['max_power'], self.c_ref, kwargs['c_m'], M=kwargs['cost_exponent'],
                         nom_min=kwargs['min_power'], nom_max=kwargs['max_power'],
                         min_part_load=kwargs['min_part_load'], base_eff=kwargs['thermal_efficiency'],
                         fit_params_nom=self.fit_params_nom, in_name='P_in',
                         out_name='Qdot_out', exists=exists, optional=optional)
        self.kwargs = kwargs

        if network_coupled:
            inp = self.get_expression('input')
            out = self.get_expression('output')
            self.add_input('Qdot_reject', inp + out)

class Demand(Component):
    """
    A demand requiring a known amount of an arbitrary commodity.
    author: n.hampel
    """

    def __init__(self, label):
        """Initialize the Demand.

        Arguments
        ---------
        - label : str
            Unique string that serves as an identifier of this Demand.
        - data : numeric data
            Amount of consumed commodity, can be a scalar or a pandas
            Series.
        """
        super().__init__(label)
        demand = self.make_parameter('demand', value=None)
        self.add_expression(label+'_'+'demand', demand)
        self.add_input('IN', demand)

class Source(Component):
    """Simplified HPC with fixed electricity and selectable recovered LT heat.

        max_power supplies the constant electrical consumption [MW]. min_power
        supplies the available LT heat limit [MW]. q_dot is heat admitted to the
        LT bus after implicit free disposal, not total waste-heat generation.
        In LLEC_BPJ these inputs are 13 MW electricity and 11 MW available heat.
        Compensated LT feed-in is a planned sale, not the disposal mechanism.

        ---------
        design variables: none
        operational variable: heat output

        """


    def __init__(self, label,**kwargs,):
        super().__init__(label=label)
        self.Q_dot_max = kwargs['min_power']
        self.make_parameter('T_out', value=None)
        self.make_parameter('T_in', value=None)
        self.make_parameter('Electric_consumption', value=kwargs['max_power'])
        self.make_operational_variable(name="q_dot", bounds=(0, None))
        self.add_le_constraint(self["q_dot"],self.Q_dot_max)
        self.add_output('HEAT_OUT', self["q_dot"])
        self.add_input('EL_IN', self['Electric_consumption'])

class HeatPump(IESComponent):
    """Heat pump (HP) parameterized as in Sass.

    Arguments
    ---------
    use_part_load : bool
        Specifies whether the detailed part load behaviour acc to Fuentes is used.
        If True, the detailed part load behaviour is used and use_pwlm is set to True.
        If False, a linear function for the efficiency is used and use_pwlm is set to False.
    use_lorenz : bool
        Specifies whether the Lorenz or the Carnot COP is used.
        If True, Lorenz COP is used.
        If False, Carnot COP is used.
    ---------
    design variable: nominal heating power to be installed
    operational variable: operational output heating for each time step
    costs: The investment costs 'c_inv' is a nonlinear function of the nominal output Qdot_nom:
        c_inv = c_inv,ref * (Qdot_nom/Qdot_ref)^M
    input-output: Input determined via the efficiency relation
        P_IN =  Qdot_out / (eff * COP)
    The efficiency 'eff' is determined via the product of the nominal
    efficiency and a polynomial fitting function of the load fraction 'q'
    q = Qdot_out/Qdot_nom.
    The COP is determined via the Lorenz COP or via the Carnot COP.
    """

    def __init__(self,
                 label,
                 T_m_eva=None,
                 T_m_con=None,
                 use_part_load=False,
                 use_lorenz=False,
                 **kwargs,):

        C_d = kwargs.get("C_d",0)
        C_c = kwargs.get("C_c",0)
        T_con_out = kwargs["T_con_in"]
        T_eva_out = kwargs["T_eva_out"]
        eff_exer = kwargs["eff_COP"]
        self.Qdot_ref = kwargs["max_power"]
        self.Qdot_min = self.Qdot_ref   # [MW], minimal nominal power allowed for the model; if exist = True this value have to be set = Qdot_ref
        self.Qdot_max = self.Qdot_ref   # [MW], maximal nominal power allowed for the model
        self.M = kwargs["cost_exponent"]
        self.c_ref = kwargs["c_ref"]
        self.c_m = kwargs["c_m"]

        if use_part_load:
            qdot_min = 0.05  # [-] minimum modeled thermal output part load
            fit_params_nom = {1: 1 - C_d,
                              2: C_d}
            fit_params_den = {0: C_d * C_c - C_d - C_c + 1,
                              1: -2 * C_d * C_c + 2 * C_d + C_c,
                              2: C_d * C_c - C_d}
            use_pwlm = True
        else:
            qdot_min = 0.05  # [-] minimum modeled thermal output part load
            fit_params_nom = {0: 0.725, 1: 0.275}
            fit_params_den = {0: 1}
            # fit_params_nom = {0: 1}
            # fit_params_den = {0: 1}
            use_pwlm = False

        if use_lorenz:
            # Calculate temperature dependent COP using Lorenz
            eff_lorenz = 0.5  # estimated for lorenz
            # T_m_con = (T_con_in - T_con_out) / (np.log(T_con_in) - np.log(T_con_out))
            # T_m_eva = (T_eva_in - T_eva_out) / (np.log(T_eva_in) - np.log(T_eva_out))
            COP_lorenz = T_m_con / (T_m_con - T_m_eva)
            COP = eff_lorenz * COP_lorenz
        else:
            # Calculate temperature dependent COP using Carnot
            # TODO: re-evalute eff_exer for carnot, if carnot should be used
            COP_carnot = T_con_out / (T_con_out - T_eva_out)
            COP = eff_exer * COP_carnot

        super().__init__(label, nom_ref=self.Qdot_ref, c_ref=self.c_ref, c_m=self.c_m, M=self.M,
                         nom_min=self.Qdot_min, nom_max=self.Qdot_max, min_part_load=qdot_min,
                         base_eff=1, fit_params_den=fit_params_den, fit_params_nom=fit_params_nom,
                         in_name='P_in', out_name='Qdot_out',
                         exists=True, optional=False, use_pwlm=use_pwlm, pwlm_breakpoints=4)

        inp = self.get_expression('input')
        out = self.get_expression('output')
        eff = self.get_expression('eff')
        self.add_eq_constraint(inp * COP * eff, out, 'input_output_relation')

        # Possible adjustments for connection to Jupiter computer
        heat_in = self.make_operational_variable('heat_in', bounds=(0,None))
        self.add_eq_constraint(heat_in + inp, out, 'energy_balance')
        self.add_input('Qdot_in', heat_in)

    def add_model(self, pwlm, name, input_vars, output_var, op_var=1):
        """ Add a model"""
        var_mapping = {}  # Mapping to ensure that no name collision happens if more than model is used in one component
        for var in pwlm["variables"]:
            var_mapping[var.name] = self.make_operational_variable(name + var.name, bounds=pwlm["bounds"][var])
        bin_var_mapping = {}  # mapping for binary variables
        for bin_var in pwlm["bin_var"]:
            bin_var_mapping[bin_var.name] = self.make_operational_variable(name + bin_var.name, domain=INTEGER,
                                                                           bounds=(0, 1))

        input_mapping = {}
        if type(pwlm["inputs"]) is list:
            for i in range(pwlm["inputs"].__len__()):
                input_mapping[pwlm["inputs"][i].name] = input_vars[i]
        else:
            input_mapping[pwlm["inputs"].name] = input_vars

        for con in pwlm["constraints"]:
            con_comando = con.subs(var_mapping)
            con_comando = con_comando.subs(bin_var_mapping)
            con_comando = con_comando.subs(input_mapping)
            con_comando = con_comando.subs(pwlm["output"], output_var)
            con_comando = con_comando.subs(pwlm["op_status"], op_var)

            if type(con) is comando.get_backend().Equality:
                name_con = f'{name}{con.args[0]} = {con.args[1]}'
                con_type = 'Eq'
            elif type(con) is comando.get_backend().LessThan:
                name_con = f'{name}{con.args[0]} <= {con.args[1]}'
                con_type = 'Le'
            elif type(con) is comando.get_backend().GreaterThan:
                name_con = f'{name}{con.args[0]} >= {con.args[1]}'
                con_type = 'Ge'
            else:
                name_con = 'UNDEFINED TYPE'
            try:
                self._handle_constraint(con_type, con_comando.lhs, con_comando.rhs, name_con)
            except AttributeError:
                if con_comando:
                    print(f'constraint {name_con} is always satisfied and is therefore skipped')
                elif not con_comando:
                    print(f'constraint {name_con} is always violated')

class Battery(Component):

    def __init__(self, label,
                  **kwargs,):
        super().__init__(label)

        soc_init = kwargs.get('soc_init', 0.0)
        P_nom = kwargs.get('max_power', 0.1)
        self.capacity = kwargs.get('max_capacity', 0.5)  # MWh
        self.max_el_in = P_nom   # MW
        self.max_el_out = P_nom  # MW
        self.initial_soc = soc_init * self.capacity
        self.c_loss = kwargs.get('storage_loss_rate', 0)
        self.charge_eff = kwargs.get('efficiency', 1)  # [-]
        self.discharge_eff = kwargs.get('discharge_efficiency', 1)
        self.c_ref = kwargs["c_ref"]
        self.c_m = kwargs["c_m"]
        self.M = kwargs["cost_exponent"]


        el_in = self.make_operational_variable("el_in", bounds=(0, self.max_el_in))
        el_out = self.make_operational_variable("el_out", bounds=(0,  self.max_el_out))
        b_op = self.make_operational_variable("b_op", domain=BINARY, bounds=(0, 1))

        self.add_le_constraint(el_in, b_op * self.max_el_in)
        self.add_le_constraint(el_out, (1 - b_op) * self.max_el_out)

        soc = self.make_operational_variable('soc', bounds=(0, self.capacity))
        dissipation = self.add_expression('dissipation', soc * self.c_loss)
        state_change = el_in * self.charge_eff - el_out / self.discharge_eff - dissipation
        self.declare_state(soc, state_change, init_state=self.initial_soc)

        investment_costs = self.c_ref * (self.capacity / 0.001) ** self.M #c_ref is EUR/kWh at this size
        # fixed cost of the component
        fixed_costs = self.c_m * investment_costs

        self.add_expression('investment_costs', investment_costs)
        self.add_expression('fixed_costs', fixed_costs)

        self.add_input('EL_IN', el_in)
        self.add_output('EL_OUT', el_out)

class Source_RE(Component):
    """ Simple Source component for the RE generators
        ---------
        design variables: none
        parameter: electricity production profile 'el_prod'
        operational variable: 'curtailed'
        costs: optional flat price per unit of commodity
        """
    def __init__(self, label, **kwargs,):
        super().__init__(label=label)

        self.price = kwargs.get('price', 0)
        self.contracted_annual_mwh = kwargs.get('contracted_annual_mwh', None)

        el_prod = self.make_parameter('el_prod')
        el_out = self.make_operational_variable('el_out', bounds=(0, None))
        curtailed = self.make_operational_variable('curtailed', bounds=(0, None))
        self.add_eq_constraint(el_out + curtailed, el_prod, 'RE_balance')
        self.add_output('OUT', el_out)

        price = self.make_parameter('price', self.price)
        if self.contracted_annual_mwh is not None:
            self.add_expression('fixed_costs', price * self.contracted_annual_mwh)
        else:
            self.add_expression('variable_costs', price * el_out)

class ReversibleAirWaterHeatPump(Component):
    """Fleet of reversible air-water heat pumps with fixed COP/EER figures.

    """

    def __init__(self, label, **kwargs,):
        super().__init__(label)

        Q_heat_max = kwargs.get('max_power', 5)
        cop_heat = kwargs.get('eff_COP', 3)
        eer_cool = kwargs.get('cool_cop', 3)
        self.c_ref_per_MW = kwargs.get('c_ref')# [€/MW_th], same basis as HeatPump (Pieper et al.)
        self.c_m = kwargs.get('c_m')# [-], maintenance coefficient (fraction of investment cost)
        P_el_nom = Q_heat_max / cop_heat
        Q_cool_max = eer_cool * P_el_nom

        P_el_heat = self.make_operational_variable('P_el_heat', bounds=(0, P_el_nom))
        P_el_cool = self.make_operational_variable('P_el_cool', bounds=(0, P_el_nom))
        self.add_le_constraint(P_el_heat + P_el_cool, P_el_nom, 'shared_compressor_capacity')

        Qdot_heat = cop_heat * P_el_heat
        Qdot_cool = eer_cool * P_el_cool
        self.add_expression('Qdot_heat', Qdot_heat)
        self.add_expression('Qdot_cool', Qdot_cool)

        investment_costs = self.c_ref_per_MW * Q_heat_max
        self.add_expression('investment_costs', investment_costs)
        self.add_expression('fixed_costs', self.c_m * investment_costs)

        self.add_input('P_IN', P_el_heat + P_el_cool)
        self.add_output('Qdot_heat_out', Qdot_heat)
        self.add_output('Qdot_cool_out', Qdot_cool)

class CombinedThermalStorage(Component):
    # efficiencies -- ASSUMPTION 2 above
    charge_eff = 0.95
    discharge_eff = 0.95
    # operational parameters -- ASSUMPTION 4 above
    in_min = 0
    in_max = 1
    out_min = 0
    out_max = 1
    # physical properties of water
    rho = 1000  # [kg/m^3]
    cp = 4186  # [J/kg/K]

    def __init__(self, label, eta_tc=0.8,
                 **kwargs,):
        """
        Arguments
        ---------
        The fixed max_capacity determines effective water volume [m^3].
        dT_cool: cooling-duty temperature spread [K] (narrower than dT_heat)
        HD_ratio: height-to-diameter ratio [-]
        K: overall wall heat transfer coefficient [W/m^2/K]
        eta_tc: thermocline efficiency, usable capacity fraction [-]
        cyclic: whether the SOC must return to its start-of-period value
        tes_T_flow, tes_T_return: fixed scalar temperatures in
            a common temperature scale. Their difference is in Kelvin.
        max_capacity: authoritative usable heat energy [MWh].
        soc_init: initial fraction of usable energy; zero is not 0 K.
        """

        super().__init__(label)
        # cost parameters (Sass2020 basis, same as HotWaterStorage)
        self.c_ref = kwargs.get("c_ref")# [EUR], reference cost (83.8 EUR/kWh at reference size)
        self.M = kwargs.get("cost_exponent")# [-], cost exponent
        self.c_m = kwargs.get("c_m") # [-], maintenance coefficient (fraction of investment cost)
        dT_cool = kwargs.get("dt_cool", 10)
        HD_ratio = kwargs.get("hd_ratio", 2)
        K = kwargs.get("k", 0.5)
        cyclic = kwargs.get("cyclic", True)
        if isinstance(cyclic, str):
            if cyclic.lower() not in {"true", "false"}:
                raise ValueError(f"{label}: cyclic must be true or false")
            cyclic = cyclic.lower() == "true"
        self.cyclic = cyclic
        self.charge_eff = kwargs.get("efficiency", type(self).charge_eff)
        self.discharge_eff = kwargs.get("discharge_efficiency", type(self).discharge_eff)
        # Fixed operating temperatures; only their difference sets usable energy.
        T_flow = kwargs.get("tes_T_flow", 10)
        T_return = kwargs.get("tes_T_return", 5)
        dT_heat = T_flow - T_return
        cap = kwargs["max_capacity"]  # MWh of usable thermal energy
        if (not all(np.isfinite(value) for value in [cap, dT_heat, dT_cool, eta_tc, HD_ratio, K])
                or cap <= 0 or dT_heat <= 0 or dT_cool <= 0
                or not 0 < eta_tc <= 1 or HD_ratio <= 0 or K < 0):
            raise ValueError(f"{label}: invalid capacity, temperature spreads, efficiency or tank geometry")
        # Capacity is fixed by the input; volume is a derived geometry, not a
        # design decision or an independent, potentially inconsistent default.
        volume = cap * 3.6e9 / (eta_tc * self.rho * self.cp * dT_heat)
        self.capacity = cap
        self.volume = volume
        dT_heat_max = dT_heat
        max_cap = cap
        cool_ratio = dT_cool / dT_heat_max
        max_cap_cool = max_cap * cool_ratio
        D = (4 * volume / (np.pi * HD_ratio)) ** (1 / 3)
        c_loss = 4 * K * 3600 / (self.rho * self.cp * D)

        # The same fixed usable capacity sets investment cost and SOC bounds.
        investment_costs = self.c_ref * (cap / 0.001) ** self.M # [MWh], reference nominal capacity (= 1 kWh)
        self.add_expression('investment_costs', investment_costs)
        self.add_expression('fixed_costs', self.c_m * investment_costs)

        soc_heat = self.make_operational_variable('soc_heat', bounds=(0, max_cap))

        Qdot_in_heat = self.make_operational_variable('Qdot_in_heat', bounds=(0, max_cap))
        b_in_heat = self.make_operational_variable('b_in_heat', domain=BINARY, init_val=0)
        self.add_le_constraint(Qdot_in_heat, b_in_heat * max_cap * self.in_max,
                               name='input_heat_max_switch')
        self.add_le_constraint(Qdot_in_heat, cap * self.in_max, name='input_heat_max')
        self.add_ge_constraint(Qdot_in_heat, b_in_heat * cap * self.in_min,
                               name='input_heat_min')

        Qdot_out_heat = self.make_operational_variable('Qdot_out_heat', bounds=(0, max_cap))
        b_out_heat = self.make_operational_variable('b_out_heat', domain=BINARY, init_val=1)
        self.add_le_constraint(Qdot_out_heat, b_out_heat * max_cap * self.out_max,
                               name='output_heat_max_switch')
        self.add_le_constraint(Qdot_out_heat, cap * self.out_max, name='output_heat_max')
        self.add_ge_constraint(Qdot_out_heat, b_out_heat * cap * self.out_min,
                               name='output_heat_min')

        dissipation_heat = self.add_expression('dissipation_heat', soc_heat * c_loss)
        state_change_heat = (Qdot_in_heat * self.charge_eff
                             - Qdot_out_heat / self.discharge_eff
                             - dissipation_heat)
        self.declare_state(soc_heat, state_change_heat,
                           float('nan') if cyclic else 0)

        self.add_le_constraint(b_in_heat + b_out_heat, 1,
                               name='charge_discharge_switch_heat')

        # ============== cool duty ==============
        soc_cool = self.make_operational_variable('soc_cool', bounds=(0, max_cap_cool))
        self.add_le_constraint(soc_heat + soc_cool / cool_ratio,
                               cap, name='joint_occupancy_max')

        Qdot_in_cool = self.make_operational_variable('Qdot_in_cool', bounds=(0, max_cap_cool))
        b_in_cool = self.make_operational_variable('b_in_cool', domain=BINARY, init_val=0)
        self.add_le_constraint(Qdot_in_cool, b_in_cool * max_cap_cool * self.in_max,
                               name='input_cool_max_switch')
        self.add_le_constraint(Qdot_in_cool, cap * cool_ratio * self.in_max,
                               name='input_cool_max')
        self.add_ge_constraint(Qdot_in_cool, b_in_cool * cap * cool_ratio * self.in_min,
                               name='input_cool_min')

        Qdot_out_cool = self.make_operational_variable('Qdot_out_cool', bounds=(0, max_cap_cool))
        b_out_cool = self.make_operational_variable('b_out_cool', domain=BINARY, init_val=1)
        self.add_le_constraint(Qdot_out_cool, b_out_cool * max_cap_cool * self.out_max,
                               name='output_cool_max_switch')
        self.add_le_constraint(Qdot_out_cool, cap * cool_ratio * self.out_max,
                               name='output_cool_max')
        self.add_ge_constraint(Qdot_out_cool, b_out_cool * cap * cool_ratio * self.out_min,
                               name='output_cool_min')

        dissipation_cool = self.add_expression('dissipation_cool', soc_cool * c_loss)
        state_change_cool = (Qdot_in_cool * self.charge_eff
                             - Qdot_out_cool / self.discharge_eff
                             - dissipation_cool)
        self.declare_state(soc_cool, state_change_cool,
                           float('nan') if cyclic else 0)

        self.add_le_constraint(b_in_cool + b_out_cool, 1,
                               name='charge_discharge_switch_cool')

        self.add_input('IN_heat', Qdot_in_heat)
        self.add_output('OUT_heat', Qdot_out_heat)
        self.add_input('IN_cool', Qdot_in_cool)
        self.add_output('OUT_cool', Qdot_out_cool)

comando_dst : dict = {
    "chp" : CombinedHeatAndPower,
    "boiler" : Boiler,
    "absorption_chiller" : AbsorptionChiller,
    "grid" : Grid,
    "compression_chiller" : CompressionChiller,
    "demand": Demand,
    "hpc" : Source,
    "heat_pump" : HeatPump,
    "battery": Battery,
    "res": Source_RE,
    "rawhp": ReversibleAirWaterHeatPump,
    "tes": CombinedThermalStorage,
}
