import pyomo.environ as pyo
import pandas as pd
import numpy as np

TOD_BLOCKS = ["Normal", "Solar", "Peak"]
EPSILON_PRIORITY_WEIGHT = 1e-6  # tie-break only
YEARS = range(26)  # 0 = CAPEX outlay year, 1..25 = operating years
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class BESSSettlementOptimizer:
    def __init__(
        self,
        excel_path,
        generation_sheet="Wind-Solar+BESS",
        settlement_sheet="Settlement",
        skiprows=4,
        max_soc_perc=0.85,
        min_soc_perc=0.15,
        max_bess_hours=12.0,     # NOTE: assumption -- not specified in the
                                 # request; needed so `hours` has a finite
                                 # upper bound (see _add_battery_constraints).
        pcs_cap=150_000.0,       # max charge OR discharge in one hour (kW)
        max_grid_allowable=199_500.0,
        max_grid_wind=49_500.0,
        max_grid_solar=150_000.0,  # cap on Solar+BESS injection; also used
                                    # as the per-hour basis for BESS sizing
        solver_name="appsi_highs",
        RTE=0.85,
        solar_capex=40,
        wind_capex=90,
        BESS_capex=10,
        discharge_degradation=0.99,
        cost_escalation=1.03,
        tariff_cap=5.5,
        min_effective_replacement=0.7,
        dinkelbach_tol=1e-6,
        dinkelbach_max_iter=30,
    ):
        self.excel_path = excel_path
        self.generation_sheet = generation_sheet
        self.settlement_sheet = settlement_sheet
        self.skiprows = skiprows

        self.max_soc_perc = max_soc_perc
        self.min_soc_perc = min_soc_perc
        self.max_bess_hours = max_bess_hours

        self.pcs_cap = pcs_cap
        self.RTE = RTE

        self.max_grid_allowable = max_grid_allowable
        self.max_grid_wind = max_grid_wind
        self.max_grid_solar = max_grid_solar

        self.solver_name = solver_name

        self.solar_capex = solar_capex
        self.wind_capex = wind_capex
        self.BESS_capex = BESS_capex

        self.discharge_degradation = discharge_degradation
        self.cost_escalation = cost_escalation
        self.tariff_cap = tariff_cap
        self.min_effective_replacement = min_effective_replacement

        self.dinkelbach_tol = dinkelbach_tol
        self.dinkelbach_max_iter = dinkelbach_max_iter

        self.model = None
        self.results = None
        self._results_df = None
        self.dinkelbach_history = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self):
        gen_df = pd.read_excel(self.excel_path, sheet_name=self.generation_sheet, skiprows=self.skiprows)
        self.solar_gen = gen_df["Solar (at XMWp)"].to_dict()
        self.wind_gen = gen_df["Wind (at XMWp)"].to_dict()
        self.n_hours = len(gen_df)

        self.hour_to_month = {}
        self.hour_to_tod = {}
        date_range = pd.date_range(start="2026-01-01 00:00", periods=self.n_hours, freq="h")
        for i, dt in enumerate(date_range):
            self.hour_to_month[i] = dt.month
            if 9 <= dt.hour < 17:
                self.hour_to_tod[i] = "Solar"
            elif 17 <= dt.hour < 24:
                self.hour_to_tod[i] = "Peak"
            else:
                self.hour_to_tod[i] = "Normal"  # 0-9 and (wrap) 24 -> midnight-9am

        settle_df = pd.read_excel(self.excel_path, sheet_name=self.settlement_sheet, header=None)
        month_cols = list(range(17, 29))  # 12 columns, Jan..Dec
        self.consumption = {}
        block_rows = {"Normal": 8, "Solar": 9, "Peak": 10}  # 0-based row indices
        for block, row_idx in block_rows.items():
            for month_num, col_idx in enumerate(month_cols, start=1):
                self.consumption[(month_num, block)] = float(settle_df.iat[row_idx, col_idx])

        self.total_annual_consumption = sum(self.consumption.values())
        return self

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def build_model(self):
        m = pyo.ConcreteModel(name="Wind-Solar+BESS-Settlement-Financial")
        self.model = m

        m.T = pyo.RangeSet(0, self.n_hours - 1)
        m.Months = pyo.Set(initialize=list(range(1, 13)))
        m.ToD_Blocks = pyo.Set(initialize=TOD_BLOCKS)

        # ---------------- Battery sizing variables ----------------
        m.hours = pyo.Var(domain=pyo.NonNegativeIntegers)          # BESS storage hours (decision)
        m.BESS_Capacity = pyo.Var(domain=pyo.NonNegativeReals)   # kWh
        m.Max_SoC = pyo.Var(domain=pyo.NonNegativeReals)
        m.Min_SoC = pyo.Var(domain=pyo.NonNegativeReals)

        # ---------------- Hourly dispatch variables ----------------
        m.BESS_Charge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.BESS_Discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.SoC = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        m.Solar_Injected = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Wind_Injected = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Curtailed_Solar = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Curtailed_Wind = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.Grid_Injected = pyo.Var(m.T, domain=pyo.NonNegativeReals)

        # ---------------- Monthly settlement variables ----------------
        m.Monthly_Generation = pyo.Var(m.Months, m.ToD_Blocks, domain=pyo.NonNegativeReals)
        m.Direct_Settled = pyo.Var(m.Months, m.ToD_Blocks, domain=pyo.NonNegativeReals)
        m.Peak_Surplus = pyo.Var(m.Months, domain=pyo.NonNegativeReals)
        m.Banked_to_Solar = pyo.Var(m.Months, domain=pyo.NonNegativeReals)
        m.Banked_to_Normal = pyo.Var(m.Months, domain=pyo.NonNegativeReals)
        m.Settled = pyo.Var(m.Months, m.ToD_Blocks, domain=pyo.NonNegativeReals)

        # explicit annual aggregates (requested explicitly, not just inlined
        # sums) -- makes them directly inspectable after solving.
        m.Total_Consumption = pyo.Var(domain=pyo.NonNegativeReals)
        m.Total_Banked = pyo.Var(domain=pyo.NonNegativeReals)
        m.Total_Settled_Energy = pyo.Var(domain=pyo.NonNegativeReals)
        m.Effective_Replacement = pyo.Var(domain=pyo.NonNegativeReals)

        # ---------------- Financial variables -----------------------
        m.Discharged = pyo.Var(YEARS, domain=pyo.NonNegativeReals)
        m.Revenue = pyo.Var(YEARS, domain=pyo.NonNegativeReals)
        m.Costs = pyo.Var(YEARS, domain=pyo.NonNegativeReals)
        m.EBITDA = pyo.Var(YEARS, domain=pyo.Reals)  # Year 0 is negative (the CAPEX outlay)
        m.tariff = pyo.Var(domain=pyo.NonNegativeReals)
        m.CAPEX = pyo.Var(domain=pyo.NonNegativeReals)

        self._add_battery_constraints()
        self._add_hourly_dispatch_constraints()
        self._add_settlement_constraints()
        self._add_financial_constraints()
        # NOTE: the objective is NOT added here -- see solve(), which builds
        # it fresh on every Dinkelbach iteration (required because the
        # Capex/EBITDA ratio isn't a fixed linear expression).
        return self

    # ------------------------------------------------------------------
    def _add_battery_constraints(self):
        m = self.model

        def bess_capacity_rule(m):
            return m.BESS_Capacity == m.hours * self.max_grid_solar
        m.bess_capacity_con = pyo.Constraint(rule=bess_capacity_rule)

        def max_soc_rule(m):
            return m.Max_SoC == self.max_soc_perc * m.BESS_Capacity
        m.max_soc_def_con = pyo.Constraint(rule=max_soc_rule)

        def min_soc_rule(m):
            return m.Min_SoC == self.min_soc_perc * m.BESS_Capacity
        m.min_soc_def_con = pyo.Constraint(rule=min_soc_rule)

        # Upper bound on `hours` -- see __init__ docstring note. Without
        # some cap, `hours` has no natural ceiling (domain is [0, inf)),
        # which is bad practice even though the capex/opex-vs-benefit
        # trade-off likely bounds the true optimum well below this anyway.
        def max_hours_rule(m):
            return m.hours <= self.max_bess_hours
        m.max_hours_con = pyo.Constraint(rule=max_hours_rule)

    def _add_hourly_dispatch_constraints(self):
        m = self.model
        solar_gen, wind_gen = self.solar_gen, self.wind_gen

        def pcs_charge_cap(m, t):
            return m.BESS_Charge[t] <= self.pcs_cap
        m.pcs_charge = pyo.Constraint(m.T, rule=pcs_charge_cap)

        def pcs_discharge_cap(m, t):
            return m.BESS_Discharge[t] <= self.pcs_cap
        m.pcs_discharge = pyo.Constraint(m.T, rule=pcs_discharge_cap)

        def soc_upper(m, t):
            return m.SoC[t] <= m.Max_SoC
        m.soc_upper = pyo.Constraint(m.T, rule=soc_upper)

        def soc_lower(m, t):
            return m.SoC[t] >= m.Min_SoC
        m.soc_lower = pyo.Constraint(m.T, rule=soc_lower)

        def soc_balance(m, t):
            prev = m.Min_SoC if t == 0 else m.SoC[t - 1]
            return m.SoC[t] == prev + m.BESS_Charge[t] - m.BESS_Discharge[t]
        m.soc_balance = pyo.Constraint(m.T, rule=soc_balance)

        def max_wind(m, t):
            return m.Wind_Injected[t] <= self.max_grid_wind
        m.max_wind = pyo.Constraint(m.T, rule=max_wind)

        def wind_injection_rule(m, t):
            return m.Wind_Injected[t] + m.Curtailed_Wind[t] == wind_gen[t]
        m.wind_injection = pyo.Constraint(m.T, rule=wind_injection_rule)

        def max_solar(m, t):
            return m.Solar_Injected[t] <= self.max_grid_solar
        m.max_solar = pyo.Constraint(m.T, rule=max_solar)

        def solar_injection_rule(m, t):
            return (
                m.Solar_Injected[t] + m.Curtailed_Solar[t]
                == solar_gen[t] - m.BESS_Charge[t] + m.BESS_Discharge[t] * self.RTE
            )
        m.solar_injection = pyo.Constraint(m.T, rule=solar_injection_rule)

        def grid_injected_rule(m, t):
            return m.Grid_Injected[t] == m.Solar_Injected[t] + m.Wind_Injected[t]
        m.grid_injected = pyo.Constraint(m.T, rule=grid_injected_rule)

        def max_grid_allowable_rule(m, t):
            return m.Grid_Injected[t] <= self.max_grid_allowable
        m.max_grid_allowable_con = pyo.Constraint(m.T, rule=max_grid_allowable_rule)

    def _add_settlement_constraints(self):
        m = self.model

        def monthly_settlement_rule(m, month, block):
            hours = [t for t in m.T if self.hour_to_month[t] == month and self.hour_to_tod[t] == block]
            return m.Monthly_Generation[month, block] == sum(m.Grid_Injected[t] for t in hours)
        m.monthly_settlement_con = pyo.Constraint(m.Months, m.ToD_Blocks, rule=monthly_settlement_rule)

        def direct_settled_gen_cap(m, month, block):
            return m.Direct_Settled[month, block] <= m.Monthly_Generation[month, block]
        m.direct_settled_gen_cap = pyo.Constraint(m.Months, m.ToD_Blocks, rule=direct_settled_gen_cap)

        def direct_settled_cons_cap(m, month, block):
            return m.Direct_Settled[month, block] <= self.consumption[(month, block)]
        m.direct_settled_cons_cap = pyo.Constraint(m.Months, m.ToD_Blocks, rule=direct_settled_cons_cap)

        def peak_surplus_rule(m, month):
            return m.Peak_Surplus[month] == m.Monthly_Generation[month, "Peak"] - m.Direct_Settled[month, "Peak"]
        m.peak_surplus_con = pyo.Constraint(m.Months, rule=peak_surplus_rule)

        def banked_to_solar_surplus_cap(m, month):
            return m.Banked_to_Solar[month] <= m.Peak_Surplus[month]
        m.banked_to_solar_surplus_cap = pyo.Constraint(m.Months, rule=banked_to_solar_surplus_cap)

        def banked_to_solar_need_cap(m, month):
            return m.Banked_to_Solar[month] <= self.consumption[(month, "Solar")] - m.Direct_Settled[month, "Solar"]
        m.banked_to_solar_need_cap = pyo.Constraint(m.Months, rule=banked_to_solar_need_cap)

        def banked_to_normal_surplus_cap(m, month):
            return m.Banked_to_Normal[month] <= m.Peak_Surplus[month] - m.Banked_to_Solar[month]
        m.banked_to_normal_surplus_cap = pyo.Constraint(m.Months, rule=banked_to_normal_surplus_cap)

        def banked_to_normal_need_cap(m, month):
            return m.Banked_to_Normal[month] <= self.consumption[(month, "Normal")] - m.Direct_Settled[month, "Normal"]
        m.banked_to_normal_need_cap = pyo.Constraint(m.Months, rule=banked_to_normal_need_cap)

        def settled_normal_rule(m, month):
            return m.Settled[month, "Normal"] == m.Direct_Settled[month, "Normal"] + m.Banked_to_Normal[month]
        m.settled_normal_con = pyo.Constraint(m.Months, rule=settled_normal_rule)

        def settled_solar_rule(m, month):
            return m.Settled[month, "Solar"] == m.Direct_Settled[month, "Solar"] + m.Banked_to_Solar[month]
        m.settled_solar_con = pyo.Constraint(m.Months, rule=settled_solar_rule)

        def settled_peak_rule(m, month):
            return m.Settled[month, "Peak"] == m.Direct_Settled[month, "Peak"]
        m.settled_peak_con = pyo.Constraint(m.Months, rule=settled_peak_rule)

        # ---- explicit annual aggregates ----
        m.Total_Consumption.fix(self.total_annual_consumption)

        def total_banked_rule(m):
            return m.Total_Banked == sum(m.Banked_to_Solar[month] + m.Banked_to_Normal[month] for month in m.Months)
        m.total_banked_con = pyo.Constraint(rule=total_banked_rule)

        def total_settled_energy_rule(m):
            return m.Total_Settled_Energy == sum(
                m.Settled[month, block] for month in m.Months for block in m.ToD_Blocks
            )
        m.total_settled_energy_con = pyo.Constraint(rule=total_settled_energy_rule)

        # Effective Replacement, computed EXACTLY the way the sheet does it:
        # total settled (post-banking) energy / total annual consumption.
        # This divides by self.total_annual_consumption -- a plain Python
        # constant, not a Pyomo Var -- so the expression stays linear.
        def effective_replacement_rule(m):
            return m.Effective_Replacement == m.Total_Settled_Energy / self.total_annual_consumption
        m.effective_replacement_def_con = pyo.Constraint(rule=effective_replacement_rule)

        def effective_replacement_floor_rule(m):
            return m.Effective_Replacement >= self.min_effective_replacement
        m.effective_replacement_floor = pyo.Constraint(rule=effective_replacement_floor_rule)

    def _add_financial_constraints(self):
        """
        See the module docstring for the two nonlinearity fixes (Dinkelbach
        for the ratio objective, tariff dominance argument for the bilinear
        Revenue term). Everything below is linear given those two.
        """
        m = self.model

        # ---- CAPEX: same formula as the previous script, but BESS_Capacity
        # is now a Var (driven by `hours`), so CAPEX genuinely varies with
        # the battery-sizing decision instead of being a fixed constant. ----
        def capex_rule(m):
            return m.CAPEX == (
                self.solar_capex * 2 * (self.max_grid_solar / 1000)
                + self.wind_capex * (self.max_grid_wind / 1000)
                + self.BESS_capex * (m.BESS_Capacity / 1000)
            )
        m.capex_con = pyo.Constraint(rule=capex_rule)

        # ---- tariff: kept as a Var with an explicit <= cap, per the ask --
        # but ALSO fixed at that cap. This is not an arbitrary shortcut: see
        # the module docstring's dominance argument (minimizing Capex/EBITDA
        # always wants tariff as high as possible, for any dispatch outcome,
        # so the unconstrained optimum always sits exactly at the cap).
        # Fixing it is what keeps Revenue = tariff * Discharged linear
        # instead of a bilinear (Var * Var) term HiGHS cannot solve.
        def tariff_cap_rule(m):
            return m.tariff <= self.tariff_cap
        m.tariff_cap_con = pyo.Constraint(rule=tariff_cap_rule)
        m.tariff.fix(self.tariff_cap)

        # ---- Year 0: no discharge/revenue, EBITDA is just -CAPEX ----
        m.Discharged[0].fix(0.0)
        m.Revenue[0].fix(0.0)
        m.Costs[0].fix(0.0)

        def year0_ebitda_rule(m):
            return m.EBITDA[0] == -m.CAPEX
        m.year0_ebitda_con = pyo.Constraint(rule=year0_ebitda_rule)

        # ---- Year 1 Discharged (Mn kWh) == the model's own optimized total
        # annual Grid_Injected (kWh), converted to Mn kWh. ----
        def year1_discharged_rule(m):
            total_grid_injected_kwh = sum(m.Grid_Injected[t] for t in m.T)
            return m.Discharged[1] == total_grid_injected_kwh / 1_000_000.0
        m.year1_discharged_con = pyo.Constraint(rule=year1_discharged_rule)

        # ---- Years 2-25: 1%/yr discharge degradation ----
        def discharge_degradation_rule(m, year):
            return m.Discharged[year] == m.Discharged[year - 1] * self.discharge_degradation
        m.discharge_degradation_con = pyo.Constraint(
            [y for y in YEARS if y >= 2], rule=discharge_degradation_rule
        )

        # ---- Costs Year 1: same O&M formula as the previous script, now
        # driven by the variable BESS_Capacity instead of a fixed constant.
        def costs_year1_rule(m):
            return m.Costs[1] == (
                100000 * m.BESS_Capacity / 1000
                + (500000 + 127100) * self.max_grid_solar / 1000
                + (910000 + 127100) * self.max_grid_wind / 1000
            ) * 1.18 / 1_000_000 + 0.5
        m.costs_year1_con = pyo.Constraint(rule=costs_year1_rule)

        # ---- Years 2-25: 3%/yr cost escalation ----
        def cost_escalation_rule(m, year):
            return m.Costs[year] == m.Costs[year - 1] * self.cost_escalation
        m.cost_escalation_con = pyo.Constraint(
            [y for y in YEARS if y >= 2], rule=cost_escalation_rule
        )

        # ---- Revenue = tariff x Discharged (linear: tariff is fixed) ----
        def revenue_rule(m, year):
            return m.Revenue[year] == m.tariff * m.Discharged[year]
        m.revenue_con = pyo.Constraint([y for y in YEARS if y >= 1], rule=revenue_rule)

        # ---- EBITDA = Revenue - Costs ----
        def ebitda_rule(m, year):
            return m.EBITDA[year] == m.Revenue[year] - m.Costs[year]
        m.ebitda_con = pyo.Constraint([y for y in YEARS if y >= 1], rule=ebitda_rule)

    # ------------------------------------------------------------------
    # Dinkelbach's algorithm: exact solution of "minimize CAPEX / EBITDA[1]"
    # ------------------------------------------------------------------
    def _set_dinkelbach_objective(self, t):
        m = self.model
        if hasattr(m, "Objective"):
            m.del_component(m.Objective)

        def dinkelbach_objective_rule(m):
            # Primary: CAPEX - t*EBITDA[1] (== 0 at the true ratio optimum).
            # Tie-break: among solutions equally good on the primary term,
            # prefer higher Effective_Replacement (subtracting a small
            # bonus, since we are minimizing).
            return m.CAPEX - t * m.EBITDA[1] - EPSILON_PRIORITY_WEIGHT * m.Effective_Replacement
        m.Objective = pyo.Objective(rule=dinkelbach_objective_rule, sense=pyo.minimize)

    def solve(self, tee: bool = False):
        """
        Three stages, all pure LP/MIP-with-one-integer solves -- no Big-M,
        no extra binaries for the min() logic. See the long comment in
        Stage B/C below for why this is necessary at all: Dinkelbach's
        objective (Stage A) only cares about CAPEX and EBITDA[1] (plus a
        tiny epsilon nudge on Effective_Replacement) -- it never looks at
        Direct_Settled, Banked_to_Solar, or Banked_to_Normal individually.
        That means, once Stage A's financial optimum is found, every value
        of Direct_Settled between 0 and min(Generation, Consumption) is
        EQUALLY good to that objective -- nothing pushes it up to the true
        ceiling. Big-M/binary encodes the exact min() directly but turns
        every one of the (many) Dinkelbach LP solves into a slow MIP; the
        fix is to leave Stage A exactly as it was, then bolt on two more
        cheap LP passes afterward that lock in the financial result and
        squeeze Direct_Settled (then Banked_to_Solar/Normal) up to their
        true ceilings -- which are already hard upper bounds via the
        existing <=Generation / <=Consumption constraints, so maximizing
        them can never overshoot min(Generation, Consumption).
        """
        m = self.model
        self.dinkelbach_history = []

        def lock_tol(value):
            # Fixed tolerances break down across wildly different scales
            # (CAPEX/EBITDA ~1e2-1e4 vs. Direct_Settled sums ~1e7-1e8) --
            # scale to the quantity's own magnitude instead, same fix as
            # in the settlement-only script.
            return max(1e-6, abs(value) * 1e-6)

        # ---- Stage A: Dinkelbach's algorithm for Capex/EBITDA[1] ----
        t = 0.0  # iteration 0: equivalent to "minimize CAPEX" alone
        final_t = t
        for iteration in range(self.dinkelbach_max_iter):
            self._set_dinkelbach_objective(t)
            # NOTE: a fresh SolverFactory instance every iteration, not one
            # reused across the loop -- appsi_highs is a persistent solver
            # interface that keeps internal state (basis/factorization)
            # tied to the model. Deleting and rebuilding m.Objective on each
            # iteration confuses that persistent state enough that later
            # solves can spuriously report infeasible even though the LP
            # is fine (confirmed by reproducing it: the exact same t value,
            # solved standalone with a fresh solver, comes back optimal).
            solver = pyo.SolverFactory(self.solver_name)
            self.results = solver.solve(m, tee=tee)
            term = str(self.results.solver.termination_condition)
            if term not in ("optimal", "feasible"):
                raise RuntimeError(f"Solver did not reach optimality (termination={term}) at Dinkelbach iteration {iteration}.")

            capex_val = pyo.value(m.CAPEX)
            ebitda_val = pyo.value(m.EBITDA[1])
            if ebitda_val <= 0:
                raise RuntimeError(
                    f"EBITDA[1] came back <= 0 ({ebitda_val:.4f}) -- Capex/EBITDA is undefined here. "
                    "Check tariff/cost parameters."
                )
            f_of_t = capex_val - t * ebitda_val
            new_t = capex_val / ebitda_val
            self.dinkelbach_history.append({"iteration": iteration, "t": t, "CAPEX": capex_val,
                                             "EBITDA_1": ebitda_val, "F(t)": f_of_t})

            final_t = t
            if abs(f_of_t) < self.dinkelbach_tol:
                break
            t = new_t
        else:
            raise RuntimeError(f"Dinkelbach's algorithm did not converge in {self.dinkelbach_max_iter} iterations.")

        # The exact expression Stage A minimized, frozen with the converged
        # t -- used below to lock in "no worse than this" for later stages.
        stage_a_value = pyo.value(m.Objective)

        # Freeze the battery-sizing decision at Stage A's optimum. Without
        # this, Stages B/C could in principle pick a different `hours`
        # value that still (barely) satisfies the locked financial
        # tolerance below -- unlikely given how tight that tolerance is,
        # but not guaranteed, and leaving `hours` as a free integer would
        # also make Stages B/C solve as MIPs instead of plain LPs for no
        # benefit. Stage A already decided the battery size; Stages B/C
        # only refine how energy already produced gets settled.
        m.hours.fix(pyo.value(m.hours))

        # ---- Stage B: lock Stage A's financial optimum, then push every
        # block's Direct_Settled up to its true ceiling. Because we are
        # MINIMIZING in Stage A, "no worse than" means <=, not >= -- the
        # opposite direction from a maximize-objective lock. ----
        def lock_stage_a_rule(m):
            return (m.CAPEX - final_t * m.EBITDA[1] - EPSILON_PRIORITY_WEIGHT * m.Effective_Replacement
                    <= stage_a_value + lock_tol(stage_a_value))
        m.lock_stage_a = pyo.Constraint(rule=lock_stage_a_rule)

        m.del_component(m.Objective)

        def maximize_direct_settled_rule(m):
            return sum(m.Direct_Settled[month, block] for month in m.Months for block in m.ToD_Blocks)
        m.Objective = pyo.Objective(rule=maximize_direct_settled_rule, sense=pyo.maximize)

        solver = pyo.SolverFactory(self.solver_name)
        self.results = solver.solve(m, tee=tee)
        term = str(self.results.solver.termination_condition)
        if term not in ("optimal", "feasible"):
            raise RuntimeError(f"Stage B (direct settlement) solve did not reach optimality (termination={term}).")
        best_direct_settled = sum(
            pyo.value(m.Direct_Settled[month, block]) for month in m.Months for block in m.ToD_Blocks
        )

        # ---- Stage C: lock Stage B's direct-settlement total too, then
        # bank as much of whatever TRUE surplus remains as possible. ----
        m.lock_stage_b = pyo.Constraint(
            expr=sum(m.Direct_Settled[month, block] for month in m.Months for block in m.ToD_Blocks)
            >= best_direct_settled - lock_tol(best_direct_settled)
        )
        m.del_component(m.Objective)

        def maximize_banked_rule(m):
            return sum(m.Banked_to_Solar[month] + m.Banked_to_Normal[month] for month in m.Months)
        m.Objective = pyo.Objective(rule=maximize_banked_rule, sense=pyo.maximize)

        solver = pyo.SolverFactory(self.solver_name)
        self.results = solver.solve(m, tee=tee)
        term = str(self.results.solver.termination_condition)
        if term not in ("optimal", "feasible"):
            raise RuntimeError(f"Stage C (banking) solve did not reach optimality (termination={term}).")

        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def capex_to_ebitda_ratio(self, year: int = 1) -> float:
        m = self.model
        return pyo.value(m.CAPEX) / pyo.value(m.EBITDA[year])

    def effective_replacement(self) -> float:
        return pyo.value(self.model.Effective_Replacement)

    def financial_summary(self) -> pd.DataFrame:
        m = self.model
        rows = []
        for year in YEARS:
            rows.append({
                "Year": year,
                "Discharged_MnkWh": pyo.value(m.Discharged[year]),
                "Revenue_RsCr": pyo.value(m.Revenue[year]),
                "Costs_RsCr": pyo.value(m.Costs[year]),
                "EBITDA_RsCr": pyo.value(m.EBITDA[year]),
            })
        return pd.DataFrame(rows)

    def hourly_results(self) -> pd.DataFrame:
        if self._results_df is not None:
            return self._results_df
        m = self.model
        rows = []
        for t in m.T:
            rows.append({
                "Hour": t,
                "Month": self.hour_to_month[t],
                "ToD_Block": self.hour_to_tod[t],
                "Solar_Generation_kW": self.solar_gen[t],
                "Wind_Generation_kW": self.wind_gen[t],
                "Solar_Injected_kW": pyo.value(m.Solar_Injected[t]),
                "Wind_Injected_kW": pyo.value(m.Wind_Injected[t]),
                "BESS_Charged_kW": pyo.value(m.BESS_Charge[t]),
                "BESS_Discharged_kW": pyo.value(m.BESS_Discharge[t]),
                "BESS_SoC_kWh": pyo.value(m.SoC[t]),
                "Curtailed_Solar_kW": pyo.value(m.Curtailed_Solar[t]),
                "Curtailed_Wind_kW": pyo.value(m.Curtailed_Wind[t]),
                "Total_Grid_Injection_kW": pyo.value(m.Grid_Injected[t]),
            })
        self._results_df = pd.DataFrame(rows)
        return self._results_df

    def _replacement_table(self, numerator_getter) -> pd.DataFrame:
        """
        Builds a block x month table (+ Total row/column), matching the
        Master Spreadsheet's own "Replacement" / "Effective Replacement"
        layout: rows = ToD blocks, columns = Jan..Dec + Total.
        `numerator_getter(month, block)` supplies the settled/generated
        energy for that cell; the denominator is always consumption.
        """
        m = self.model
        data = {}
        for block in TOD_BLOCKS:
            row = {}
            num_total, den_total = 0.0, 0.0
            for month in range(1, 13):
                num = numerator_getter(month, block)
                den = self.consumption[(month, block)]
                row[MONTH_NAMES[month - 1]] = num / den
                num_total += num
                den_total += den
            row["Total"] = num_total / den_total
            data[block] = row
        df = pd.DataFrame(data).T
        df = df.reindex(TOD_BLOCKS)

        # Bottom "Total" row: sum across blocks per month / per month.
        total_row = {}
        for month in range(1, 13):
            num = sum(numerator_getter(month, block) for block in TOD_BLOCKS)
            den = sum(self.consumption[(month, block)] for block in TOD_BLOCKS)
            total_row[MONTH_NAMES[month - 1]] = num / den
        grand_num = sum(numerator_getter(month, block) for month in range(1, 13) for block in TOD_BLOCKS)
        grand_den = self.total_annual_consumption
        total_row["Total"] = grand_num / grand_den
        df.loc["Total"] = total_row
        return df

    def replacement_before_banking_table(self) -> pd.DataFrame:
        m = self.model
        return self._replacement_table(lambda month, block: pyo.value(m.Monthly_Generation[month, block]))

    def effective_replacement_table(self) -> pd.DataFrame:
        m = self.model
        return self._replacement_table(lambda month, block: pyo.value(m.Settled[month, block]))

    def summary_stats(self) -> dict:
        m = self.model
        hourly = self.hourly_results()
        generated_kwh = (hourly["Solar_Generation_kW"] + hourly["Wind_Generation_kW"]).sum()
        curtailed_kwh = (hourly["Curtailed_Solar_kW"] + hourly["Curtailed_Wind_kW"]).sum()
        discharged_kwh = hourly["Total_Grid_Injection_kW"].sum()  # post-curtailment, delivered to grid
        banked_settled_kwh = pyo.value(m.Total_Settled_Energy)
        lapsed_kwh = discharged_kwh - banked_settled_kwh

        return {
            "BESS_Storage_Hours": pyo.value(m.hours),
            "BESS_Capacity_kWh": pyo.value(m.BESS_Capacity),
            "Max_Grid_Solar_kW": self.max_grid_solar,
            "Max_Grid_Wind_kW": self.max_grid_wind,
            "Max_Grid_Allowable_kW": self.max_grid_allowable,
            "Max_SoC_Percent": self.max_soc_perc,
            "Min_SoC_Percent": self.min_soc_perc,
            "Round_Trip_Efficiency": self.RTE,
            "CAPEX_RsCr": pyo.value(m.CAPEX),
            "EBITDA_Year1_RsCr": pyo.value(m.EBITDA[1]),
            "Tariff_RsPerkWh": pyo.value(m.tariff),
            "Capex_to_EBITDA_Year1": self.capex_to_ebitda_ratio(1),
            "Effective_Replacement_Percent": self.effective_replacement() * 100,
            "Generated_MnkWh_P50": generated_kwh / 1_000_000.0,
            "Discharged_MnkWh": discharged_kwh / 1_000_000.0,
            "Curtailed_MnkWh": curtailed_kwh / 1_000_000.0,
            "Banked_and_Settled_MnkWh": banked_settled_kwh / 1_000_000.0,
            "Lapsed_MnkWh": lapsed_kwh / 1_000_000.0,
        }

    def export_to_excel(self, path: str):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Sheet 1: Hourly dispatch
            self.hourly_results().to_excel(writer, sheet_name="Hourly_Dispatch", index=False)

            # Sheet 2: Monthly settlement -- two stacked tables
            before = self.replacement_before_banking_table()
            after = self.effective_replacement_table()
            (before * 100).round(1).to_excel(
                writer, sheet_name="Monthly_Settlement", startrow=0,
                startcol=0, index=True, header=True,
            )
            title_ws = writer.sheets["Monthly_Settlement"]
            title_ws.cell(row=1, column=1, value="Replacement (before banking, %)")
            gap = len(before) + 3
            title_ws.cell(row=gap, column=1, value="Effective Replacement (after banking, %)")
            (after * 100).round(1).to_excel(
                writer, sheet_name="Monthly_Settlement", startrow=gap,
                startcol=0, index=True, header=True,
            )

            # Sheet 3: Summary
            stats = self.summary_stats()
            summary_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            self.financial_summary().to_excel(writer, sheet_name="Summary", index=False,
                                               startrow=len(summary_df) + 3)


if __name__ == "__main__":
    opt = BESSSettlementOptimizer("Master Spreadsheet.xlsx")
    opt.load_data().build_model().solve(tee=False)

    print("Solver termination:", opt.results.solver.termination_condition)
    print("\nDinkelbach convergence history:")
    for row in opt.dinkelbach_history:
        print(f"  iter {row['iteration']}: t={row['t']:.6f}  CAPEX={row['CAPEX']:.2f}  "
              f"EBITDA[1]={row['EBITDA_1']:.2f}  F(t)={row['F(t)']:.6f}")

    stats = opt.summary_stats()
    print("\nSummary stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nReplacement BEFORE banking (%):")
    print((opt.replacement_before_banking_table() * 100).round(1))

    print("\nEffective Replacement AFTER banking (%):")
    print((opt.effective_replacement_table() * 100).round(1))

    opt.export_to_excel("Optimised_CAPEXEBITDA_Schedule.xlsx")
    print("\nSaved: Optimized_BESS_Settlement_Schedule.xlsx")