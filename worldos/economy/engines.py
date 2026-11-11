"""Monthly engines of the Alpha economy.

Behavioural rules follow the macro agent-based literature, in particular the
CATS family of models (Delli Gatti, Desiderio, Gaffeo, Cirillo & Gallegati,
"Macroeconomics from the Bottom-up", 2011; Assenza, Delli Gatti & Grazzini,
JEDC 2015) for firm price/quantity adaptation (applied here to the markup of a cost-plus
price, as in Dosi, Fagiolo & Roventini, JEDC 2010), and Lengnick (JEBO 2013) for
wage adaptation to vacancy signals. Household consumption is the
income-and-wealth consumption function used in stock-flow consistent models
(Godley & Lavoie, "Monetary Economics", 2007). Rule parameters are
assumptions (see config.py); Alpha does not claim to replicate those papers.

Monthly sequence (engine order):
  20 planning   firms update demand expectations, prices, wage offers, labour demand
  30 credit     firms finance planned wage bills; bank rations by capital
  40 labour     separations, firing of excess staff, search & matching
  50 production output = min(productivity x effective labour, kappa x capital); depreciation
  60 income     wages, income tax, benefits, pensions, deposit interest
  70 goods      household, firm-investment and government purchases with rationing
  80 finance    debt service, corporate tax, dividends, bankruptcy, entry
  85 expect.    households update expected income
"""

from __future__ import annotations

import numpy as np

from worldos.core.engine import Engine
from worldos.economy import ledger
from worldos.population.tables import EMPLOYED, RETIRED, UNEMPLOYED

EPS = 1e-12


def edu_productivity(world) -> np.ndarray:
    return np.asarray(world.config.economy.education_productivity, dtype=float)


def employed_index(world) -> np.ndarray:
    p = world.persons
    return np.flatnonzero(p.alive & (p.status == EMPLOYED))


def effective_labour(world) -> np.ndarray:
    p, f = world.persons, world.firms
    emp = employed_index(world)
    eff = p.weight[emp] * edu_productivity(world)[p.edu[emp]]
    return np.bincount(p.employer[emp], weights=eff, minlength=len(f))


def mean_wage_per_worker(world) -> float:
    """Weighted mean gross monthly wage offer per employed person."""
    p, f = world.persons, world.firms
    emp = employed_index(world)
    if not len(emp):
        return float(np.mean(f.wage_offer[f.alive])) if f.alive.any() else 0.0
    w = f.wage_offer[p.employer[emp]] * edu_productivity(world)[p.edu[emp]]
    return float(np.average(w, weights=p.weight[emp]))


def capital_productivity(world) -> float:
    """kappa: monthly output per unit of capital at full utilisation."""
    ec = world.config.economy
    return 1.0 / (12.0 * ec.capital_output_ratio * ec.target_utilisation)


def capital_value(world, idx=slice(None)) -> np.ndarray:
    return world.firms.capital[idx] * average_price(world)


def firm_net_worth(world, idx) -> np.ndarray:
    f = world.firms
    return f.deposits[idx] + f.inventory[idx] * f.unit_cost()[idx] + capital_value(world, idx) - f.loans[idx]


def average_price(world) -> float:
    f = world.firms
    a = f.alive
    units = f.sales_units[a].sum()
    return float(f.sales_value[a].sum() / units) if units > EPS else float(np.mean(f.price[a]))


class FirmPlanningEngine(Engine):
    name = "firm_planning"
    order = 20

    def step(self, world) -> None:
        f, ec = world.firms, world.config.economy
        rng = world.rng.get(self.name)
        a = np.flatnonzero(f.alive)
        if not len(a):
            return
        pbar = average_price(world)
        sales, inv, price, exp = f.sales_units[a], f.inventory[a], f.price[a], f.exp_demand[a]
        uq = rng.random(len(a)) * ec.quantity_step
        um = rng.random(len(a)) * ec.markup_step
        uw = rng.random(len(a)) * ec.wage_step
        # Demand signals relative to the firm's target inventory buffer.
        ref = ec.inventory_buffer * np.maximum(sales, EPS)
        short = inv < 0.5 * ref  # excess demand
        glut = inv > 1.5 * ref  # excess supply
        base = ec.expectation_smoothing * sales + (1 - ec.expectation_smoothing) * exp
        base = np.where(short, np.maximum(base, sales), base)
        # CATS rule: the firm's price relative to the market decides whether it adjusts its
        # quantity or its price; here the price lever is the markup over unit labour cost.
        new_exp = np.where(short & (price >= pbar), base * (1 + uq), np.where(glut & (price <= pbar), base * (1 - uq), base))
        markup = f.markup[a]
        markup = np.where(short & (price < pbar), markup + um, np.where(glut & (price > pbar), markup - um, markup))
        markup += ec.markup_reversion * (ec.initial_markup - markup)  # competitive pressure anchors markups
        lo, hi = ec.markup_bounds
        f.markup[a] = np.clip(markup, lo, hi)
        # Wage norm (cf. Dosi et al. 2010): indexation to positive past inflation and the anchored target, plus trend
        # productivity growth; firm-specific adjustment to vacancy (Lengnick 2013) and layoff signals,
        # with downward nominal rigidity (cuts are scaled down).
        # expectations: weight wage_indexation on past inflation, the rest anchored on the CB target
        target_m = world.config.central_bank.inflation_target / 12
        norm = ec.wage_indexation * max(recent_monthly_inflation(world), 0.0) + (1 - ec.wage_indexation) * target_m \
            + ec.wage_productivity_passthrough * ec.productivity_growth / 12
        wage = f.wage_offer[a] * (1 + norm)
        wage = np.where(f.unfilled[a] > EPS, wage * (1 + uw),
                        np.where(f.fired[a], wage * (1 - ec.downward_wage_flexibility * uw), wage))
        f.wage_offer[a] = wage
        # full-cost pricing: labour cost + depreciation per unit of output at normal utilisation
        f.capital_cost[a] = ec.depreciation_rate * ec.capital_output_ratio * pbar
        f.price[a] = (1 + f.markup[a]) * (wage / f.productivity[a] + f.capital_cost[a])
        f.exp_demand[a] = new_exp
        kappa = capital_productivity(world)
        capacity = kappa * f.capital[a]
        f.target_output[a] = np.minimum(np.maximum(0.0, new_exp * (1 + ec.inventory_buffer) - inv), capacity)
        f.labour_demand[a] = f.target_output[a] / f.productivity[a]
        # Investment: replacement + partial adjustment towards desired capital (flexible accelerator),
        # scaled by the gap between the return on capital and the loan rate.
        sm = ec.investment_demand_smoothing
        f.normal_demand[a] = (1 - sm) * f.normal_demand[a] + sm * sales
        desired = f.normal_demand[a] / (kappa * ec.target_utilisation)
        k_value = np.maximum(f.capital[a] * pbar, EPS)
        r_profit = 12 * f.profit[a] / k_value
        m = np.clip(1 + ec.investment_rate_sensitivity * (r_profit - f.loan_rate[a]), 0.0, 2.0)
        units = m * np.maximum(0.0, ec.depreciation_rate / 12 * f.capital[a] + ec.investment_adjustment * (desired - f.capital[a]))
        f.invest_budget[a] = units * pbar


class CreditEngine(Engine):
    name = "credit"
    order = 30

    def step(self, world) -> None:
        f, bc, fl = world.firms, world.config.bank, world.flows
        a = np.flatnonzero(f.alive)
        if not len(a):
            return
        policy = world.central_bank.policy_rate
        nw = firm_net_worth(world, a)
        leverage = f.loans[a] / np.maximum(f.loans[a] + np.maximum(nw, 0.0), EPS)
        f.loan_rate[a] = policy + bc.loan_spread + bc.risk_premium * np.clip(leverage, 0, 1)
        debt_service = f.loans[a] * (f.loan_rate[a] / 12 + 1 / bc.amortization_months)
        planned_bill = f.wage_offer[a] * f.labour_demand[a]
        need = np.maximum(0.0, planned_bill + debt_service + f.invest_budget[a] - f.deposits[a])
        revenue_base = np.maximum(f.sales_value[a], f.price[a] * f.exp_demand[a])
        debt_limit = np.maximum(bc.max_debt_to_sales_months * revenue_base,
                                world.config.economy.max_debt_to_capital * capital_value(world, a))
        firm_cap = np.maximum(0.0, debt_limit - f.loans[a])
        request = np.minimum(need, firm_cap)
        capacity = bank_lending_capacity(world)
        total = float(request.sum())
        scale = 1.0 if total <= capacity else (capacity / total if total > 0 else 0.0)
        grant = request * scale
        fl.credit_rationed = float(need.sum() - grant.sum())
        fl.new_loans = ledger.lend_to_firms(world, a, grant)
        # firms that remain short of finance scale down planned hiring
        affordable = np.maximum(0.0, f.deposits[a] - debt_service) / f.wage_offer[a]
        f.labour_demand[a] = np.minimum(f.labour_demand[a], affordable)
        # investment only from funds left after wages and debt service (wages have priority)
        spare = np.maximum(0.0, f.deposits[a] - debt_service - f.wage_offer[a] * f.labour_demand[a])
        f.invest_budget[a] = np.minimum(f.invest_budget[a], spare)


class LabourMarketEngine(Engine):
    name = "labour_market"
    order = 40

    def step(self, world) -> None:
        p, f, ec, fl = world.persons, world.firms, world.config.economy, world.flows
        rng = world.rng.get(self.name)
        h_eff = edu_productivity(world)
        f.fired = False

        # exogenous separations
        emp = employed_index(world)
        sep = emp[rng.random(len(emp)) < ec.separation_rate]
        p.status[sep] = UNEMPLOYED
        p.employer[sep] = -1
        fl.separations = float(p.weight[sep].sum())

        # firing of excess labour (no overshoot below demand)
        labour = effective_labour(world)
        excess = labour - f.labour_demand
        excess[~f.alive] = labour[~f.alive]  # dead firms keep nobody
        over = excess > ec.firing_tolerance * np.maximum(f.labour_demand, EPS)
        emp = employed_index(world)
        cand = emp[over[p.employer[emp]]]
        if len(cand):
            key = rng.random(len(cand))
            order = np.lexsort((key, p.employer[cand]))
            cand = cand[order]
            eff = p.weight[cand] * h_eff[p.edu[cand]]
            firm = p.employer[cand]
            cum = np.cumsum(eff)
            start = np.r_[0, np.flatnonzero(np.diff(firm)) + 1]
            group_start_cum = np.repeat(cum[start] - eff[start], np.diff(np.r_[start, len(cand)]))
            within = cum - group_start_cum
            fire = cand[within <= excess[firm] + EPS]
            f.fired[np.unique(p.employer[fire])] = True
            p.status[fire] = UNEMPLOYED
            p.employer[fire] = -1

        # search and matching
        labour = effective_labour(world)
        vac = np.where(f.alive, np.maximum(0.0, f.labour_demand - labour), 0.0)
        applicants = np.flatnonzero(p.alive & (p.status == UNEMPLOYED))
        applicants = applicants[rng.permutation(len(applicants))]
        hires = 0.0
        for _ in range(ec.market_rounds):
            open_f = np.flatnonzero(vac > EPS)
            if not len(open_f) or not len(applicants):
                break
            sample = open_f[rng.integers(0, len(open_f), size=(len(applicants), ec.search_breadth))]
            choice = sample[np.arange(len(applicants)), np.argmax(f.wage_offer[sample], axis=1)]
            eff = p.weight[applicants] * h_eff[p.edu[applicants]]
            order = np.argsort(choice, kind="stable")  # stable keeps the random applicant order
            ch, ef, ap = choice[order], eff[order], applicants[order]
            cum = np.cumsum(ef)
            start = np.r_[0, np.flatnonzero(np.diff(ch)) + 1]
            before = cum - ef - np.repeat(cum[start] - ef[start], np.diff(np.r_[start, len(ch)]))
            accept = before < vac[ch] - EPS
            hired = ap[accept]
            p.status[hired] = EMPLOYED
            p.employer[hired] = ch[accept]
            np.subtract.at(vac, ch[accept], ef[accept])
            hires += float(p.weight[hired].sum())
            applicants = ap[~accept]
            applicants = applicants[rng.permutation(len(applicants))]
        f.unfilled = np.maximum(vac, 0.0)
        fl.hires = hires


class ProductionEngine(Engine):
    name = "production"
    order = 50

    def step(self, world) -> None:
        f, ec, fl = world.firms, world.config.economy, world.flows
        capacity = capital_productivity(world) * f.capital
        out = np.where(f.alive, np.minimum(f.productivity * effective_labour(world), capacity), 0.0)
        f.output_units = out
        f.inventory += out
        fl.units_produced = float(out.sum())
        # depreciation of installed capital (all slots: capital survives firm turnover)
        dep = f.capital * ec.depreciation_rate / 12
        f.capital -= dep
        f.depreciation_value = dep * average_price(world)
        fl.depreciation_units = float(dep.sum())
        fl.depreciation_value = float(f.depreciation_value.sum())


class IncomeEngine(Engine):
    name = "income"
    order = 60

    def step(self, world) -> None:
        p, f, h, fl = world.persons, world.firms, world.households, world.flows
        gov, gc = world.government, world.config.government
        h_eff = edu_productivity(world)
        avg_wage = mean_wage_per_worker(world)

        # wages (firms pay what they can; a shortfall flags distress)
        emp = employed_index(world)
        employer = p.employer[emp]
        gross = f.wage_offer[employer] * h_eff[p.edu[emp]] * p.weight[emp]
        bill = np.bincount(employer, weights=gross, minlength=len(f))
        # working-capital credit line covers payroll shortfalls if the bank has capital headroom
        gap = np.where(f.alive, np.maximum(0.0, bill - np.maximum(f.deposits, 0.0)), 0.0)
        if gap.sum() > 0:
            grant = gap * min(1.0, bank_lending_capacity(world) / gap.sum())
            fl.working_capital_loans = ledger.lend_to_firms(world, np.arange(len(f)), grant)
            fl.new_loans += fl.working_capital_loans
        ratio = np.ones(len(f))
        pos = bill > EPS
        ratio[pos] = np.minimum(1.0, np.maximum(f.deposits[pos], 0.0) / bill[pos])
        f.distressed = ratio < 1.0 - 1e-12
        paid = gross * ratio[employer]
        f.wage_bill = np.bincount(employer, weights=paid, minlength=len(f))
        fl.wages_paid = float(f.wage_bill.sum())
        fl.wages_received = ledger.firms_pay_households(world, employer, p.hh[emp], paid)

        # progressive-by-allowance flat income tax on wages
        allowance = gc.tax_allowance_ratio * avg_wage
        per_person = paid / p.weight[emp]
        tax = gov.income_tax_rate * np.maximum(0.0, per_person - allowance) * p.weight[emp]
        fl.income_tax = ledger.households_pay_government(world, p.hh[emp], tax)

        # transfers indexed to the mean wage
        unemp = np.flatnonzero(p.alive & (p.status == UNEMPLOYED))
        fl.unemployment_benefits = ledger.government_pays_households(
            world, p.hh[unemp], gov.unemployment_replacement * avg_wage * p.weight[unemp])
        ret = np.flatnonzero(p.alive & (p.status == RETIRED))
        fl.pensions = ledger.government_pays_households(
            world, p.hh[ret], gov.pension_replacement * avg_wage * p.weight[ret])

        # deposit interest
        rate = max(0.0, world.central_bank.policy_rate - world.config.bank.deposit_markdown) / 12
        ih, jf = ledger.bank_pays_deposit_interest(world, rate)
        fl.deposit_interest = ih + jf
        fl.reserve_interest, fl.gov_interest = ledger.settle_central_bank_interest(
            world, world.central_bank.policy_rate / 12)


class GoodsMarketEngine(Engine):
    name = "goods_market"
    order = 70

    def step(self, world) -> None:
        f, h, ec, fl = world.firms, world.households, world.config.economy, world.flows
        rng = world.rng.get(self.name)
        f.sales_units = 0.0
        f.sales_value = 0.0

        buyers = np.flatnonzero(h.alive & (h.n_members > 0))
        cb = world.config.central_bank
        real_deposit_rate = max(0.0, world.central_bank.policy_rate - world.config.bank.deposit_markdown) \
            - 12 * recent_monthly_inflation(world)
        c_y = float(np.clip(ec.consumption_income_propensity
                            - ec.consumption_rate_sensitivity * (real_deposit_rate - cb.neutral_real_rate), 0.5, 1.0))
        budget = c_y * np.maximum(h.income_exp[buyers], 0.0) \
            + ec.consumption_wealth_propensity * np.maximum(h.deposits[buyers], 0.0)
        budget = np.clip(budget, 0.0, np.maximum(h.deposits[buyers], 0.0))
        spent = np.zeros(len(buyers))
        for _ in range(ec.market_rounds):
            active = np.flatnonzero(budget - spent > 1e-9)
            sellers = np.flatnonzero(f.alive & (f.inventory > 1e-9))
            if not len(active) or not len(sellers):
                break
            sample = sellers[rng.integers(0, len(sellers), size=(len(active), ec.search_breadth))]
            choice = sample[np.arange(len(active)), np.argmin(f.price[sample], axis=1)]
            want_units = (budget[active] - spent[active]) / f.price[choice]
            demand = np.bincount(choice, weights=want_units, minlength=len(f))
            fill = np.ones(len(f))
            short = demand > f.inventory
            fill[short] = f.inventory[short] / demand[short]
            units = want_units * fill[choice]
            value = units * f.price[choice]
            sold_units = np.bincount(choice, weights=units, minlength=len(f))
            f.inventory = np.maximum(f.inventory - sold_units, 0.0)
            f.sales_units += sold_units
            f.sales_value += np.bincount(choice, weights=value, minlength=len(f))
            ledger.households_pay_firms(world, buyers[active], choice, value)
            spent[active] += value
            fl.consumption_units += float(units.sum())
        h.consumption_last = 0.0
        h.consumption_last[buyers] = spent
        fl.consumption = float(spent.sum())

        # investment goods: each investing firm's budget is spread over sellers in proportion to their stock
        f.investment_units = 0.0
        f.investment_value = 0.0
        inv_budget = np.where(f.alive, f.invest_budget, 0.0)
        total_inv = float(inv_budget.sum())
        stock_value = np.where(f.alive, f.inventory * f.price, 0.0)
        total_stock = float(stock_value.sum())
        if total_inv > EPS and total_stock > EPS:
            fill = min(1.0, total_stock / total_inv)
            spend = inv_budget * fill  # per buyer
            seller_share = stock_value / total_stock
            sales = spend.sum() * seller_share  # per seller (value)
            units_sold = sales / np.where(f.price > 0, f.price, 1.0)
            ledger.settle_between_firms(world, spend, sales)
            f.inventory = np.maximum(f.inventory - units_sold, 0.0)
            f.sales_units += units_sold
            f.sales_value += sales
            avg_unit_price = sales.sum() / max(units_sold.sum(), EPS)
            f.investment_value = spend
            f.investment_units = spend / avg_unit_price
            f.capital += f.investment_units
            fl.investment = float(spend.sum())
            fl.investment_units = float(units_sold.sum())

        # public consumption, spread over firms in proportion to available stock
        g_budget = world.government.spending_share * reference_monthly_gdp(world)
        stock_value = np.where(f.alive, f.inventory * f.price, 0.0)
        total_stock = float(stock_value.sum())
        if g_budget > 0 and total_stock > EPS:
            spend = stock_value * min(1.0, g_budget / total_stock)
            units = spend / f.price
            f.inventory = np.maximum(f.inventory - units, 0.0)
            f.sales_units += units
            f.sales_value += spend
            idx = np.arange(len(f))
            fl.gov_purchases = ledger.government_pays_firms(world, idx, spend)
            fl.gov_purchase_units = float(units.sum())
        fl.firm_revenue = float(f.sales_value.sum())
        fl.units_sold = float(f.sales_units.sum())
        fl.inventory_change_value = float(((f.output_units - f.sales_units) * f.unit_cost()).sum())


def recent_monthly_inflation(world) -> float:
    """Average monthly CPI inflation over the last 12 published months (0 before history exists)."""
    m = world.metrics
    if len(m) < 2:
        return 0.0
    window = m[-13:]
    return float((window[-1]["cpi"] / window[0]["cpi"]) ** (1.0 / (len(window) - 1)) - 1.0)


def bank_lending_capacity(world) -> float:
    f, h = world.firms, world.households
    loans = float(f.loans[f.alive].sum() + h.loans[h.alive].sum())
    return max(0.0, world.bank.equity / world.config.bank.capital_requirement - loans)


def reference_monthly_gdp(world) -> float:
    """Trailing 12-month mean nominal GDP (falls back to planned output value before history exists)."""
    hist = [m["gdp_nominal"] for m in world.metrics[-12:]]
    if hist:
        return float(np.mean(hist))
    f = world.firms
    return float((f.price * f.exp_demand)[f.alive].sum())


class FirmFinanceEngine(Engine):
    name = "firm_finance"
    order = 80

    def step(self, world) -> None:
        f, fl, bc, ec = world.firms, world.flows, world.config.bank, world.config.economy
        gov = world.government
        a = np.flatnonzero(f.alive)

        # debt service: interest first, then amortisation
        interest = f.loans[a] * f.loan_rate[a] / 12
        principal = f.loans[a] / bc.amortization_months
        cash = np.maximum(f.deposits[a], 0.0)
        int_paid = np.minimum(cash, interest)
        prin_paid = np.minimum(cash - int_paid, principal)
        fl.loan_interest = ledger.firms_pay_interest(world, a, int_paid)
        fl.repayments = ledger.firms_repay(world, a, prin_paid)
        f.interest_paid[a] = int_paid
        f.distressed[a] |= (int_paid + prin_paid) < (interest + principal) - 1e-9

        # profit, tax, dividends
        profit = f.sales_value[a] - f.wage_bill[a] - int_paid + f.deposit_interest[a] - f.depreciation_value[a]
        f.profit[a] = profit
        tax = np.minimum(gov.corporate_tax_rate * np.maximum(profit, 0.0), np.maximum(f.deposits[a], 0.0))
        fl.corporate_tax = ledger.firms_pay_government(world, a, tax)
        buffer = ec.liquidity_buffer_months * f.wage_bill[a] + f.loans[a] * (f.loan_rate[a] / 12 + 1 / bc.amortization_months)
        # dividends come out of after-tax profit, never out of borrowed funds or required liquidity
        net_worth = firm_net_worth(world, a)
        distributable = np.minimum.reduce([profit - tax, f.deposits[a] - buffer, net_worth - buffer])
        div = ec.dividend_payout * np.maximum(0.0, distributable)
        per_firm = np.zeros(len(f))
        per_firm[a] = div
        fl.firm_dividends = float(div.sum())
        ledger.distribute_to_owners(world, "firms", fl.firm_dividends, per_firm)

        self._bankruptcies(world)
        self._entry(world)

        # bank distributes capital above its target ratio
        loans = float(f.loans[f.alive].sum())
        excess = world.bank.equity - bc.equity_target * loans
        if excess > 0:
            payout = 0.1 * excess
            ledger.distribute_to_owners(world, "bank", payout)
            fl.bank_dividends = payout

    @staticmethod
    def _bankruptcies(world) -> None:
        f, p, fl = world.firms, world.persons, world.flows
        a = np.flatnonzero(f.alive)
        nw = firm_net_worth(world, a)
        bust = a[f.distressed[a] | (nw < 0)]
        if not len(bust):
            return
        recover = np.minimum(np.maximum(f.deposits[bust], 0.0), f.loans[bust])
        fl.repayments += ledger.firms_repay(world, bust, recover)
        fl.write_offs = ledger.write_off_firm_loans(world, bust)
        leftover = np.maximum(f.deposits[bust], 0.0)
        per_firm = np.zeros(len(f))
        per_firm[bust] = leftover
        ledger.distribute_to_owners(world, "firms", float(leftover.sum()), per_firm)
        fl.units_written_off = float(f.inventory[bust].sum())
        f.inventory[bust] = 0.0
        f.alive[bust] = False
        f.dead_months[bust] = 0
        workers = np.flatnonzero(p.alive & (p.status == EMPLOYED) & np.isin(p.employer, bust))
        p.status[workers] = UNEMPLOYED
        p.employer[workers] = -1
        fl.bankruptcies = int(len(bust))

    @staticmethod
    def _entry(world) -> None:
        f, ec, bc, fl = world.firms, world.config.economy, world.config.bank, world.flows
        dead = np.flatnonzero(~f.alive)
        f.dead_months[dead] += 1
        ready = dead[f.dead_months[dead] > ec.entry_delay_months]
        alive = f.alive
        if not len(ready) or not alive.any():
            return
        size = 0.5 * float(np.median(f.labour_demand[alive]))
        wage = float(np.mean(f.wage_offer[alive]))
        startup = (ec.liquidity_buffer_months + 1.0) * wage * size
        n = int(min(len(ready), bank_lending_capacity(world) // max(startup, EPS)))
        if n <= 0:
            return
        new = ready[:n]
        f.alive[new] = True
        f.age_m[new] = 0
        # an entrant takes over the slot's existing capital (assets survive restructuring); any further
        # capital must be bought through investment, so capital is never created outside the goods market
        f.wage_offer[new] = wage
        f.productivity[new] = float(np.mean(f.productivity[alive]))
        f.markup[new] = float(np.median(f.markup[alive]))
        f.capital_cost[new] = float(np.median(f.capital_cost[alive]))
        f.price[new] = (1 + f.markup[new]) * (wage / f.productivity[new] + f.capital_cost[new])
        f.exp_demand[new] = 0.5 * float(np.median(f.exp_demand[alive]))
        f.normal_demand[new] = f.exp_demand[new]
        f.inventory[new] = 0.0
        f.unfilled[new] = 0.0
        f.fired[new] = False
        f.deposits[new] = 0.0
        f.loan_rate[new] = world.central_bank.policy_rate + bc.loan_spread
        f.sales_units[new] = 0.0
        f.sales_value[new] = 0.0
        fl.new_loans += ledger.lend_to_firms(world, new, np.full(n, startup))
        fl.entries = n


class HouseholdExpectationsEngine(Engine):
    name = "household_expectations"
    order = 85

    def step(self, world) -> None:
        h = world.households
        w = world.config.economy.income_expectation_weight
        h.income_last = h.income_cur
        h.income_exp = np.where(h.alive, (1 - w) * h.income_exp + w * h.income_cur, 0.0)


class TechnologyEngine(Engine):
    """Exogenous labour-augmenting technical progress (Solow-neutral), firm-level noise.

    A placeholder for the endogenous innovation engine is NOT implied: Alpha
    treats productivity growth as an explicit scenario assumption.
    """

    name = "technology"
    order = 15

    def step(self, world) -> None:
        f, ec = world.firms, world.config.economy
        rng = world.rng.get(self.name)
        a = f.alive
        shock = rng.normal(0.0, ec.productivity_noise, size=int(a.sum()))
        f.productivity[a] *= np.exp(ec.productivity_growth / 12 + shock)
        f.age_m[a] += 1
