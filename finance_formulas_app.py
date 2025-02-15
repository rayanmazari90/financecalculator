import streamlit as st
import numpy as np
import math
import sympy as sp  
####################################
# Helper Functions 
# (Time Value, Bonds, Capital Budgeting, Stock Valuation)
####################################

def future_value(present_value, r, n):
    return present_value * (1 + r)**n

def present_value(fv, r, n):
    return fv / ((1 + r)**n)

def annuity_pv(C, r, n):
    return C * (1 - (1 + r)**(-n)) / r

def annuity_fv(C, r, n):
    return C * ((1 + r)**n - 1) / r

def growing_annuity_pv(C, r, g, n):
    if abs(r - g) < 1e-12:  # special case if r ≈ g
        return C * n / (1 + r)
    return C * (1 - ((1 + g)/(1 + r))**n) / (r - g)

def perpetuity_pv(C, r):
    return C / r

def growing_perpetuity_pv(C, r, g):
    if r <= g:
        return None
    return C / (r - g)

def npv_calc(cash_flows, r):
    total = 0.0
    for i, cf in enumerate(cash_flows):
        total += cf / ((1 + r)**i)
    return total

def irr_calc(cash_flows, guess=0.1, max_iterations=100, tolerance=1e-6):
    def npv_rate(rate):
        total = 0.0
        for i, cf in enumerate(cash_flows):
            total += cf / ((1 + rate)**i)
        return total
    
    rate0 = guess
    rate1 = guess + 0.01  
    npv0 = npv_rate(rate0)
    for _ in range(max_iterations):
        npv1 = npv_rate(rate1)
        if (npv1 - npv0) == 0:
            break
        rate2 = rate1 - npv1 * (rate1 - rate0)/(npv1 - npv0)
        if abs(rate2 - rate1) < tolerance:
            return rate2
        rate0, rate1 = rate1, rate2
        npv0 = npv1
    return rate1

def payback_period(cash_flows):
    cumulative = 0.0
    for i, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            return i
    return None

def discounted_payback_period(cash_flows, r):
    cumulative = 0.0
    for i, cf in enumerate(cash_flows):
        discounted_cf = cf / ((1 + r)**i)
        cumulative += discounted_cf
        if cumulative >= 0:
            return i
    return None

# ---- BOND Helpers ----
def bond_price_coupon(face_value, coupon_rate, years_to_maturity, coupons_per_year, ytm_annual):
    C = (coupon_rate * face_value) / coupons_per_year
    r = ytm_annual / coupons_per_year
    T = int(coupons_per_year * years_to_maturity)
    price = annuity_pv(C, r, T) + present_value(face_value, r, T)
    return price

def bond_ytm_coupon(face_value, coupon_rate, years_to_maturity, coupons_per_year, bond_price):
    # CF0 = -bond_price, CF1..CF(T-1) = C, CF(T) = C + face_value
    C = (coupon_rate * face_value) / coupons_per_year
    T = int(coupons_per_year * years_to_maturity)
    cf_list = [-bond_price] + [C]*(T-1) + [C + face_value]
    period_irr = irr_calc(cf_list, guess=0.05)
    annual_irr = period_irr * coupons_per_year
    return period_irr, annual_irr

def bond_coupon_rate(bond_price, face_value, annual_ytm, years_to_maturity, coupons_per_year):
    r = annual_ytm / coupons_per_year
    T = coupons_per_year * years_to_maturity
    discounted_face = present_value(face_value, r, T)
    annuity_factor = annuity_pv(1, r, T)
    numerator = bond_price - discounted_face
    denominator = (face_value / coupons_per_year) * annuity_factor
    if abs(denominator) < 1e-12:
        return None
    c = numerator / denominator
    return c

# ---- STOCK Valuation Helpers ----
def stock_price_constant_dividend(dividend, r):
    # P0 = D / r
    return dividend / r

def stock_price_constant_growth(D0, r, g):
    # P0 = D0*(1+g)/(r - g)
    if r <= g:
        return None
    return (D0 * (1 + g)) / (r - g)

def stock_required_return_gordon(D1, P0, g):
    # r = (D1 / P0) + g
    return (D1 / P0) + g

def stock_price_nonconstant(dividends, r, g, start_growth_year):
    # 1) Present value of each explicit dividend
    pv_sum = 0.0
    for i, div in enumerate(dividends):
        year = i + 1
        pv_sum += div / ((1 + r)**year)
    # 2) Terminal Value at 'start_growth_year'
    #    TVn = D_{n+1} / (r - g)
    Dn = dividends[-1]
    D_next = Dn * (1 + g)
    if r <= g:
        return None
    tv = D_next / (r - g)
    pv_tv = tv / ((1 + r)**start_growth_year)
    return pv_sum + pv_tv

def corporate_value_model(fcf_list, r, g, debt, shares):
    # PV of explicit FCF
    pv_sum = 0.0
    for i, fcf in enumerate(fcf_list):
        year = i + 1
        pv_sum += fcf / ((1 + r)**year)
    if len(fcf_list) == 0 or r <= g:
        return None
    # Terminal value at end of last forecast year
    FCFn = fcf_list[-1]
    FCF_next = FCFn * (1 + g)
    tv = FCF_next / (r - g)
    N = len(fcf_list)
    pv_tv = tv / ((1 + r)**N)
    mv_firm = pv_sum + pv_tv
    mv_equity = mv_firm - debt
    return mv_equity / shares

def pe_multiple_valuation(eps, pe_ratio):
    return eps * pe_ratio

def free_cash_flow(ebit, tax_rate, depreciation, capex, delta_nwc):
    return ebit * (1 - tax_rate) + depreciation - capex - delta_nwc

####################################
# App Layout with Categorization
####################################

st.title("Finance Formulas App")

st.write("""
Use the selectors below to navigate formulas by **category**. 
Then pick a specific **formula** to reveal the inputs and outputs.
""")

# 1) Define categories and formulas in a dictionary
categories = {
    "Time Value of Money": [
        "Future Value (FV)",
        "Present Value (PV)",
        "Annuity (PV)",
        "Annuity (FV)",
        "Growing Annuity (PV)",
        "Perpetuity",
        "Growing Perpetuity"
    ],
    "Bond": [
        "Bond Price (Coupon)",
        "Bond YTM (Coupon)",
        "Bond Coupon Rate",
    ],
    "Capital Budgeting": [
        "Net Present Value (NPV)",
        "Internal Rate of Return (IRR)",
        "Payback Period",
        "Discounted Payback Period"
    ],
    "Stock Valuation": [
        "Stock - Constant Dividend Price",
        "Stock - Constant Growth Dividend Price (Gordon Growth)",
        "Stock - Required Return (Gordon Growth)",
        "Stock - Non-Constant Growth Dividend Price",
        "Corporate Value (FCF) Model",
        "Stock Price from PE Multiple",
        "Free Cash Flow (FCF)"
    ]
}

# 2) Let user select a category first
selected_category = st.selectbox("Select a Category:", list(categories.keys()))

# 3) Let user select a formula within that category
selected_formula = st.selectbox("Select a Formula:", categories[selected_category])

st.write("---")

####################################
# Time Value of Money
####################################

if selected_formula == "Future Value (FV)":
    st.subheader("Future Value")
    st.latex(r"\text{FV} = \text{PV} \times (1+r)^n")
    pv = st.number_input("Present Value", value=1000.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.10, 
                        step=1e-4, format="%.4f")
    n = st.number_input("Number of Periods (n)", value=5, step=1)
    if st.button("Calculate FV"):
        fv = future_value(pv, r, n)
        st.success(f"Future Value = {fv:,.6f}")

elif selected_formula == "Present Value (PV)":
    st.subheader("Present Value")
    st.latex(r"\text{PV} = \frac{\text{FV}}{(1+r)^n}")
    fv_amt = st.number_input("Future Value", value=1000.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", value=0.10, 
                        step=1e-4, format="%.4f")
    n = st.number_input("Number of Periods (n)", value=5, step=1)
    if st.button("Calculate PV"):
        pv = present_value(fv_amt, r, n)
        st.success(f"Present Value = {pv:,.6f}")

elif selected_formula == "Annuity (PV)":
    st.subheader("Present Value of an Annuity")
    st.latex(r"\text{PV}_{\text{annuity}} = C \times \frac{1 - (1+r)^{-n}}{r}")
    C = st.number_input("Regular Payment (C)", value=500.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", value=0.08, 
                        step=1e-4, format="%.4f")
    n = st.number_input("Number of Periods", value=10, step=1)
    if st.button("Calculate PV Annuity"):
        result = annuity_pv(C, r, n)
        st.success(f"PV of Annuity = {result:,.6f}")

elif selected_formula == "Annuity (FV)":
    st.subheader("Future Value of an Annuity")
    st.latex(r"\text{FV}_{\text{annuity}} = C \times \frac{(1+r)^n - 1}{r}")
    C = st.number_input("Regular Payment (C)", value=500.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.08, 
                        step=1e-4, format="%.4f")
    n = st.number_input("Number of Periods", value=10, step=1)
    if st.button("Calculate FV Annuity"):
        result = annuity_fv(C, r, n)
        st.success(f"FV of Annuity = {result:,.6f}")

elif selected_formula == "Growing Annuity (PV)":
    st.subheader("Present Value of a Growing Annuity")
    st.latex(r"\text{PV} = C_1 \times \frac{1 - \left(\frac{1+g}{1+r}\right)^n}{r-g}")
    C = st.number_input("First Payment (C1)", value=500.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", value=0.10, 
                        step=1e-4, format="%.4f")
    g = st.number_input("Growth Rate (decimal)", value=0.03, 
                        step=1e-4, format="%.4f")
    n = st.number_input("Number of Periods", value=10, step=1)
    if st.button("Calculate PV Growing Annuity"):
        result = growing_annuity_pv(C, r, g, n)
        st.success(f"PV of Growing Annuity = {result:,.6f}")

elif selected_formula == "Perpetuity":
    st.subheader("Present Value of a Perpetuity")
    st.latex(r"\text{PV}_{\text{perpetuity}} = \frac{C}{r}")
    C = st.number_input("Constant Payment (C)", value=100.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", value=0.05, 
                        step=1e-4, format="%.4f")
    if st.button("Calculate PV Perpetuity"):
        result = perpetuity_pv(C, r)
        st.success(f"PV of Perpetuity = {result:,.6f}")

elif selected_formula == "Growing Perpetuity":
    st.subheader("Present Value of a Growing Perpetuity")
    st.latex(r"\text{PV}_{\text{growing perpetuity}} = \frac{C_1}{r - g}")
    C = st.number_input("Payment in NEXT Period (C1)", value=100.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", value=0.07, 
                        step=1e-4, format="%.4f")
    g = st.number_input("Growth Rate (decimal)", value=0.03, 
                        step=1e-4, format="%.4f")
    if st.button("Calculate PV Growing Perpetuity"):
        if g >= r:
            st.error("Growth rate must be less than discount rate.")
        else:
            result = growing_perpetuity_pv(C, r, g)
            st.success(f"PV of Growing Perpetuity = {result:,.6f}")

####################################
# Bond
####################################

elif selected_formula == "Bond Price (Coupon)":
    st.subheader("Bond Price (Coupon Bond)")
    st.latex(r"""
    P = \text{Coupon PV (annuity)} + \text{Face Value PV}
    = \left(\frac{C}{r}\left[1 - (1+r)^{-N}\right]\right) + \frac{F}{(1+r)^N}
    """)
    face_value = st.number_input("Face Value", value=1000.0, step=1.0)
    coupon_rate = st.number_input("Annual Coupon Rate (decimal)", value=0.06, step=1e-6,format="%.6f")
    years_to_maturity = st.number_input("Years to Maturity", value=10.0,  step=1e-4)
    coupons_per_year = st.number_input("Coupons per Year", value=2, step=1)
    ytm_annual = st.number_input("Annual YTM (decimal)", value=0.006, step=1e-6,format="%.6f")
    
    if st.button("Calculate Bond Price"):
        price = bond_price_coupon(face_value, coupon_rate, years_to_maturity, coupons_per_year, ytm_annual)
        st.success(f"The Bond Price is ${price:,.2f}")

elif selected_formula == "Bond YTM (Coupon)":
    st.subheader("Bond YTM (Coupon Bond)")
    st.write("We solve for the yield (r) that sets the present value of all coupon and principal payments equal to the bond price.")
    face_value = st.number_input("Face Value", value=1000.0, step=1.0)
    coupon_rate = st.number_input("Annual Coupon Rate (decimal)", value=0.05, step=1e-4)
    years_to_maturity = st.number_input("Years to Maturity", value=10, step=1)
    coupons_per_year = st.number_input("Coupons per Year", value=2, step=1)
    bond_price = st.number_input("Current Bond Price", value=950.0, step=1.0)
    
    if st.button("Calculate Bond YTM"):
        period_irr, annual_irr = bond_ytm_coupon(face_value, coupon_rate, years_to_maturity, coupons_per_year, bond_price)
        st.write(f"Periodic YTM = {period_irr*100:,.4f}% per period")
        st.success(f"Annualized YTM = {annual_irr*100:,.4f}% per year")

elif selected_formula == "Bond Coupon Rate":
    st.subheader("Bond Coupon Rate from Price and YTM")
    st.write("Compute the annual coupon rate (c) given the bond's price, YTM, maturity, etc.")
    bond_price_val = st.number_input("Current Bond Price", value=1071.06, step=1.0)
    face_value_val = st.number_input("Face Value", value=1000.0, step=1.0)
    annual_ytm_val = st.number_input("Annual YTM (decimal)", value=0.07, step=1e-4)
    years_to_maturity_val = st.number_input("Years to Maturity", value=10, step=1)
    coupons_per_year_val = st.number_input("Coupons per Year", value=2, step=1)

    if st.button("Calculate Coupon Rate"):
        c = bond_coupon_rate(bond_price_val, face_value_val, annual_ytm_val, years_to_maturity_val, coupons_per_year_val)
        if c is None:
            st.error("Could not compute coupon rate (division by zero or invalid inputs).")
        else:
            st.write(f"Annual Coupon Rate (decimal) = {c:.6f}")
            st.success(f"Annual Coupon Rate (percent) = {c*100:.4f}%")

####################################
# Capital Budgeting
####################################

elif selected_formula == "Net Present Value (NPV)":
    st.subheader("Net Present Value")
    st.latex(r"\text{NPV} = \sum_{t=0}^{n} \frac{\text{CF}_t}{(1+r)^t}")
    st.write("Enter a sequence of cash flows including the initial cost (negative) at t=0.")
    num_flows = st.number_input("Number of cash flow periods (including t=0)?", value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", value=0.0, step=1.0, key=f"cf_{i}")
        cf_list.append(val)
    
    st.write("---")
    st.write("**Optional: Include a Salvage Value**")
    salvage_value = st.number_input("Salvage Value (if any)", value=0.0, step=1.0)
    salvage_period = st.number_input("Salvage Value Received at End of Period t=?",
                                     value=int(num_flows - 1),
                                     min_value=0,
                                     max_value=max(num_flows - 1, 0),
                                     step=1)
    st.write("---")

    r = st.number_input("Discount Rate (decimal)", value=0.10, step=1e-4)
    
    if st.button("Calculate NPV"):
        if salvage_value != 0 and 0 <= salvage_period < num_flows:
            cf_list[salvage_period] += salvage_value
        result = npv_calc(cf_list, r)
        st.success(f"NPV = {result:,.4f}")
        if result > 0:
            st.info("NPV > 0, Accept the project!")
        else:
            st.warning("NPV <= 0, Reject the project.")

elif selected_formula == "Internal Rate of Return (IRR)":
    st.subheader("Internal Rate of Return")
    st.latex(r"\text{Solve for } r \text{ where } \text{NPV}=0")
    st.write("Enter a sequence of cash flows including the initial cost (negative) at t=0.")
    num_flows = st.number_input("Number of cash flow periods?", value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", value=0.0, step=1.0, key=f"irr_cf_{i}")
        cf_list.append(val)
    
    if st.button("Calculate IRR"):
        result = irr_calc(cf_list)
        st.success(f"IRR = {result*100:,.4f}%")

elif selected_formula == "Payback Period":
    st.subheader("Payback Period")
    st.write("Enter a sequence of cash flows including the initial cost (negative) at t=0.")
    num_flows = st.number_input("Number of cash flow periods?", value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", value=0.0, step=1.0, key=f"pb_cf_{i}")
        cf_list.append(val)
    
    if st.button("Calculate Payback Period"):
        result = payback_period(cf_list)
        if result is not None:
            st.success(f"Payback Period ≈ {result} periods")
        else:
            st.warning("The initial investment was never fully recovered.")

elif selected_formula == "Discounted Payback Period":
    st.subheader("Discounted Payback Period")
    st.write("Enter a sequence of cash flows including the initial cost (negative) at t=0.")
    num_flows = st.number_input("Number of cash flow periods?", value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", value=0.0, step=1.0, key=f"dpb_cf_{i}")
        cf_list.append(val)
    
    r = st.number_input("Discount Rate (decimal)", value=0.10, step=1e-4)
    
    if st.button("Calculate Discounted Payback Period"):
        result = discounted_payback_period(cf_list, r)
        if result is not None:
            st.success(f"Discounted Payback Period ≈ {result} periods")
        else:
            st.warning("The present value of cash flows never recovers the initial cost.")

####################################
# Stock Valuation
####################################

elif selected_formula == "Stock - Constant Dividend Price":
    st.subheader("Stock Price (Constant Dividend)")
    # Add some explanation
    st.markdown("**For a zero-growth stock (like preferred shares):**")
    st.latex(r"P_0 = \frac{D}{r}")
    
    dividend = st.number_input("Dividend per period (D)", value=2.0, step=0.1)
    r = st.number_input("Required Return (decimal)", value=0.10, step=1e-4)
    
    if st.button("Calculate Stock Price"):
        if r <= 0:
            st.error("Required return must be positive.")
        else:
            price = stock_price_constant_dividend(dividend, r)
            st.success(f"Stock Price = {price:,.2f}")
            st.markdown(f"**Dividend Yield** = D / P₀ = {dividend:.2f} / {price:.2f} = {(dividend/price)*100:.2f}%")

elif selected_formula == "Stock - Constant Growth Dividend Price (Gordon Growth)":
    st.subheader("Stock Price (Constant Growth)")
    st.markdown("**Gordon Growth Model:**")
    st.latex(r"P_0 = \frac{D_1}{r - g} \;=\; \frac{D_0 \,(1+g)}{r - g}")
    st.markdown("**Also**, total return \(r =\) dividend yield + capital gains yield.")
    
    D0 = st.number_input("Most recent dividend D0", value=2.0, step=0.1)
    g = st.number_input("Growth Rate g (decimal)", value=0.03, step=1e-4, format="%.4f")
    r = st.number_input("Required Return r (decimal)", value=0.10, step=1e-4, format="%.4f")
    
    if st.button("Calculate Price"):
        price = stock_price_constant_growth(D0, r, g)
        if price is None:
            st.error("r must be greater than g.")
        else:
            st.success(f"Stock Price = {price:,.2f}")
            D1 = D0 * (1+g)
            div_yield = D1 / price
            cap_gain_yield = g
            st.markdown(
                f"**Dividend Yield** = D₁ / P₀ = {D1:.2f}/{price:.2f} = {div_yield*100:.2f}%  \n"
                f"**Capital Gains Yield** = g = {cap_gain_yield*100:.2f}%  \n"
                f"**Total Return** = {div_yield*100 + cap_gain_yield*100:.2f}%"
            )

elif selected_formula == "Stock - Required Return (Gordon Growth)":
    st.subheader("Required Return from Gordon Growth")
    st.latex(r"r = \frac{D_1}{P_0} + g")
    
    P0 = st.number_input("Current Stock Price P0", value=40.0, step=1.0)
    D1 = st.number_input("Next Period Dividend D1", value=4.0, step=0.1)
    g = st.number_input("Growth Rate g (decimal)", value=0.06, step=1e-4)
    
    if st.button("Calculate Required Return"):
        if P0 <= 0:
            st.error("Stock price must be > 0.")
        else:
            r_val = stock_required_return_gordon(D1, P0, g)
            st.success(f"Required Return = {r_val*100:,.2f}%")
            st.markdown(
                f"**Dividend Yield** = D₁/P₀ = {D1}/{P0} = {(D1/P0)*100:.2f}%  \n"
                f"**Capital Gains Yield** = g = {g*100:.2f}%  \n"
                f"**Total Return** = {r_val*100:.2f}%"
            )

elif selected_formula == "Stock - Non-Constant Growth Dividend Price":
    st.subheader("Stock Price with Non-Constant (Multi-Stage) Growth")
    st.markdown("**General formula:**")
    st.latex(r"""
    P_0
    = \sum_{t=1}^{n} \frac{D_t}{(1+r)^t}
    \;+\;
    \frac{D_{n+1}}{(r - g)\,(1+r)^n}.
    """)
    st.write("Enter a series of dividends for each year of supernormal growth; beyond that, dividends grow at a constant rate g.")
    
    n = st.number_input("Number of years with explicit dividends", value=3, step=1)
    r = st.number_input("Required Return (decimal)", value=0.10, step=1e-4)
    g = st.number_input("Perpetual Growth Rate after year n (decimal)", value=0.05, step=1e-4)
    
    dividends = []
    for i in range(n):
        d_val = st.number_input(
            f"Dividend at end of year {i+1}", 
            value=2.0 + i*0.1, 
            step=0.1, 
            format="%.2f", 
            key=f"nonconst_{i}"
        )
        dividends.append(d_val)
    
    if st.button("Calculate Non-Constant Growth Stock Price"):
        price = stock_price_nonconstant(dividends, r, g, n)
        if price is None:
            st.error("Invalid: r must be greater than g.")
        else:
            st.success(f"Stock Price = {price:,.2f}")

elif selected_formula == "Corporate Value (FCF) Model":
    st.subheader("Corporate Value Model (Free Cash Flow)")
    st.markdown("**Firm Value** = Present Value of all FCF + PV of Terminal Value:")
    st.latex(r"""
    \mathrm{Value}_{\text{firm}} 
    = \sum_{t=1}^{n} \frac{\mathrm{FCF}_t}{(1+r)^t}
    \;+\;
    \frac{\mathrm{FCF}_{n+1}}{(r - g)\,(1+r)^n}.
    """)
    st.markdown("**Equity Value** = Firm Value − Debt.")
    st.markdown("**Stock Price** = Equity Value / Shares Outstanding.")
    
    num_years = st.number_input("Number of years of explicit FCF", value=3, step=1)
    fcf_list = []
    for i in range(num_years):
        val = st.number_input(f"FCF at end of year {i+1}", value=10.0 + i*2, step=1.0, key=f"fcf_{i}")
        fcf_list.append(val)
    
    r = st.number_input("Required Return / WACC (decimal)", value=0.10, step=1e-4)
    g = st.number_input("Long-term Growth Rate (decimal)", value=0.04, step=1e-4)
    debt = st.number_input("Market Value of Debt", value=40.0, step=1.0)
    shares = st.number_input("Number of Shares", value=10.0, step=1.0)
    
    if st.button("Calculate Price per Share"):
        result = corporate_value_model(fcf_list, r, g, debt, shares)
        if result is None:
            st.error("Check that r > g and that you have at least 1 FCF.")
        else:
            st.success(f"Intrinsic Value per Share = ${result:,.2f}")

elif selected_formula == "Stock Price from PE Multiple":
    st.subheader("Stock Price from P/E Multiple")
    st.markdown("**Multiple approach:**")
    st.latex(r"\text{Price} = \text{EPS} \times \text{(P/E ratio)}")
    
    eps = st.number_input("Earnings per Share (EPS)", value=3.45, step=0.01)
    pe_ratio = st.number_input("Benchmark PE Ratio", value=12.5, step=0.1)
    
    if st.button("Calculate Stock Price"):
        price = pe_multiple_valuation(eps, pe_ratio)
        st.success(f"Estimated Stock Price = ${price:,.2f}")

elif selected_formula == "Free Cash Flow (FCF)":
    st.subheader("Free Cash Flow Calculation")
    st.latex(r"""
    \text{FCF} = EBIT(1 - \tau_c) + \text{Depreciation} - \text{CapEx} - \Delta NWC
    """)
    ebit = st.number_input("Earnings Before Interest & Taxes (EBIT)", value=100.0, step=1.0)
    tax_rate = st.number_input("Corporate Tax Rate (decimal)", value=0.25, step=0.01)
    depreciation = st.number_input("Depreciation Expense", value=10.0, step=1.0)
    capex = st.number_input("Capital Expenditures (CapEx)", value=20.0, step=1.0)
    delta_nwc = st.number_input("Change in Net Working Capital (ΔNWC)", value=5.0, step=1.0)
    
    if st.button("Calculate Free Cash Flow"):
        fcf_result = free_cash_flow(ebit, tax_rate, depreciation, capex, delta_nwc)
        st.success(f"Free Cash Flow (FCF) = ${fcf_result:,.2f}")

st.markdown("---")
st.header("Scientific Calculator")
st.markdown("Enter any mathematical expression. For example: `sin(pi/2) + log(10)`.\n\n"
            "This calculator leverages **Sympy** to parse and evaluate your expression, "
            "supporting a wide range of functions, constants, and even symbolic operations.")

# Text input for the user to enter an expression
expression = st.text_input("Expression", value="sin(pi/2) + log(10)")

if st.button("Evaluate Expression"):
    try:
        # Parse and evaluate the expression using sympy
        expr = sp.sympify(expression)
        result = sp.N(expr)
        st.success(f"Result: {result}")
    except Exception as e:
        st.error(f"Error evaluating expression: {e}")