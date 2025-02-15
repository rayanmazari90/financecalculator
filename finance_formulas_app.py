import streamlit as st
import numpy as np

####################################
# Helper Functions (Existing)
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
    if abs(r - g) < 1e-12:
        # Handle the special case where r ≈ g
        return C * n / (1 + r)
    return C * (1 - ((1 + g)/(1 + r))**n) / (r - g)

def perpetuity_pv(C, r):
    return C / r

def growing_perpetuity_pv(C, r, g):
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
    for i in range(max_iterations):
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
    return None  # Not recovered

def discounted_payback_period(cash_flows, r):
    cumulative = 0.0
    for i, cf in enumerate(cash_flows):
        discounted_cf = cf / ((1 + r)**i)
        cumulative += discounted_cf
        if cumulative >= 0:
            return i
    return None  # Not recovered

####################################
# NEW Helper Functions (Stock Valuation)
####################################

def stock_price_constant_dividend(dividend, r):
    """
    Price of a stock with a constant (zero-growth) dividend stream.
    P0 = D / r
    """
    return dividend / r

def stock_price_constant_growth(D0, r, g):
    """
    Price of a stock with constant growth in dividends (Gordon Growth).
    P0 = D0*(1+g)/(r - g)
    """
    if r <= g:
        return None  # not mathematically valid if r <= g
    return (D0 * (1 + g)) / (r - g)

def stock_required_return_gordon(D1, P0, g):
    """
    Required return, given next dividend D1, current price P0, and growth g.
    r = D1/P0 + g
    """
    return (D1 / P0) + g

def stock_price_nonconstant(dividends, r, g, start_growth_year):
    """
    Non-constant (multi-stage) dividend model. 
    - 'dividends' is a list/array of *explicit* dividends for years 1..N.
    - After year N, dividends grow at a constant rate 'g' forever.
    - 'start_growth_year' = N (the last year with explicit dividend).
    
    Steps to compute:
      1) Discount each of the 'dividends' back to time 0.
      2) Compute terminal value at year N, discount back to time 0.
         Terminal Value at year N = D_{N+1} / (r - g),
         where D_{N+1} = dividends[N-1] * (1+g).
    """
    # 1) Sum of PV of short-term dividends
    pv_sum = 0.0
    for i, div in enumerate(dividends):
        year = i + 1  # Div at end of year (i+1)
        pv_sum += div / ((1 + r)**year)
    
    # 2) Terminal Value at year 'start_growth_year'
    Dn = dividends[-1]  # last explicit dividend
    D_next = Dn * (1 + g)  # next dividend after last explicit
    if r <= g:
        return None  # not valid if r <= g
    terminal_value = D_next / (r - g)
    
    # PV of that terminal value
    pv_terminal = terminal_value / ((1 + r)**start_growth_year)
    
    return pv_sum + pv_terminal

def corporate_value_model(fcf_list, r, g, debt, shares):
    """
    Corporate (Free Cash Flow) Model
    1) We discount a list of free cash flows (fcf_list) from year 1..N.
    2) Starting from year N+1, FCF grows at rate g forever => compute terminal value at year N.
       Terminal Value at N = FCF_{N+1} / (r - g) 
         where FCF_{N+1} = FCF_N * (1 + g)
    3) MV of firm = PV(all FCFs) + PV(terminal value).
    4) Equity value = MV of firm - debt
    5) Price per share = Equity value / # of shares
    """
    # 1) Sum of PV of explicit free cash flows
    pv_sum = 0.0
    for i, fcf in enumerate(fcf_list):
        year = i + 1  # FCF at the end of year (i+1)
        pv_sum += fcf / ((1 + r)**year)
    
    # 2) Terminal value at year N
    #    FCF_{N+1} = fcf_list[-1] * (1 + g)
    if len(fcf_list) == 0:
        return None
    FCFn = fcf_list[-1]
    FCF_next = FCFn * (1 + g)
    if r <= g:
        return None
    terminal_value = FCF_next / (r - g)
    # discount it back to time 0
    N = len(fcf_list)
    pv_terminal = terminal_value / ((1 + r)**N)
    
    # 3) Market value of firm
    mv_firm = pv_sum + pv_terminal
    
    # 4) Equity value
    mv_equity = mv_firm - debt
    
    # 5) Price per share
    price_per_share = mv_equity / shares
    
    return price_per_share

def pe_multiple_valuation(eps, pe_ratio):
    """
    Stock price using P/E multiple = EPS * Benchmark PE
    """
    return eps * pe_ratio


####################################
# Streamlit App
####################################

st.title("Finance Formulas App")
st.write("""
This application calculates various finance formulas taught in your sessions. 
Use the dropdown menu to pick a formula, input parameters, and see the result!
""")

formula = st.selectbox("Select a formula to compute:", [
    "Future Value (FV)",
    "Present Value (PV)",
    "Annuity (PV)",
    "Annuity (FV)",
    "Growing Annuity (PV)",
    "Perpetuity",
    "Growing Perpetuity",
    "Net Present Value (NPV)",
    "Internal Rate of Return (IRR)",
    "Payback Period",
    "Discounted Payback Period",
    "Bond Price (Coupon)",
    "Bond YTM (Coupon)",
    "Bond Coupon Rate",

    # -------------------------------
    # NEW STOCK VALUATION FORMULAS
    # -------------------------------
    "Stock - Constant Dividend Price",
    "Stock - Constant Growth Dividend Price (Gordon Growth)",
    "Stock - Required Return (Gordon Growth)",
    "Stock - Non-Constant Growth Dividend Price",
    "Corporate Value (FCF) Model",
    "Stock Price from PE Multiple",
])

####################################
# Existing Code Selections
####################################

if formula == "Future Value (FV)":
    st.subheader("Future Value")
    pv = st.number_input("Present Value", value=1000.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.10, 
                        step=1e-12, format="%.12f")
    n = st.number_input("Number of Periods (n)", value=5, step=1)
    if st.button("Calculate FV"):
        fv = future_value(pv, r, n)
        st.success(f"Future Value = {fv:,.6f}")

elif formula == "Present Value (PV)":
    st.subheader("Present Value")
    fv_amt = st.number_input("Future Value to be discounted", value=1000.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.10, 
                        step=1e-12, format="%.12f")
    n = st.number_input("Number of Periods (n)", value=5, step=1)
    if st.button("Calculate PV"):
        pv = present_value(fv_amt, r, n)
        st.success(f"Present Value = {pv:,.6f}")

elif formula == "Annuity (PV)":
    st.subheader("Present Value of an Annuity")
    C = st.number_input("Regular Payment (C)", value=500.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.08, 
                        step=1e-12, format="%.12f")
    n = st.number_input("Number of Periods (n)", value=10, step=1)
    if st.button("Calculate PV Annuity"):
        result = annuity_pv(C, r, n)
        st.success(f"PV of Annuity = {result:,.6f}")

elif formula == "Annuity (FV)":
    st.subheader("Future Value of an Annuity")
    C = st.number_input("Regular Payment (C)", value=500.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.08, 
                        step=1e-12, format="%.12f")
    n = st.number_input("Number of Periods (n)", value=10, step=1)
    if st.button("Calculate FV Annuity"):
        result = annuity_fv(C, r, n)
        st.success(f"FV of Annuity = {result:,.6f}")

elif formula == "Growing Annuity (PV)":
    st.subheader("Present Value of a Growing Annuity")
    C = st.number_input("Payment in the 1st period (C1)", value=500.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", value=0.10, 
                        step=1e-12, format="%.12f")
    g = st.number_input("Growth Rate (decimal)", value=0.03, 
                        step=1e-12, format="%.12f")
    n = st.number_input("Number of Periods (n)", value=10, step=1)
    if st.button("Calculate PV Growing Annuity"):
        result = growing_annuity_pv(C, r, g, n)
        st.success(f"PV of Growing Annuity = {result:,.6f}")

elif formula == "Perpetuity":
    st.subheader("Present Value of a Perpetuity")
    C = st.number_input("Regular Payment (C) each period forever", 
                        value=100.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", value=0.05, 
                        step=1e-12, format="%.12f")
    if st.button("Calculate PV Perpetuity"):
        result = perpetuity_pv(C, r)
        st.success(f"PV of Perpetuity = {result:,.6f}")

elif formula == "Growing Perpetuity":
    st.subheader("Present Value of a Growing Perpetuity")
    C = st.number_input("Payment in NEXT period (C1)", 
                        value=100.0, step=1.0)
    r = st.number_input("Discount Rate (decimal)", 
                        value=0.07, step=1e-12, format="%.12f")
    g = st.number_input("Growth Rate (decimal)", 
                        value=0.03, step=1e-12, format="%.12f")
    if st.button("Calculate PV Growing Perpetuity"):
        if g >= r:
            st.error("Growth rate must be less than discount rate.")
        else:
            result = growing_perpetuity_pv(C, r, g)
            st.success(f"PV of Growing Perpetuity = {result:,.6f}")

elif formula == "Net Present Value (NPV)":
    st.subheader("Net Present Value")
    st.write("Enter a sequence of cash flows (including the initial cost at t=0).")
    num_flows = st.number_input("Number of cash flow periods (including t=0)?", 
                                value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", value=0.0, step=1.0, key=f"cf_{i}")
        cf_list.append(val)
    
    st.write("---")
    st.write("**Optional: Include a Salvage Value**")
    salvage_value = st.number_input("Salvage Value (if any)", 
                                    value=0.0, step=1.0)
    salvage_period = st.number_input(
        "Salvage Value Received at End of Period t=?",
        value=int(num_flows - 1 if num_flows > 0 else 0),
        min_value=0,
        max_value=max(num_flows - 1, 0),
        step=1
    )
    st.write("---")

    r = st.number_input("Discount Rate (decimal)", value=0.10, 
                        step=1e-12, format="%.12f")
    
    if st.button("Calculate NPV"):
        if salvage_value != 0 and 0 <= salvage_period < num_flows:
            cf_list[salvage_period] += salvage_value

        result = npv_calc(cf_list, r)
        st.success(f"NPV = {result:,.6f}")
        if result > 0:
            st.info("NPV > 0, Accept the project!")
        else:
            st.warning("NPV <= 0, Reject the project.")

elif formula == "Internal Rate of Return (IRR)":
    st.subheader("Internal Rate of Return")
    st.write("Enter a sequence of cash flows (including the initial cost at t=0).")
    num_flows = st.number_input("Number of cash flow periods?", value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", 
                              value=0.0, step=1.0, key=f"irr_cf_{i}")
        cf_list.append(val)
    
    if st.button("Calculate IRR"):
        result = irr_calc(cf_list)
        st.success(f"IRR = {result*100:,.6f}%")

elif formula == "Payback Period":
    st.subheader("Payback Period")
    st.write("Enter a sequence of cash flows (including the initial cost at t=0).")
    num_flows = st.number_input("Number of cash flow periods?", value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", 
                              value=0.0, step=1.0, key=f"pb_cf_{i}")
        cf_list.append(val)
    
    if st.button("Calculate Payback Period"):
        result = payback_period(cf_list)
        if result is not None:
            st.success(f"Payback Period ≈ {result} periods")
        else:
            st.warning("The initial investment was never fully recovered.")

elif formula == "Discounted Payback Period":
    st.subheader("Discounted Payback Period")
    st.write("Enter a sequence of cash flows (including the initial cost at t=0).")
    num_flows = st.number_input("Number of cash flow periods?", value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", 
                              value=0.0, step=1.0, key=f"dpb_cf_{i}")
        cf_list.append(val)
    
    r = st.number_input("Discount Rate (decimal)", value=0.10, 
                        step=1e-12, format="%.12f")
    
    if st.button("Calculate Discounted Payback Period"):
        result = discounted_payback_period(cf_list, r)
        if result is not None:
            st.success(f"Discounted Payback Period ≈ {result} periods")
        else:
            st.warning("The present value of cash flows never recovers the initial cost.")

elif formula == "Bond Price (Coupon)":
    st.subheader("Bond Price (Coupon Bond)")
    face_value = st.number_input("Face Value (F)", value=1000.0, step=1.0)
    coupon_rate = st.number_input("Annual Coupon Rate (decimal)", 
                                  value=0.06, step=1e-12, format="%.12f")
    years_to_maturity = st.number_input("Years to Maturity (N)", value=10, step=1)
    coupons_per_year = st.number_input("Coupons per Year (m)", value=2, step=1)
    ytm_annual = st.number_input("Annual YTM (decimal)", 
                                 value=0.06, step=1e-12, format="%.12f")
    
    if st.button("Calculate Bond Price"):
        C = (coupon_rate * face_value) / coupons_per_year
        r = ytm_annual / coupons_per_year
        T = int(coupons_per_year * years_to_maturity)
        
        price = annuity_pv(C, r, T) + present_value(face_value, r, T)
        
        st.success(f"The Bond Price is ${price:,.6f}")

elif formula == "Bond YTM (Coupon)":
    st.subheader("Bond YTM (Coupon Bond)")
    face_value = st.number_input("Face Value (F)", value=1000.0, step=1.0)
    coupon_rate = st.number_input("Annual Coupon Rate (decimal)", 
                                  value=0.05, step=1e-12, format="%.12f")
    years_to_maturity = st.number_input("Years to Maturity (N)", value=10, step=1)
    coupons_per_year = st.number_input("Coupons per Year (m)", value=2, step=1)
    bond_price = st.number_input("Current Market Price of the Bond (P)", 
                                 value=950.0, step=1.0)
    
    if st.button("Calculate Bond YTM"):
        # Periodic coupon
        C = (coupon_rate * face_value) / coupons_per_year
        T = int(coupons_per_year * years_to_maturity)
        
        # CFs: CF0=-bond_price, CF1..CF(T-1)=C, CF(T)=C+face_value
        cf_list = [-bond_price] + [C]*(T-1) + [C + face_value]
        
        period_irr = irr_calc(cf_list, guess=0.05) 
        annual_irr = period_irr * coupons_per_year
        
        st.write(f"Periodic YTM = {period_irr*100:,.6f}% per period")
        st.success(f"Annualized YTM = {annual_irr*100:,.6f}% per year")

elif formula == "Bond Coupon Rate":
    st.subheader("Bond Coupon Rate from Price and YTM")
    st.write("Calculate the **annual coupon rate** given the bond's price, YTM, maturity, face value, and coupons/year.")
    
    # 1) Inputs
    bond_price = st.number_input("Current Bond Price (P)", value=1071.06, step=1.0)
    face_value = st.number_input("Face Value (F)", value=1000.0, step=1.0)
    annual_ytm = st.number_input("Annual YTM (decimal)", value=0.07, step=1e-4, format="%.6f")
    years_to_maturity = st.number_input("Years to Maturity (N)", value=10, step=1)
    coupons_per_year = st.number_input("Coupons per Year (m)", value=2, step=1)

    if st.button("Calculate Coupon Rate"):
        r = annual_ytm / coupons_per_year  # periodic rate
        T = coupons_per_year * years_to_maturity  # total coupon periods
        
        # P = (c * F / m) * [ (1 - (1+r)^(-T)) / r ] + F / (1+r)^T
        # => c = [ P - F/(1+r)^T ] / [ (F/m) * ( (1 - (1+r)^(-T)) / r ) ]
        
        discounted_face = present_value(face_value, r, T)
        annuity_factor = annuity_pv(1, r, T)
        
        numerator = bond_price - discounted_face
        denominator = (face_value / coupons_per_year) * annuity_factor
        try:
            c = numerator / denominator
            st.write(f"Annual Coupon Rate (decimal) = {c:.6f}")
            st.success(f"Annual Coupon Rate (percent) = {c * 100:.4f}%")
        except ZeroDivisionError:
            st.error("Could not compute coupon rate (division by zero). Check inputs.")

####################################
# NEW CODE: Stock Valuation
####################################

elif formula == "Stock - Constant Dividend Price":
    st.subheader("Stock Price (Constant Dividend)")
    st.write("For a **preferred stock** or zero-growth dividend:")
    st.latex(r"P_0 = \frac{D}{r}")
    
    dividend = st.number_input("Dividend per period (D)", value=2.0, step=0.1)
    r = st.number_input("Required Return (decimal)", value=0.10, step=1e-4, format="%.4f")
    
    if st.button("Calculate Stock Price"):
        if r <= 0:
            st.error("Required return must be positive.")
        else:
            price = stock_price_constant_dividend(dividend, r)
            st.success(f"Stock Price = {price:,.4f}")

elif formula == "Stock - Constant Growth Dividend Price (Gordon Growth)":
    st.subheader("Stock Price (Constant Growth Dividend)")
    st.latex(r"P_0 = \frac{D_1}{r - g} = \frac{D_0(1+g)}{r - g}")
    
    D0 = st.number_input("Most recent dividend (D0)", value=2.0, step=0.1)
    g = st.number_input("Growth Rate (decimal)", value=0.03, step=1e-4, format="%.4f")
    r = st.number_input("Required Return (decimal)", value=0.10, step=1e-4, format="%.4f")
    
    if st.button("Calculate Gordon Growth Price"):
        price = stock_price_constant_growth(D0, r, g)
        if price is None:
            st.error("Invalid: r must be greater than g.")
        else:
            st.success(f"Stock Price = {price:,.4f}")

elif formula == "Stock - Required Return (Gordon Growth)":
    st.subheader("Required Return from Gordon Growth Model")
    st.latex(r"r = \frac{D_1}{P_0} + g")
    
    P0 = st.number_input("Current Stock Price (P0)", value=40.0, step=1.0)
    D1 = st.number_input("Next Period Dividend (D1)", value=4.0, step=0.1)
    g = st.number_input("Growth Rate (decimal)", value=0.06, step=1e-4, format="%.4f")
    
    if st.button("Calculate Required Return"):
        if P0 <= 0:
            st.error("Stock price must be positive.")
        else:
            r = stock_required_return_gordon(D1, P0, g)
            st.success(f"Required Return = {r*100:,.4f}%")

elif formula == "Stock - Non-Constant Growth Dividend Price":
    st.subheader("Stock Price with Non-Constant (Multi-Stage) Dividend Growth")
    st.write("Enter a series of dividends for each year of ‘supernormal’ growth, then specify the perpetual growth rate afterwards.")
    
    n = st.number_input("Number of years with explicitly forecasted dividends:", value=3, step=1)
    r = st.number_input("Required Return (decimal)", value=0.10, step=1e-4, format="%.4f")
    g = st.number_input("Perpetual Growth Rate AFTER year n (decimal)", value=0.05, step=1e-4, format="%.4f")
    
    dividends = []
    for i in range(n):
        d_val = st.number_input(f"Dividend at end of Year {i+1}", value=2.0 + i*0.1, step=0.1, format="%.4f", key=f"nonconst_div_{i}")
        dividends.append(d_val)
    
    if st.button("Calculate Non-Constant Growth Stock Price"):
        price = stock_price_nonconstant(dividends, r, g, n)
        if price is None:
            st.error("Invalid: r must be greater than g.")
        else:
            st.success(f"Stock Price = {price:,.4f}")
            st.info("This is the intrinsic value today (time 0).")

elif formula == "Corporate Value (FCF) Model":
    st.subheader("Corporate Value Model (Free Cash Flow)")
    st.write("Discount future free cash flows, compute terminal value, subtract debt, divide by shares outstanding.")
    
    num_years = st.number_input("Number of years with explicit FCF forecasts:", value=3, step=1)
    fcf_list = []
    for i in range(num_years):
        val = st.number_input(f"FCF at end of Year {i+1}:", value=10.0 + i*2.0, step=1.0, key=f"fcf_{i}")
        fcf_list.append(val)
    
    r = st.number_input("WACC / Required Return (decimal)", value=0.10, step=1e-4, format="%.4f")
    g = st.number_input("Long-term Growth Rate after last forecast (decimal)", value=0.04, step=1e-4, format="%.4f")
    debt = st.number_input("Market Value of Debt", value=40.0, step=1.0)
    shares = st.number_input("Number of Shares Outstanding", value=10.0, step=1.0)
    
    if st.button("Calculate Intrinsic Stock Price"):
        price_per_share = corporate_value_model(fcf_list, r, g, debt, shares)
        if price_per_share is None:
            st.error("Invalid: required return must be greater than growth rate, and have at least 1 FCF.")
        else:
            st.success(f"Intrinsic Value per Share = ${price_per_share:,.4f}")

elif formula == "Stock Price from PE Multiple":
    st.subheader("Stock Valuation using P/E Multiple")
    st.latex(r" \text{Price} = \text{EPS} \times \text{Benchmark PE} ")
    
    eps = st.number_input("Earnings per Share (EPS)", value=3.45, step=0.01)
    pe_ratio = st.number_input("Benchmark P/E Ratio", value=12.5, step=0.1)
    
    if st.button("Calculate Price from PE Multiple"):
        price = pe_multiple_valuation(eps, pe_ratio)
        st.success(f"Estimated Stock Price = ${price:,.2f}")
