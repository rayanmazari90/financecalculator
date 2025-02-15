import streamlit as st
import numpy as np

####################################
# Helper Functions
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
        # If r == g (or extremely close), the usual formula blows up.
        # You could handle it specially or approximate:
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
        return npv_calc(cash_flows, rate)
    
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
    return None  # If never recovered

def discounted_payback_period(cash_flows, r):
    cumulative = 0.0
    for i, cf in enumerate(cash_flows):
        discounted_cf = cf / ((1 + r)**i)
        cumulative += discounted_cf
        if cumulative >= 0:
            return i
    return None

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
])

if formula == "Future Value (FV)":
    st.subheader("Future Value")
    pv = st.number_input("Present Value", value=1000.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.10, 
                        step=1e-12, format="%.12f")
    n = st.number_input("Number of Periods (n)", value=5, step=1)
    if st.button("Calculate FV"):
        fv = future_value(pv, r, n)
        st.success(f"Future Value = {fv:,.6f}")
if formula == "Bond Coupon Rate":
    st.subheader("Bond Coupon Rate from Price and YTM")
    st.write("Calculate the **annual coupon rate** given the bond's current market price, its yield to maturity (YTM), maturity, face value, and the number of coupons per year.")
    
    # 1) Inputs
    bond_price = st.number_input("Current Bond Price (P)", value=1071.06, step=1.0)
    face_value = st.number_input("Face Value (F)", value=1000.0, step=1.0)
    annual_ytm = st.number_input("Annual YTM (decimal)", value=0.07, step=1e-4, format="%.6f")
    years_to_maturity = st.number_input("Years to Maturity (N)", value=10, step=1)
    coupons_per_year = st.number_input("Coupons per Year (m)", value=2, step=1)

    if st.button("Calculate Coupon Rate"):
        # 2) Convert inputs for formula
        r = annual_ytm / coupons_per_year  # periodic rate
        T = coupons_per_year * years_to_maturity  # total coupon periods
        
        # 3) Solve for c in: 
        # P = (c * F / m) * [ (1 - (1+r)^(-T)) / r ] + F / (1+r)^T
        # => c = [ P - F/(1+r)^T ] / [ (F/m) * ( (1 - (1+r)^(-T)) / r ) ]
        
        discounted_face = present_value(face_value, r, T)
        annuity_factor = annuity_pv(1, r, T)  # i.e., PV of 1 each period
        # but note that the coupon each period is (c * F / m),
        # so we multiply the annuity factor by (F/m) and then solve for c.
        
        numerator = bond_price - discounted_face
        denominator = (face_value / coupons_per_year) * annuity_factor
        # c = annual coupon rate in decimal
        c = 0.0
        try:
            c = numerator / denominator
        except ZeroDivisionError:
            st.error("Could not compute coupon rate (division by zero). Check inputs.")
        
        st.write(f"Annual Coupon Rate (decimal) = {c:.6f}")
        st.success(f"Annual Coupon Rate (percent) = {c * 100:.4f}%")

elif formula == "Present Value (PV)":
    st.subheader("Present Value")
    fv_amt = st.number_input("Future Value to be discounted", value=1000.0, step=1.0)
    r = st.number_input("Interest/Discount Rate (decimal)", value=0.10, 
                        step=1e-12, format="%.12f")
    n = st.number_input("Number of Periods (n)", value=5, step=1)
    if st.button("Calculate PV"):
        pv = present_value(fv_amt, r, n)
        st.success(f"Present Value = {pv:,.6f}")

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
        
        # CFs: CF0 = -bond_price, CF1..CF(T-1)=C, CF(T)=C+face_value
        cf_list = [-bond_price] + [C]*(T-1) + [C + face_value]
        
        period_irr = irr_calc(cf_list, guess=0.05) 
        annual_irr = period_irr * coupons_per_year
        
        st.write(f"Periodic YTM = {period_irr*100:,.6f}% per period")
        st.success(f"Annualized YTM = {annual_irr*100:,.6f}% per year")

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
    st.write("Enter a sequence of cash flows (including the initial negative cost at t=0).")
    num_flows = st.number_input("Number of cash flow periods (including t=0)?", 
                                value=3, step=1)
    
    cf_list = []
    for i in range(num_flows):
        val = st.number_input(f"CF at time t={i}", value=0.0, step=1.0, key=f"cf_{i}")
        cf_list.append(val)
    
    # Optional salvage inputs
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
        if salvage_value != 0:
            cf_list[salvage_period] += salvage_value

        result = npv_calc(cf_list, r)
        st.success(f"NPV = {result:,.6f}")
        if result > 0:
            st.info("NPV > 0, Accept the project!")
        else:
            st.warning("NPV <= 0, Reject the project.")

elif formula == "Internal Rate of Return (IRR)":
    st.subheader("Internal Rate of Return")
    st.write("Enter a sequence of cash flows (including the initial negative cost at t=0).")
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
    st.write("Enter a sequence of cash flows (including the initial negative cost at t=0).")
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
    st.write("Enter a sequence of cash flows (including the initial negative cost at t=0).")
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
