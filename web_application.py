import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
from pyxirr import xirr
from PIL import Image



st.title("Кредитный калькулятор")

img = Image.open('calculator.jpg')
img_resized = img.resize((600, 350))
st.image(img_resized, use_container_width = True)

# ----------------------->>>>>>>>>>>>  Initial Parameters  <<<<<<<<<<<------------------------------
start_date = st.date_input("Дата начала кредита:", value=datetime.today().date())
payment_period = st.number_input("Период платежа (в днях):", value=30, min_value=1)
loan_period = st.number_input("Срок кредита (в днях):", value=360, min_value=1)
amount = st.number_input("Сумма:", value=1000, min_value=0)
comission = st.number_input("Разовая комиссия (%):", value=5)
product_4_5 = st.checkbox("Продукт 4 / 5", value=False)
dynamic_body_paments = st.checkbox("Предусмотрены частичные погашения тела", value=False)
accrued_period = st.number_input("Начисления процентов раз в (дни):", value=1, min_value=1)
accrued_end = st.selectbox("Начисления по принципу:", options=[0, 1, 2],
                            format_func=lambda x: {0: "Начисления на начало периода",
                                                    1: "Начисления на конец периода",
                                                    2: "Начисления по принципу нашего продукта"}[x],
    index=2)

st.subheader("Введите параметры для интервалов:")
num_intervals = st.number_input("Количество интервалов:", value=3, min_value=1, max_value=30, step=1)

days = [0] * num_intervals
rates = [0.0000] * num_intervals
supports = [0.0000] * num_intervals
bodies = [0] * num_intervals

for i in range(num_intervals):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        days[i] = st.number_input(f"К-во дней, # {i + 1}:", value=days[i], min_value=0)

    with col2:
        rates[i] = st.number_input(f"Ставка, # {i + 1} (%)", value=rates[i], min_value=0.0000, format="%.4f")

    with col3:
        supports[i] = st.number_input(f"Комиссия за обслуживание, # {i + 1} (%)", value=supports[i], min_value=0.0000, format="%.4f")

    with col4:
        bodies[i] = st.number_input(f"Процент от тела для интервала №{i + 1}:", value=bodies[i], min_value=0)

interval_params = [([days, rates, supports, bodies])]

# ---------------------------->>>>>>>>>   Functions <<<<<<< ---------------------------------------
def payment_calendar_builder(start_date, payment_period, loan_period):
    """
    Generate a payment calendar DataFrame based on a start date, payment period, and loan period.

    Args:
        start_date (str or datetime): The start date of the loan, either as a string in the format '%Y-%m-%d'
                                      or a `datetime` object.
        payment_period (int): The number of days between each payment.
        loan_period (int): The total duration of the loan in days.

    Returns:
        pd.DataFrame: A DataFrame with a single column 'payment_date' containing the scheduled payment dates.
                      The last date corresponds to the loan end date.

    Example:
        >>> payment_calendar_builder('2025-01-01', 30, 120)
              payment_date
        0  2025-01-01
        1  2025-01-31
        2  2025-03-02
        3  2025-05-01
    """
    if not isinstance(start_date, str):
        start_date = start_date.strftime('%Y-%m-%d')
        
    date = datetime.strptime(start_date, '%Y-%m-%d')
    payment_calendar = [date]

    end_date = date + timedelta(days=loan_period)

    while date < end_date:
        date = date + timedelta(days=payment_period)
        if date < end_date:
            payment_calendar.append(date)
    
    payment_calendar.append(end_date)

    df = pd.DataFrame(payment_calendar, columns=['payment_date'])
    return df

def build_payments_schema(interval_params, comission, payment_period, loan_period):
    total_sum = 0
    period = payment_period
    local_comission = comission
    percents_per_year = loan_period - 3
    coeficients, rates, supports, comissions, bodies, sums = [], [], [], [], [], []
    
    for days, rate, support, body in interval_params:
        days[0] = max(days[0], payment_period)
        i = 0
        body_list = []
        
        while total_sum < percents_per_year and i < len(days):
            days_ = days[i]
            coef_ = days_ // period
            rate_ = rate[i]
            support_ = support[i]
            body_ = body[i]
            
            if i == 0:
                local_comission = comission
            else:
                local_comission = 0
            
            coeficients.append(coef_)
            rates.append(rate_)
            supports.append(support_)
            comissions.append(local_comission)
            bodies.append(body_)
            
            if sum(bodies) >= 100:
                body_list = []
                for body_val in bodies:
                    body_list.append(body_val)
                    if sum(body_list) >= 100:
                        break
                body_list = body_list[:-1]
                while len(body_list) < len(bodies):
                    body_list.append(0.0)
            else:
                body_list = bodies[:]
            
            if i == 0:
                sum_i = rate_ * coef_ * period + support_ * coef_ * period + local_comission
            else:
                sum_i = rate_ * (1 - sum(body_list[:i]) / 100) * coef_ * period + support_ * (1 - sum(body_list[:i]) / 100) * coef_ * period
            
            sums.append(sum_i)
            total_sum = sum(sums)
            
            i += 1
    
    base_df = pd.DataFrame({
        'Коэфициент': coeficients,
        'Ставка': rates,
        'Комиссия за обслуживание': supports,
        'Разовая комиссия': comissions,
        'Проценты начисленые': sums,
        'Сума по телу в процентах': body_list
    })
    
    if total_sum > percents_per_year and len(days) > 1:
        last_row = base_df.iloc[-1].copy()
        last_row['Проценты начисленые'] = percents_per_year - sum(base_df.iloc[:-1]['Проценты начисленые'])
        
        last_row['Коэфициент'] = (
            last_row['Проценты начисленые'] / (
                last_row['Ставка'] * (1 - sum(base_df.iloc[:-1]['Сума по телу в процентах'] / 100)) 
                + last_row['Комиссия за обслуживание'] * (1 - sum(base_df.iloc[:-1]['Сума по телу в процентах'] / 100))
                + last_row['Разовая комиссия'] * (1 - sum(base_df.iloc[:-1]['Сума по телу в процентах'] / 100))
            ) / period
        ).astype(int)  
        last_row['Проценты начисленые'] = (last_row['Коэфициент'] * period * last_row['Ставка'] +
                                           last_row['Коэфициент'] * period * last_row['Комиссия за обслуживание'] +
                                           last_row['Коэфициент'] * period * last_row['Разовая комиссия']
                                          ) * (1 - sum(base_df.iloc[:-1]['Сума по телу в процентах'] / 100))
        
        base_df = base_df.iloc[:-1]
        base_df = pd.concat([base_df, pd.DataFrame([last_row])], ignore_index=True)
        
    elif total_sum > percents_per_year and len(days) == 1:
        base_df['Коэфициент'] = (percents_per_year -  local_comission) / (period  * (base_df['Ставка'] + base_df['Комиссия за обслуживание']))//1
        
        base_df['Проценты начисленые'] = (
        base_df['Коэфициент'] * period * base_df['Ставка'] +
        base_df['Коэфициент'] * period * base_df['Комиссия за обслуживание'] +
        local_comission
        )   
    else:
        base_df = base_df
    
    base_df = base_df[base_df['Коэфициент'] != 0]
        
    
    return base_df


def adding_dates(df, start_date, payment_period):
    period = payment_period
    if isinstance(start_date, str):
        date = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        date = start_date

    df.loc[0, 'date'] = date + timedelta(days=int(df.loc[0, 'Коэфициент'] * period))

    for i in range(1, len(df)):
        df.loc[i, 'date'] = df.loc[i-1, 'date'] + timedelta(days=int(df.loc[i, 'Коэфициент'] * period))

    date_ranges = []

    date_range = pd.date_range(start=date, end=df.loc[0, 'date']).tolist()
    date_ranges.append(date_range)

    for i in range(1, len(df)):
        start = df.loc[i-1, 'date'] + timedelta(days=1)
        end = df.loc[i, 'date']
        date_range = pd.date_range(start=start, end=end).tolist()
        date_ranges.append(date_range)

    df['date_ranges'] = date_ranges
    return df

def last_rate_calculation(df, payment_calendar):
    rest_percent = ((0.98 * (loan_period)) - sum(df['Проценты начисленые']))*(1 - sum(df['Сума по телу в процентах']) / 100)
  # rest_percent = ((0.98 * (loan_period + 1)) - sum(df['Проценты начисленые']))*(1 - sum(df['Сума по телу в процентах']) / 100)
    # rest_percent = ((loan_period - 3) - sum(df['Проценты начисленые']))*(1 - sum(df['Сума по телу в процентах']) / 100)

    max_date_df = pd.to_datetime(max(df.date))
    max_date_payment_date = pd.to_datetime(max(payment_calendar.payment_date))

    rest_days = (max_date_payment_date - max_date_df).days
    if rest_days == 0:
        last_percent = 0 
    else:
        last_percent = ((rest_percent / rest_days)*100//1)/100
    return last_percent

def product_constructor(df, payment_calendar_df, amount, comission, payment_period, last_rate):
    for i in range(1, len(payment_calendar_df)):
        for j in range(len(df)):
            if payment_calendar_df.loc[i, 'payment_date'] in df.loc[j, 'date_ranges']:
                payment_calendar_df.loc[i, 'rate'] = df.loc[j, 'Ставка']
                payment_calendar_df.loc[i, 'support'] = df.loc[j, 'Комиссия за обслуживание']

    if last_rate > 0.01:
        payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')] = payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')].fillna(last_rate)
    else:
        payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')] = payment_calendar_df.iloc[1:, payment_calendar_df.columns.get_loc('rate')].fillna(0.01)

    payment_calendar_df['comission'] = 0.0
    payment_calendar_df.loc[1, 'comission'] = comission
    payment_calendar_df = payment_calendar_df.fillna(0)
    return payment_calendar_df


def result_function (dynamic_body_paments, interval_params, start_date, payment_period, loan_period, amount, comission):
    df = build_payments_schema (interval_params, comission, payment_period, loan_period)
        
    df_dates = adding_dates (df, start_date, payment_period)

    payment_calendar_df = payment_calendar_builder (start_date, payment_period, loan_period)

    last_rate = last_rate_calculation(df_dates, payment_calendar_df)

    final = product_constructor(df_dates, payment_calendar_df, amount, comission, payment_period, last_rate)
    return final, df

def collect_end_of_month_dates(start_date, loan_period):

    date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = date + timedelta(days=loan_period)
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Generate a range of dates from start to end, with a frequency of 'M' (month-end)
    month_end_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    # Convert to list of date objects (without time)
    month_end_dates_list = [date.date() for date in month_end_dates]
    timestamp_list = [pd.Timestamp(date) for date in month_end_dates_list]

    end_dates_df = pd.DataFrame(timestamp_list, columns=['payment_date'])
    
    return end_dates_df

# xnpv
def xnpv(rate, values, dates):
    '''Equivalent of Excel's XNPV function.

    >>> from datetime import date
    >>> dates = [date(2010, 12, 29), date(2012, 1, 25), date(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xnpv(0.1, values, dates)
    -966.4345...
    '''
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])

# adding data_ranges
def adding_date_ranges (df):
    date_ranges = []
    for i in range(1, len(df)):
        start_date = df.loc[i - 1, 'payment_date'] + pd.Timedelta(days=1)
        end_date = df.loc[i, 'payment_date']
        date_ranges.append(pd.date_range(start=start_date, end=end_date).tolist())
    
    date_ranges.insert(0, [])
    
    df['date_ranges'] = date_ranges
    return df

def rates_schema (product_4_5, dynamic_body_paments, interval_params, start_date, payment_period, loan_period, amount, comission):
    if product_4_5:
        df, payment_schema = result_function (dynamic_body_paments, interval_params, start_date, payment_period, loan_period, amount, comission)
        df.loc[df.index[-1], 'rate'] = 1.0
        df.loc[df.index[-2], 'rate'] = 1.0
        print("Product 4-5")
    else:
        df, payment_schema = result_function (dynamic_body_paments, interval_params, start_date, payment_period, loan_period, amount, comission)
        
    df =  adding_date_ranges (df)

    return {"df" : df, 
         "schema" : payment_schema}

def collect_payments(schema, amount):
    payments_df = schema[['date', 'Сума по телу в процентах']].copy()
    payments_df['payment_date'] = payments_df['date']
    payments_df['body_amount'] = amount * payments_df['Сума по телу в процентах'] / 100
    
    payments_df = payments_df[['payment_date', 'body_amount']]

    return payments_df

def calendar_body(dynamic_body_paments, schema, start_date, payment_period, loan_period, amount):
    payment_calendar = payment_calendar_builder(start_date, payment_period, loan_period)
    
    payment_calendar['payment_date'] = pd.to_datetime(payment_calendar['payment_date'], errors='coerce')

    if dynamic_body_paments:
        payments_df = collect_payments(schema, amount)
        payments_df['payment_date'] = pd.to_datetime(payments_df['payment_date'], errors='coerce')

        merged_df = pd.merge(payment_calendar, payments_df, on='payment_date', how='outer')

        merged_df.loc[0, 'body_amount'] = -amount
        merged_df.loc[merged_df.index[-1], 'body_amount'] = -merged_df['body_amount'].iloc[:-1].sum()

        merged_df = merged_df.fillna(0.0)
        calendar_body_df = merged_df
    else:
        payment_calendar['body_amount'] = 0.0
        payment_calendar.loc[0, 'body_amount'] = -amount
        payment_calendar.loc[payment_calendar.index[-1], 'body_amount'] = amount
        calendar_body_df = payment_calendar

    date_ranges = []
    for i in range(1, len(calendar_body_df)):
        start_date = calendar_body_df.loc[i - 1, 'payment_date'] + pd.Timedelta(days=1)
        end_date = calendar_body_df.loc[i, 'payment_date']
        date_ranges.append(pd.date_range(start=start_date, end=end_date).tolist())
    
    date_ranges.insert(0, [])
    
    calendar_body_df['date_ranges'] = date_ranges
    
    return calendar_body_df
    
def daily_rates_amount_schema (rates_df, body_df, accrued_period):
    one_day_df = payment_calendar_builder(start_date, accrued_period, loan_period)
    for i in range(1, len(one_day_df)):
        for j in range(len(rates_df)):
            if one_day_df.loc[i, 'payment_date'] in rates_df.loc[j, 'date_ranges']:
                one_day_df.loc[i, 'rate'] = rates_df.loc[j, 'rate']
                one_day_df.loc[i, 'support'] = rates_df.loc[j, 'support']
                    
    for i in range(len(one_day_df)):
        for j in range(len(rates_df)):
            if one_day_df.loc[i, 'payment_date'] == rates_df.loc[j, 'payment_date']:
                one_day_df.loc[i, 'comission'] = rates_df.loc[j, 'comission']
    
    for i in range(len(one_day_df)):
        for j in range(len(body_df)):
            if one_day_df.loc[i, 'payment_date'] == body_df.loc[j, 'payment_date']:
                one_day_df.loc[i, 'amount'] = body_df.loc[j, 'body_amount']
    
    one_day_df = one_day_df.fillna(0.0)
    
    return one_day_df

def shift_one_day_payment(one_day_df, amount, accrued_end, payment_period):
    if accrued_end == 1: # end period
        for i in range(1, len(one_day_df)):
                one_day_df.loc[i, 'rate_absolute'] = one_day_df.loc[i, 'rate'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
                one_day_df.loc[i,'support_absolute'] = one_day_df.loc[i, 'support'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
    elif accrued_end == 0: # beginning period
        for i in range(1, len(one_day_df)-1):
            one_day_df.loc[0, 'rate_absolute'] = one_day_df.loc[1, 'rate'] * amount / 100
            one_day_df.loc[0,'support_absolute'] = one_day_df.loc[1, 'support'] * amount / 100
            one_day_df.loc[i, 'rate_absolute'] = one_day_df.loc[i + 1, 'rate'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
            one_day_df.loc[i,'support_absolute'] = one_day_df.loc[i + 1, 'support'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
    elif accrued_end == 2:  # Custom period
        for i in range(1, payment_period):
            one_day_df.loc[0, 'rate_absolute'] = one_day_df.loc[1, 'rate'] * amount / 100
            one_day_df.loc[0, 'support_absolute'] = one_day_df.loc[1, 'support'] * amount / 100
            one_day_df.loc[i, 'rate_absolute'] = one_day_df.loc[i + 1, 'rate'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
            one_day_df.loc[i, 'support_absolute'] = one_day_df.loc[i + 1, 'support'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
        for i in range(payment_period + 1, len(one_day_df)):
            one_day_df.loc[i, 'rate_absolute'] = one_day_df.loc[i, 'rate'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
            one_day_df.loc[i, 'support_absolute'] = one_day_df.loc[i, 'support'] * (-one_day_df.loc[:i-1, 'amount'].sum()) / 100
        
    one_day_df['comission_absolute'] = one_day_df['comission'] * amount / 100
    one_day_df = one_day_df.fillna(0.0)
    return one_day_df

def rates_payment_calendar (rates_absolute, rates_df):
    rates_absolute_ = rates_absolute[['payment_date',
           'rate_absolute', 'support_absolute', 'comission_absolute']].copy()
    
    for i in range(len(rates_absolute_)):
        for j in range(len(rates_df)):
            if rates_absolute_.loc[i, 'payment_date'] in rates_df.loc[j, 'date_ranges']:
                rates_absolute_.loc[i, 'calendar_date'] = rates_df.loc[j, 'payment_date']
    
    rates_absolute_.loc[0, 'calendar_date'] = rates_absolute_.loc[1, 'calendar_date']
    
    rates_absolute_ = rates_absolute_.drop(columns=['payment_date'])
    
    rates_absolute_grouped = rates_absolute_.groupby('calendar_date', as_index=False).sum()
    
    return rates_absolute_grouped

def calculating_cf (rates_payments, body_df):
    merged_df = pd.merge(
        body_df,
        rates_payments,
        left_on='payment_date',
        right_on='calendar_date',
        how='outer'
    )
    
    merged_df = merged_df.drop(columns=['calendar_date', 'date_ranges'])
    merged_df = merged_df.fillna(0.0)
    
    merged_df['CF'] = merged_df['body_amount'] + merged_df['rate_absolute'] + merged_df['support_absolute'] + merged_df['comission_absolute']
    return merged_df

def first_part_builder (start_date, 
                        payment_period,
                        loan_period,
                        amount,
                        comission,
                        product_4_5, 
                        interval_params,
                        dynamic_body_paments,
                        accrued_period,
                        accrued_end):
    rates = rates_schema (product_4_5, dynamic_body_paments, interval_params, start_date, payment_period, loan_period, amount, comission)
    
    body_df = calendar_body (dynamic_body_paments, rates['schema'], start_date, payment_period, loan_period, amount)
    
    daily_schema = daily_rates_amount_schema (rates["df"], body_df, accrued_period)
    
    rates_absolute = shift_one_day_payment(daily_schema, amount, accrued_end, payment_period)
    
    rates_payments = rates_payment_calendar(rates_absolute, rates["df"])
    cf_df = calculating_cf(rates_payments, body_df)
    cf_df.rename(columns={'payment_date': 'payment_date', 
                        'CF': 'CF',
                         'body_amount': 'CF_main_loan',
                        'rate_absolute': 'CF_interest',
                        'support_absolute': 'CF_support',
                        'comission_absolute': 'CF_comission'}, inplace=True)
    return cf_df, rates_absolute, rates["schema"] 

def collect_end_of_month_dates(start_date, loan_period):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    if isinstance(start_date, datetime):
        start_date = start_date.date() 
    
    end_date = start_date + timedelta(days=loan_period)
    
    month_end_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    month_end_dates_list = [date.date() for date in month_end_dates]
    
    timestamp_list = [pd.Timestamp(date) for date in month_end_dates_list]
    
    end_dates_df = pd.DataFrame(timestamp_list, columns=['payment_date'])
    
    return end_dates_df
    
def append_end_month (start_date, loan_period, payment_calendar_df):
    end_dates_df = collect_end_of_month_dates(start_date = start_date, loan_period = loan_period)
    
    end_dates_df['CF'] = 0.0
    end_dates_df['CF_main_loan'] = 0.0
    end_dates_df['CF_comission'] = 0.0
    end_dates_df['CF_support'] = 0.0
    end_dates_df['CF_interest'] = 0.0
    
    merged = pd.concat([payment_calendar_df, end_dates_df], ignore_index=True)
    sorted = merged.sort_values(by='payment_date', ascending=True).reset_index(drop=True)
    
    sorted.CF = sorted.CF_main_loan + sorted.CF_comission + sorted.CF_interest + sorted.CF_support
    idx = sorted.groupby('payment_date')['CF_interest'].idxmax()
    
    result_df = sorted.loc[idx].reset_index(drop=True)
    
    return result_df

def second_part_df_builder (main_part, daily_percents_sum, start_date, loan_period, comission, amount):
    main_part_full = append_end_month (start_date, loan_period, main_part)
    first_part_df = main_part_full
    end_of_month_dates_list = first_part_df['payment_date'].dt.date.tolist()
    
    end_m_dates = first_part_df[(first_part_df['CF_interest'] == 0) & (
        first_part_df['payment_date'].dt.date.isin(end_of_month_dates_list))]['payment_date'].tolist()[1:]
    
    first_part_df['XNPV'] = 0.0
    
    first_part_df.loc[0, 'XNPV_principal_loan_amount'] = amount
    for i in range(1,len(first_part_df)):
        first_part_df.loc[i, 'XNPV_principal_loan_amount'] = first_part_df.loc[i - 1, 'XNPV_principal_loan_amount'] - first_part_df.loc[i, 'CF_main_loan']
        
    first_part_df['XNPV_accrued_interest'] = 0.0
    
    first_part_df['XNPV_unamortized discount/premium'] = 0.0
    first_part_df.loc[0, 'XNPV_unamortized discount/premium'] = round(- (comission * amount / 100), 2)
    
    first_part_df['XNPV_reserve'] = 0.0
    
    first_part_df.loc[0, 'XNPV'] = (first_part_df.loc[0,'XNPV_principal_loan_amount'] + 
                                    first_part_df.loc[0, 'XNPV_accrued_interest'] +
                                    first_part_df.loc[0, 'XNPV_unamortized discount/premium'] + 
                                    first_part_df.loc[0, 'XNPV_reserve'] 
                                   )
    
    # xirr
    dates = first_part_df['payment_date'].to_list()
    date_list = [date_obj.date() for date_obj in dates]
    amounts = first_part_df['CF'].to_list()
        
    xirr_ = xirr(pd.DataFrame({"dates": date_list, "amounts": amounts}))
    xirr_day = (1+xirr_)**(1/365)-1   
    
    # xnpv
    for i in range(1, len(first_part_df)):
        dates = first_part_df['payment_date'][i:].to_list()
        amounts = first_part_df['CF'][i:].to_list()
        date_list = [date_obj.date() for date_obj in dates]
            
        first_part_df.loc[i, 'XNPV'] = xnpv(xirr_, amounts, date_list) - first_part_df.loc[i, 'CF']
        
    first_part_df['payment_diff'] = first_part_df['payment_date'].diff().dt.days.fillna(0).astype(int)
    
    first_part_df = adding_date_ranges (first_part_df)
    
    percents_accrued_df = rates_payment_calendar(daily_percents_sum, first_part_df)
    percents_accrued_df['sum_payments'] = (
        percents_accrued_df['rate_absolute'] + 
        percents_accrued_df['support_absolute']
        )
    
    merged_df = first_part_df.merge(
        percents_accrued_df[['calendar_date', 'sum_payments']],
        left_on='payment_date',
        right_on='calendar_date',
        how='left'
    )
    merged_df = merged_df.fillna(0.0)
    
    for i in range(1,len(merged_df)):
        merged_df.loc[i, 'XNPV_accrued_interest'] = (merged_df.loc[0:i, 'sum_payments'].sum() - 
                                                     merged_df.loc[0:i, 'CF_interest'].sum() - 
                                                     merged_df.loc[0:i, 'CF_support'].sum()
                                                    )
    
    
    merged_df.loc[1:, 'XNPV_unamortized discount/premium'] = (merged_df.loc[1:, 'XNPV'] - 
                                                     merged_df.loc[1:, 'XNPV_principal_loan_amount'] - 
                                                     merged_df.loc[1:, 'XNPV_accrued_interest']
                                                    )
    
    
    value = sum(merged_df.CF) + amount
    cost = sum(merged_df.CF)
    fist_payment = merged_df['CF'][2] if merged_df['CF'][1] == 0 else merged_df['CF'][1]


    results = {
    "XIRR год": round(xirr_ * 100, 2),
    "XIRR день": round(xirr_day * 100, 2),
    "Платеж в первый период" : fist_payment,
    "Прибыль в первый период" : round(fist_payment / amount * 100, 2),
    "Ежедневная прибыль в первый период" : round((fist_payment / amount * 100) / payment_period, 4),
    "Общие расходы по кредиту": round(cost, 2),
    "Общая стоимость по кредита": round(value, 2),
    "Среднедневная номинальная ставка": round(cost / amount / loan_period * 100, 4)
    }

    metrics_df = pd.DataFrame([results])

    st.write("-------------------------------- >>>>>>> Расчетные показатели <<<<<< -----------------------------------")
    st.write(f"XIRR год: {round(xirr_ * 100, 2)}%")
    st.write(f"XIRR день: {round(xirr_day * 100, 2)}%")
    st.write(f"Платеж в первый период : {fist_payment}")
    st.write(f"Прибыль за первый период : {round(fist_payment / amount * 100, 2)}%")
    st.write(f"Ежедневная прибыль в первый период : {round((fist_payment / amount * 100) / payment_period, 4)}%")
    st.write(f"Общие расходы по кредиту: {round(cost, 2)} грн.")
    st.write(f"Общая стоимость кредита: {round(value, 2)} грн.")
    st.write(f"Среднедневная номинальная ставка: {round(cost / amount / loan_period * 100, 4)}%")
    st.write(f"Всего проценты: {round(sum(merged_df.CF_interest),2)} грн.")
    st.write(f"Всего комиссия за обслуживание: {round(sum(merged_df.CF_support),2)} грн.")
    st.write(f"Всего комиссия за выдачу: {round(sum(merged_df.CF_comission),2)} грн.")
    
    return merged_df, metrics_df

def third_part_df (merged_df):
    second_part_df = merged_df
    second_part_df['effective_interest_total'] = 0.0
    second_part_df['nominal_interest_total'] = second_part_df['sum_payments']
    second_part_df['discount_total'] = 0.0
    
    for i in range(1, len(second_part_df)):
        second_part_df.loc[i, 'discount_total'] = second_part_df.loc[i, 'XNPV_unamortized discount/premium'] - second_part_df.loc[i-1, 'XNPV_unamortized discount/premium']
    
        second_part_df.loc[i, 'effective_interest_total'] = second_part_df.loc[i, 'nominal_interest_total'] + second_part_df.loc[i, 'discount_total']
    
    final_df = second_part_df
    return final_df


def rename_cut (final_df):
    final_df.rename(columns={'payment_date': 'Дата', 
                         'CF': 'Денежные потоки - Всего',
                        'CF_main_loan': 'Сумма кредита',
                        'CF_comission': 'Первоначальная комиссия (дисконт)',
                        'CF_interest': 'Проценты',
                        'CF_support': 'Комиссия за обслуживание',
                        'XNPV': 'Балансовая стоимость (амортизированная себестоимость) - Всего',
                        'XNPV_principal_loan_amount': 'Тело',
                        'XNPV_accrued_interest': 'Начисленные проценты',
                        'XNPV_unamortized discount/premium': 'Неамортизированный дисконт/премия',
                        'XNPV_reserve': 'Резерв',
                        'payment_diff': 'Расчетный период',
                        'effective_interest_total': 'Всего по эфективной ставке',
                        'nominal_interest_total': 'Всего по номинальной ставке',
                        'discount_total': 'Корректировка доходов (амортизация)'}, inplace=True)
    
    final_df = final_df[['Дата', 'Денежные потоки - Всего', 'Сумма кредита',
       'Первоначальная комиссия (дисконт)', 'Проценты', 'Комиссия за обслуживание',
       'Балансовая стоимость (амортизированная себестоимость) - Всего', 'Тело',
       'Начисленные проценты', 'Неамортизированный дисконт/премия', 'Резерв',
       'Всего по эфективной ставке',
       'Всего по номинальной ставке', 'Корректировка доходов (амортизация)'
                        ]]

    columns_to_round = [
    'Денежные потоки - Всего', 'Сумма кредита', 'Первоначальная комиссия (дисконт)',
    'Проценты', 'Комиссия за обслуживание', 'Балансовая стоимость (амортизированная себестоимость) - Всего',
    'Тело', 'Начисленные проценты', 'Неамортизированный дисконт/премия', 'Резерв',
    'Всего по эфективной ставке', 'Всего по номинальной ставке', 'Корректировка доходов (амортизация)'
    ]
    
    final_df.loc[:, columns_to_round] = final_df[columns_to_round].round(2)

    return final_df

def function_builder (start_date, 
                        payment_period,
                        loan_period,
                        amount,
                        comission,
                        product_4_5, interval_params,
                        dynamic_body_paments,
                        accrued_period,
                        accrued_end):
    cf_df, rates_absolute, schema = first_part_builder (start_date, 
                        payment_period,
                        loan_period,
                        amount,
                        comission,
                        product_4_5, interval_params,
                        dynamic_body_paments,
                        accrued_period,
                        accrued_end)
    second_df, metrics_df = second_part_df_builder (cf_df, rates_absolute, start_date, loan_period, comission, amount)
    third_df = third_part_df (second_df)
    final = rename_cut (third_df)
    return {"final" : final,
            "daily_rates" : rates_absolute,
            "schema" : schema,
            "metrics" : metrics_df
           }

def create_excel_file(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Payments', index=False)
    output.seek(0)
    return output

calculate_button = st.button("Рассчитать")

# ---------------------------->>>>>>>>>   Results <<<<<<< ---------------------------------------
if calculate_button:
    st.subheader("Показатели")
    data = function_builder (start_date, 
                            payment_period,
                            loan_period,
                            amount,
                            comission,
                            product_4_5, interval_params,
                            dynamic_body_paments,
                            accrued_period,
                            accrued_end)
    
    # st.dataframe(data['metrics'])
    metrics = create_excel_file(data['metrics'])
    st.download_button(
        label="Скачать Excel",
        data=metrics,
        file_name="metrics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.subheader("Схема начислений")
    st.dataframe(data['schema'])
    schema = create_excel_file(data['schema'])
    st.download_button(
        label="Скачать Excel",
        data=schema,
        file_name="payments_schema.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.subheader("Ежедневные начисления")
    st.dataframe(data['daily_rates'])
    rates_absolute = create_excel_file(data['daily_rates'])
    st.download_button(
        label="Скачать Excel",
        data=rates_absolute,
        file_name="rates_absolute.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("Таблица платежей")
    st.dataframe(data['final'])
    excel_file = create_excel_file(data['final'])
    st.download_button(
        label="Скачать Excel",
        data=excel_file,
        file_name="payments.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
