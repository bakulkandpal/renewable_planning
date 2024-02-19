import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *
import numpy as np

day=151 # Chosen day of the year for analysis.
house_type = 'H3'  # Choose house_type from H1 to H9
df = pd.read_excel('H03.xlsx', sheet_name='Final', nrows=35040)  # Change excel for data.
pv_kw_cost=560  # In Euro. Per kW cost of PV.
wind_kw_cost=360  # In Euro. Per kW cost of wind installation.
E_rated=0.8 # BESS capacity size in kWh. BESS assumed to have 50% initial state of charge.
EV_Charger_Capacity=7  # Single EV charging power in kW
preferred_start_hours = [1,2,3,4,5,6,7,8,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20,21,22]


penalty_obj_flex=1e2  # Lagrangian penalty to satisfy flexible demand constraint.
df_solar = pd.read_csv('ninja_pv_spain.csv')
df_price=pd.read_csv('Day-ahead Prices_Spain.csv')
df_price.index = pd.date_range(start='2023-01-01', periods=len(df_price), freq='H')
df_price = df_price.resample('15T').ffill()
df_solar.index = pd.date_range(start='2023-01-01', periods=len(df_solar), freq='H')
df_solar = df_solar.resample('15T').ffill()
df_wind = pd.read_csv('Wind data - Spain.csv')
df_car_usage=pd.read_excel('daily_car_usage.xlsx')
house_col_index = df_car_usage.columns.get_loc([col for col in df_car_usage.columns if house_type in col][0])
house_ev_data = df_car_usage.iloc[:, house_col_index:house_col_index+2]
df_car_pattern = pd.read_excel('car_usage_profiles.xlsx', sheet_name='Sheet1')
#spot_price_15min = np.random.uniform(low=0.9, high=2, size=96)  
multiplier = {'H1': 5, 'H2': 17, 'H3': 18, 'H4': 3, 'H5': 5, 
    'H6': 6, 'H7': 6, 'H8': 5, 'H9': 10}
factor = multiplier[house_type]

df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
specific_day_data = house_ev_data.iloc[day]
energy_data_list = specific_day_data.tolist()

day_data = df.iloc[0+(96*(day-1)):96+(96*(day-1))]
solar_data = df_solar.iloc[0+(96*(day-1)):96+(96*(day-1))]
df_wind.index = pd.date_range(start='2023-01-01', periods=len(df_wind), freq='H')
df_wind_15min = df_wind.resample('15T').ffill()
wind_data = df_wind_15min.iloc[0+(96*(day-1)):96+(96*(day-1))]
df_car_pattern_data=df_car_pattern.iloc[0+(96*(day-1)):96+(96*(day-1))]
df_price_day=df_price.iloc[0+(96*(day-1)):96+(96*(day-1))]/1000
spot_price_15min=df_price_day["Day-ahead Price [EUR/MWh]"].reset_index(drop=True)

washing_machine_kwh = day_data["Washing machine (kWh)"]
dishwasher_kwh = day_data["Dishwasher (kWh)"]
dryer_kwh = day_data["Dryer (kWh)"] 
total_demand_15min = washing_machine_kwh + dishwasher_kwh + dryer_kwh
total_demand_15min = total_demand_15min.fillna(0)
total_demand_15min_adjust = total_demand_15min*factor
fixed_demand=day_data["Non-Flexible appliances (kWh)"].reset_index(drop=True)
fixed_demand = fixed_demand*factor
solar_day_data = solar_data['electricity'].reset_index(drop=True)
wind_day_data = wind_data["Power 4400 W"] / 4400  # Scenario 1 Wind
#wind_day_data = wind_data["Power 8800 W"] / 8800  # Scenario 2 Wind
wind_day_data = wind_day_data.reset_index(drop=True) 
car_pattern_day_data = df_car_pattern_data["Profile_1"].reset_index(drop=True)

car_available=car_pattern_day_data.tolist()
car_available = [value * EV_Charger_Capacity * factor for value in car_available]

#print(car_available)

positive_values = total_demand_15min_adjust[total_demand_15min_adjust > 0]
operation_duration = len(positive_values)


preferred_start_intervals = [(h * 4) for h in preferred_start_hours if (h * 4) <= (96 - operation_duration)]

np.random.seed(41)
pv_profile_15min = solar_day_data
wind_profile_15min = wind_day_data  

model = ConcreteModel()
model.T = RangeSet(0, 95) 
model.Intervals = RangeSet(0, 95)
ev_number=len(energy_data_list)
model.ev_num=RangeSet(0, ev_number-1)
model.pv_capacity = Var(domain=NonNegativeReals)  # Capacity in kW
model.wind_capacity = Var(domain=NonNegativeReals)  
model.grid_energy = Var(model.T, domain=NonNegativeReals)  
model.flexible_demand = Var(model.T, domain=NonNegativeReals)  
model.x = Var(model.Intervals, within=Binary)
model.bat=Var(model.Intervals, within=Binary)
model.Pchar=Var(model.ev_num,model.T, domain=NonNegativeReals)  
model.preferred_start = ConstraintList()
model.preferred_start.add(sum(model.x[interval] for interval in range(96)) == 1)
model.EnergyConserv = ConstraintList() 
model.power_balance = ConstraintList()
model.ev_energy=ConstraintList()
model.ev_power_bound=ConstraintList()
model.charge_state=ConstraintList()
model.discharge=ConstraintList()    
model.charge =ConstraintList()  
model.Rmax = Param(initialize=E_rated/2) 
model.Smax = Param(initialize=E_rated) 

eta = 0.95 # Efficiency
model.pin = Var(model.T, domain=NonNegativeReals )
model.pout = Var(model.T, domain=NonNegativeReals)
model.S = Var(model.T, bounds=(0, model.Smax))
 
for t in model.T:
    
        model.discharge.add(model.pout[t] <= model.Rmax*model.bat[t])
        
for t in model.T:
    
        model.charge.add( model.pin[t] <= model.Rmax*(1-model.bat[t]) )
for t in model.T:   
        if t == 0:
           model.charge_state.add(model.S[t] == model.Smax/2)
        else:
             model.charge_state.add(model.S[t] == (model.S[t-1]+(model.pin[t-1] * eta)-(model.pout[t-1] / eta)))
           
model.EnergyConserv.add(model.S[0] ==model.S[23])
def objective_rule(model):
    Objective_pv = pv_kw_cost * model.pv_capacity
    Objective_wind = wind_kw_cost * model.wind_capacity
    Objective_spot = sum(spot_price_15min[i] * model.grid_energy[i] for i in model.T)
    Objective_demand = sum(model.flexible_demand[i] for i in model.T)
   # Objective_ev=sum(model.Pchar[ev,i]*spot_price_15min[i] for ev in range(ev_number)for i in model.T)
    return Objective_pv + Objective_wind + (Objective_spot * 365 * 20)+penalty_obj_flex*Objective_demand
model.Objective = Objective(rule=objective_rule, sense=minimize)

for ev in range(ev_number):   
   model.ev_energy.add(sum(model.Pchar[ev,i] for i in model.T)==energy_data_list[ev]*factor)
   
for ev in range(ev_number):
    for i in model.T:
       model.ev_power_bound.add(model.Pchar[ev,i] <=car_available[i])

for interval in range(96):  
        t = interval  
        model.power_balance.add(model.pout[t]+model.pv_capacity * pv_profile_15min[t] + 
                                model.wind_capacity * wind_profile_15min[t] + 
                                model.grid_energy[t] == model.pin[t]+fixed_demand[t] + model.flexible_demand[t]+sum(model.Pchar[ev,t] for ev in range(ev_number)))

model.flex_demand_constraints = ConstraintList()
for t in range(96 - operation_duration + 1):  
        for op_hour_idx in range(operation_duration):  
            actual_hour = t + op_hour_idx         
            model.flex_demand_constraints.add(model.flexible_demand[actual_hour] >= model.x[t]*positive_values[op_hour_idx])

model.binary_restrictions = ConstraintList()
for i in model.Intervals:
    if i not in preferred_start_intervals:
        model.binary_restrictions.add(model.x[i] == 0)
                
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

pv_sol=value(model.pv_capacity)
wind_sol=value(model.wind_capacity)
grid_energy_values = [value(model.grid_energy[i]) for i in model.T]
x_values = [value(model.x[i]) for i in model.T]
battery_char=[value(model.pin[i]) for i in model.T]
battery_dis=[value(model.pout[i]) for i in model.T]
flex_demand_value=[value(model.flexible_demand[i]) for i in model.T]

start_interval = [i for i in model.Intervals if value(model.x[i]) == 1][0]

data = {i: {} for i in model.T}

for ev in model.ev_num:
    for i in model.T:
        Pchar_value = value(model.Pchar[ev, i])       
        data[i][f'EV{ev}'] = Pchar_value

Pchar_df = pd.DataFrame.from_dict(data, orient='index')

new_demand_profile = np.zeros(96)
for i, val in enumerate(positive_values):
    if start_interval + i < 96:
        new_demand_profile[start_interval + i] = val
flex_orig_value = total_demand_15min_adjust.tolist()

Objective_spot_value = sum(value(spot_price_15min[i]) * value(model.grid_energy[i]) for i in model.T)*365*20
Objective_wind_value = wind_kw_cost*wind_sol
Objective_PV_value = pv_kw_cost*pv_sol
values_plot = [Objective_spot_value, Objective_wind_value, Objective_PV_value]
labels = ['Spot Price', 'Wind', 'PV']

total_objective_value = Objective_spot_value + Objective_wind_value + Objective_PV_value
print("Total Objective Value:", total_objective_value)


plt.figure(figsize=(12, 6))
plt.plot(flex_orig_value, label='Original Total Demand', marker='o')
plt.plot(new_demand_profile, label='New Total Demand', linestyle='--', marker='x')
plt.plot(fixed_demand, label='Fixed Demand', linestyle='--')
plt.xlabel('Interval (15 min)')
plt.ylabel('Demand (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(pv_profile_15min*pv_sol, label='PV', marker='o')
plt.plot(wind_profile_15min*wind_sol, label='Wind', linestyle='--', marker='x')
plt.plot(grid_energy_values, label='Grid', linestyle='--', marker='x')
plt.xlabel('Interval (15 min)')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(Pchar_df['EV0'], label='All EV0', marker='o')
plt.plot(Pchar_df['EV1'], label='All EV1', linestyle='--', marker='x')
plt.xlabel('Interval (15 min)')
plt.ylabel('Charging (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))  
plt.plot(battery_char, label='Battery Charging', color='blue', linestyle='-', marker='o')
plt.plot(battery_dis, label='Battery Discharging', color='red', linestyle='--', marker='x')
plt.xlabel('Interval (15 min)')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))  
plt.plot(flex_demand_value, label='Flexible Optimization', color='blue', linestyle='-', marker='o')
plt.plot(flex_orig_value, label='Flexible Original', color='red', linestyle='--', marker='x')
plt.xlabel('Interval (15 min)')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(8,6))
plt.plot(wind_day_data, label='Wind per 1kW')
plt.plot(pv_profile_15min, label='PV per 1kW')
plt.xlabel('Interval (15 min)')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
plt.bar(labels, values_plot, color=['blue', 'green', 'orange'])
plt.ylabel('Cost [Euro]')
plt.show()



