import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *
import numpy as np

week=50 # Chosen week of the year for analysis.
house_type = 'H3'  # Choose house_type from H1 to H9
df = pd.read_excel('H03.xlsx', sheet_name='Final', nrows=35040)  # Change excel for data.
pv_kw_cost=560  # In Euro. Per kW cost of PV.
wind_kw_cost=360  # In Euro. Per kW cost of wind installation.
EV_Charger_Capacity=7  # Single EV charging power in kW
preferred_start_hours = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
EV_battery_capacity = 60 # A single EV battery capacity in kWh
Initial_SOC = 0.5  # Initial state of charge of battery of Each EV.


penalty_obj_flex=1  # Lagrangian penalty to satisfy flexible demand constraint.
eta = 1 # Efficiency EV charging/discharging
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

day_ev_charge={}
for day in range(1, 8): 
    day_con=(week-1)*7+day
    day_ev_charge[day] = house_ev_data.iloc[day_con]

week_data = df.iloc[0+(672*(week-1)):672+(672*(week-1))]
solar_data = df_solar.iloc[0+(672*(week-1)):672+(672*(week-1))]
df_wind.index = pd.date_range(start='2023-01-01', periods=len(df_wind), freq='H')
df_wind_15min = df_wind.resample('15T').ffill()
wind_data = df_wind_15min.iloc[0+(672*(week-1)):672+(672*(week-1))]
df_car_pattern_data=df_car_pattern.iloc[0+(672*(week-1)):672+(672*(week-1))]
df_price_day=df_price.iloc[0+(672*(week-1)):672+(672*(week-1))]/1000
spot_price_15min=df_price_day["Day-ahead Price [EUR/MWh]"].reset_index(drop=True)

washing_machine_kwh = week_data["Washing machine (kWh)"]
dishwasher_kwh = week_data["Dishwasher (kWh)"]
dryer_kwh = week_data["Dryer (kWh)"] 
total_demand_15min = washing_machine_kwh + dishwasher_kwh + dryer_kwh
total_demand_15min = total_demand_15min.fillna(0)
total_demand_15min_adjust = total_demand_15min*factor
fixed_demand=week_data["Non-Flexible appliances (kWh)"].reset_index(drop=True)
fixed_demand = fixed_demand*factor
solar_week_data = solar_data['electricity'].reset_index(drop=True)
wind_week_data = wind_data["Power 4400 W"] / 4400  # Scenario 1 Wind
#wind_week_data = wind_data["Power 8800 W"] / 8800  # Scenario 2 Wind
wind_week_data = wind_week_data.reset_index(drop=True) 
car_pattern_week_data = df_car_pattern_data["Profile_1"].reset_index(drop=True)

car_available=car_pattern_week_data.tolist()
car_available = [value * EV_Charger_Capacity * factor for value in car_available]

#print(car_available)

daily_positive_values = {}
for day in range(1, 8):  
    day_start_idx = (day - 1) * 96 
    day_end_idx = day * 96  
    daily_demand = total_demand_15min_adjust[day_start_idx:day_end_idx]
    daily_positive_values[day] = daily_demand[daily_demand > 0]
    
operation_day = {}
for day in daily_positive_values:
    operation_day[day] = len(daily_positive_values[day])

preferred_start_intervals_day = {}
for day in range(1, 8):
    operation_duration = operation_day[day]
    preferred_start_intervals_day[day] = [(h * 4) for h in preferred_start_hours if (h * 4) <= (96 - operation_duration)]

pv_profile_15min = solar_week_data
wind_profile_15min = wind_week_data  

model = ConcreteModel()
model.T = RangeSet(0, 671) 
model.Days = RangeSet(1, 7)
model.Intervals = RangeSet(0, 671)
ev_number=2
model.ev_num=RangeSet(0, 1)
model.pv_capacity = Var(domain=NonNegativeReals)  # Capacity in kW
model.wind_capacity = Var(domain=NonNegativeReals)  
model.charging_status = Var(model.ev_num, model.T, within=Binary)
model.grid_energy = Var(model.T, domain=NonNegativeReals)  
model.flexible_demand = Var(model.T, domain=NonNegativeReals)  
model.x = Var(model.Days, RangeSet(0, 95), within=Binary)
model.Pchar=Var(model.ev_num,model.T, domain=NonNegativeReals) 
model.Pdichar=Var(model.ev_num,model.T, domain=NonNegativeReals)
model.preferred_start = ConstraintList()
model.power_balance = ConstraintList()
model.ev_energy=ConstraintList()
model.ev_power_bound=ConstraintList()
model.flex_demand_constraints = ConstraintList()
model.binary_restrictions = ConstraintList()

model.SOC = Var(model.ev_num, model.T, bounds=(0.2, 1.0)) # 20% to 100% bound
model.SOC_evolution = ConstraintList() 


def objective_rule(model):
    Objective_pv = pv_kw_cost * model.pv_capacity
    Objective_wind = wind_kw_cost * model.wind_capacity
    Objective_spot = sum(spot_price_15min[i] * model.grid_energy[i] for i in model.T)
    Objective_demand = sum(model.flexible_demand[i] for i in model.T)
    ev_sum = sum(sum(model.Pchar[ev, t] for t in model.T) for ev in model.ev_num) + sum(sum(model.Pdichar[ev, t] for t in model.T) for ev in model.ev_num)
    return Objective_pv + Objective_wind + (Objective_spot * 52 * 20)+penalty_obj_flex*Objective_demand+ev_sum
model.Objective = Objective(rule=objective_rule, sense=minimize)


###### EV Constraints 
for day in range(1, 8): 
    energy_data_list = day_ev_charge[day].tolist()
    for ev in range(ev_number):   
        day_start_idx = (day - 1) * 96
        day_end_idx = day * 96
        current_day_intervals = range(day_start_idx, day_end_idx)
        model.ev_energy.add(sum(model.Pchar[ev, i] - model.Pdichar[ev, i] for i in current_day_intervals) == energy_data_list[ev] * factor)
           
for ev in model.ev_num:
    for i in model.T:
       model.ev_power_bound.add(model.Pchar[ev,i] <= car_available[i])  # G2V
       model.ev_power_bound.add(model.Pdichar[ev,i] <= car_available[i])  # V2G 
       

### Big M constraint for EV charging/discharging
for ev in model.ev_num:
    for i in model.T:
        model.ev_power_bound.add(model.Pchar[ev, i] <= 5000 * model.charging_status[ev, i])  
        model.ev_power_bound.add(model.Pdichar[ev, i] <= 5000 * (1 - model.charging_status[ev, i]))

#### Constraint for SOC between bounds        
for ev in model.ev_num:
    for i in model.T:
        if i == model.T.first():
            model.SOC_evolution.add(model.SOC[ev, i] == Initial_SOC)
        else:
            charging_energy = model.Pchar[ev, i-1] * eta * 0.25
            discharging_energy = model.Pdichar[ev, i-1] * 0.25 / eta 
            SOC_change = (charging_energy - discharging_energy) / (EV_battery_capacity * factor)
            model.SOC_evolution.add(model.SOC[ev, i] == model.SOC[ev, i-1] + SOC_change)        

        
#### Demand Generation Matching
for t in range(672):   
        model.power_balance.add(model.pv_capacity * pv_profile_15min[t] + model.wind_capacity * wind_profile_15min[t] + 
                                model.grid_energy[t] + sum(model.Pdichar[ev,t] for ev in range(ev_number)) == fixed_demand[t] + model.flexible_demand[t]+sum(model.Pchar[ev,t] for ev in range(ev_number)))


#### Demand Shifting Constraint
for day in range(1, 8):
    operation_duration=operation_day[day]
    positive_values = daily_positive_values[day].tolist()
    for t in range(96 - operation_duration + 1):  
        for op_hour_idx in range(operation_duration):  
            actual_hour = (day-1)*96 + t + op_hour_idx         
            model.flex_demand_constraints.add(model.flexible_demand[actual_hour] >= model.x[day,t]*positive_values[op_hour_idx])

#### Binary constraint
for day in range(1, 8):
    for i in range(96):
        if i not in preferred_start_intervals_day[day]:
            model.binary_restrictions.add(model.x[day,i] == 0)
    model.preferred_start.add(sum(model.x[day, interval] for interval in range(96)) == 1)    

                
solver = SolverFactory('gurobi')
results = solver.solve(model, tee=True)

pv_sol=value(model.pv_capacity)
wind_sol=value(model.wind_capacity)
grid_energy_values = [value(model.grid_energy[i]) for i in model.T]
x_values = [[value(model.x[d, i]) for i in range(96)] for d in model.Days]
flex_demand_value=[value(model.flexible_demand[i]) for i in model.T]
flex_orig_value = total_demand_15min_adjust.tolist()

preferred_start_index_per_day = [x_values[day].index(1) for day in range(len(x_values))]

filled_values_per_day = [0] * 672
for day in range(1, 8):
    preferred_start_index = preferred_start_index_per_day[day-1]
    positive_values = daily_positive_values[day].tolist()
    filled_values_per_day[(day-1)*96 + preferred_start_index : (day-1)*96 + len(positive_values) + preferred_start_index] = positive_values   
            
            
data = {i: {} for i in model.T}
for ev in model.ev_num:
    for i in model.T:
        Pchar_value = value(model.Pchar[ev, i])  
        Pdichar_value = value(model.Pdichar[ev, i])  
        data[i][f'EV{ev}_Pchar'] = Pchar_value  
        data[i][f'EV{ev}_Pdichar'] = Pdichar_value
Pchar_df = pd.DataFrame.from_dict(data, orient='index')



Objective_spot_value = sum(value(spot_price_15min[i]) * value(model.grid_energy[i]) for i in model.T)*52*20
Objective_wind_value = wind_kw_cost*wind_sol
Objective_PV_value = pv_kw_cost*pv_sol
values_plot = [Objective_spot_value, Objective_wind_value, Objective_PV_value]
labels = ['Spot Price', 'Wind', 'PV']

total_objective_value = Objective_spot_value + Objective_wind_value + Objective_PV_value
print("Total Objective Value:", total_objective_value)

charge_EV0 = Pchar_df['EV0_Pchar']
discharge_EV0 = Pchar_df['EV0_Pdichar']
charge_EV1 = Pchar_df['EV1_Pchar']
discharge_EV1 = Pchar_df['EV1_Pdichar']


plt.plot(filled_values_per_day, label='Flex Optimization', marker='o')
plt.plot(flex_orig_value, label='Original Flex', marker='x')
plt.xlabel('Interval (15 min)')
plt.ylabel('Demand (kW)')
plt.legend()


plt.figure(figsize=(12, 6))
plt.plot(flex_orig_value, label='Original Total Demand', marker='o')
plt.plot(flex_demand_value, label='New Total Demand', linestyle='--', marker='x')
plt.plot(fixed_demand, label='Fixed Demand', linestyle='--')
plt.xlabel('Interval (15 min)')
plt.ylabel('Demand (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(pv_profile_15min*pv_sol, label='PV')
plt.plot(wind_profile_15min*wind_sol, label='Wind', linestyle='--')
plt.plot(grid_energy_values, label='Grid', linestyle='--')
plt.xlabel('Interval (15 min)')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()


plt.figure(figsize=(8,6))
plt.plot(wind_week_data, label='Wind per 1kW')
plt.plot(pv_profile_15min, label='PV per 1kW')
plt.xlabel('Interval (15 min)')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()



fig, axs = plt.subplots(2, 2, figsize=(14, 10)) 
axs[0, 0].plot(charge_EV0.index, charge_EV0.values, marker='o', linestyle='', color='green')
axs[0, 0].set_title('Charging EV0')
axs[0, 0].set_xlabel('Interval (15 min)')
axs[0, 0].set_ylabel('Power (kW)')
axs[0, 1].plot(discharge_EV0.index, discharge_EV0.values, marker='x', linestyle='', color='red')
axs[0, 1].set_title('Discharging EV0')
axs[0, 1].set_xlabel('Interval (15 min)')
axs[0, 1].set_ylabel('Power (kW)')
axs[1, 0].plot(charge_EV1.index, charge_EV1.values, marker='o', linestyle='', color='green')
axs[1, 0].set_title('Charging EV1')
axs[1, 0].set_xlabel('Interval (15 min)')
axs[1, 0].set_ylabel('Power (kW)')
axs[1, 1].plot(discharge_EV1.index, discharge_EV1.values, marker='x', linestyle='', color='red')
axs[1, 1].set_title('Discharging EV1')
axs[1, 1].set_xlabel('Interval (15 min)')
axs[1, 1].set_ylabel('Power (kW)')
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
plt.bar(labels, values_plot, color=['blue', 'green', 'orange'])
plt.ylabel('Cost [Euro]')
plt.show()



