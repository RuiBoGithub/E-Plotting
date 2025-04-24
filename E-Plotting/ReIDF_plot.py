import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.size': 14,                # Default font size
    'axes.titlesize': 18,           # Title font size
    'axes.labelsize': 16,           # X and Y labels font size
    'xtick.labelsize': 16,          # X-axis tick labels font size
    'ytick.labelsize': 16,          # Y-axis tick labels font size
    'legend.fontsize': 16,          # Legend font size
    'legend.title_fontsize': 14,    # Legend title font size
    'figure.titlesize': 20          # Figure title font size (if using plt.suptitle)
})



def load_and_prepare_data(filepath): 
    df = pd.read_csv(filepath)
    date_str, time_str = val.strip().split()
    if time_str == "24:00:00":
        month, day = map(int, date_str.split("/"))
        new_date = pd.Timestamp(year=2020, month=month, day=day) + pd.Timedelta(days=1)
        return new_date.replace(hour=0, minute=0, second=0)
    else:
        return pd.to_datetime(f"2020/{date_str} {time_str}", format="%Y/%m/%d %H:%M:%S")

    df['Date/Time'] = df['Date/Time'].apply(fix_datetime)
    df['Month'] = df['Date/Time'].dt.month
    df_data = df.drop(columns=["Date/Time"]).copy()
    df_data = df_data.apply(pd.to_numeric, errors='coerce')
    df_data['Month'] = df['Month']
    return df, df_data


def process_energy_data(csv_file, verbose=True):
    df = pd.read_csv(csv_file)
    
    # Print the column names for debugging if verbose is True.
    if verbose:
        print(f"\nColumns in {csv_file}:")
        print(df.columns.tolist())
    
    # Identify a Date/Time column (using any column that contains "Date" or "Time").
    date_time_col = None
    for col in df.columns:
        if "Date" in col or "Time" in col:
            date_time_col = col
            break
    
    if date_time_col is None:
        raise ValueError(f"No suitable Date/Time column found in {csv_file}")
    
    # Rename the detected Date/Time column for consistency.
    df.rename(columns={date_time_col: 'Date/Time'}, inplace=True)
    
    # Fix the 24:00:00 issue and convert the 'Date/Time' column.
    df['Date/Time'] = df['Date/Time'].str.replace(' 24:00:00', ' 00:00:00', regex=False)
    df['Date/Time'] = pd.to_datetime('2023 ' + df['Date/Time'], format='%Y %m/%d %H:%M:%S', errors='coerce')
    df.dropna(subset=['Date/Time'], inplace=True)
    return df


def process_metering_data(csv_file, verbose=True):
    """
    Reads the metered data and converts the date/time column to a datetime object.
    """
    df = pd.read_csv(csv_file)
    
    # Print the columns to debug header issues if verbose is True
    if verbose:
        print(f"\nColumns in {csv_file}:")
        print(df.columns.tolist())
    
    # Determine the timestamp column (rename if necessary)
    timestamp_col = None
    if 'Timestamp' in df.columns:
        timestamp_col = 'Timestamp'
    else:
        for col in df.columns:
            if "Date" in col or "Time" in col:
                timestamp_col = col
                df.rename(columns={col: 'Timestamp'}, inplace=True)
                break
        if timestamp_col is None:
            raise ValueError(f"No suitable Timestamp/Date column found in {csv_file}")

    # Convert the Timestamp column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    return df

def plot_energy_data_inline(df, columns, start_date, end_date, num_xticks=10, verbose=True):
    """
    Plot energy data for specified columns with control on the number of x ticks.
    """
    if verbose:
        print("\nAvailable columns in the data:")
        print(df.columns.tolist())
        print("\nSample data:")
        print(df.head())
    
    # Sort the data by time and filter by the specified date range
    df = df.sort_values(by='Date/Time')
    filtered_df = df[(df['Date/Time'] >= start_date) & (df['Date/Time'] <= end_date)]
    
    # Plot each specified column
    for column in columns:
        if column not in df.columns:
            if verbose:
                print(f"Column '{column}' not found in the data. Skipping...")
            continue

        plt.figure(figsize=(14, 7))
        plt.plot(filtered_df['Date/Time'], filtered_df[column] / 3.6e6,  
                 label=column, linestyle='-', marker='o', alpha=0.8)
        
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(num_xticks))
        
        plt.xlabel('Time')
        plt.ylabel('Energy Use (kWh·m$^{-2}$·y$^{-1}$)')
        plt.title(f'Energy Data ({column}) from {start_date} to {end_date}')
        plt.legend()
        plt.grid(True, alpha=0.6)
        plt.show()

def profiling_subhourly(parent_folder, start_date, end_date, energy_columns, combined=True, alpha=0.7, area = 8894,
                           metered_csv=None, metered_column='Main Heating', num_xticks=10, verbose=True):
    """
    Process simulation data from subfolders and, if provided, overlay the metered profile.
    Debug output is controlled by the 'verbose' flag.
    """
    # Convert start and end dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    combined_data = []

    # Process simulated energy profiles from each subfolder
    for subfolder in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(full_path):
            energy_file = None
            # Look for eplusmtr.csv in the subfolder
            for file in os.listdir(full_path):
                if file == "eplusmtr.csv":
                    energy_file = os.path.join(full_path, file)
                    break

            if energy_file:
                if verbose:
                    print(f"Processing energy data: {energy_file}")
                try:
                    energy_df = process_energy_data(energy_file, verbose=verbose)
                    energy_df = energy_df[(energy_df['Date/Time'] >= start_date) &
                                          (energy_df['Date/Time'] <= end_date)]
                    energy_df = energy_df.sort_values(by='Date/Time')
                    energy_df['Simulation'] = os.path.basename(subfolder)
                    combined_data.append(energy_df)
                except Exception as e:
                    if verbose:
                        print(f"Error processing results in {subfolder}: {e}")
            else:
                if verbose:
                    print(f"No eplusmtr.csv found in {full_path}. Skipping...")

    # Process metered data if the CSV file exists
    metered_df = None
    if metered_csv and os.path.exists(metered_csv):
        try:
            metered_df = process_metering_data(metered_csv, verbose=verbose)
            metered_df = metered_df[(metered_df['Timestamp'] >= start_date) & 
                                    (metered_df['Timestamp'] <= end_date)]
            metered_df = metered_df.sort_values(by='Timestamp')
        except Exception as e:
            if verbose:
                print(f"Error processing metered data: {e}")
            metered_df = None
    elif metered_csv:
        if verbose:
            print(f"Metered CSV file {metered_csv} not found. Skipping metered profile.")
    
    if combined_data:
        combined_df = pd.concat(combined_data)
        
        if combined:
            # Create a green colormap for the simulation lines
            unique_simulations = combined_df['Simulation'].unique()
            n_sim = len(unique_simulations)
            cmap = plt.get_cmap('PRGn')
            color_map = {sim: cmap(i / max(1, n_sim - 1)) for i, sim in enumerate(unique_simulations)}
            
            # Plot each energy column combining all simulations
            for column in energy_columns:
                fig, ax = plt.subplots(figsize=(14, 7))
                for simulation_label, group_df in combined_df.groupby('Simulation'):
                    ax.plot(
                        group_df['Date/Time'],
                        group_df[column] / (area * 3.6e6),
                        label=f"{simulation_label} (simulation)",
                        linestyle='-',
                        linewidth=1,
                        alpha=alpha,
                        color=color_map[simulation_label]
                    )
                # Overlay the metered profile if available
                if metered_df is not None:
                    ax.plot(
                        metered_df['Timestamp'],
                        metered_df[metered_column] / area,
                        label='Metered',
                        linestyle='-',
                        linewidth=1,
                        color='r',
                        alpha=0.6
                    )
                
                # Add a right-handed y-axis for outdoor temperature if available
                if metered_df is not None and 'T_out' in metered_df.columns:
                    ax2 = ax.twinx()
                    ax2.plot(
                        metered_df['Timestamp'],
                        metered_df['T_out'],
                        label='Outdoor Temp',
                        linestyle=':',
                        linewidth=1,
                        color='tab:blue',
                        alpha=0.7
                    )
                    ax2.set_ylim(-10, 30)
                    ax2.set_ylabel('Outdoor Temperature (°C)')
                
                ax.set_xlim(start_date, end_date)
                ax.xaxis.set_major_locator(MaxNLocator(num_xticks))
                ax.set_xlabel('Time')
                ax.set_ylabel('EUI (kWh·m$^{-2}$·y$^{-1}$???)')
                ax.set_title(f'Metered and simulated energy profiling (as per meter reading and {column} from E+) ')
                
                # Combine legend handles from both axes
                handles, labels = ax.get_legend_handles_labels()
                if metered_df is not None and 'T_out' in metered_df.columns:
                    handles2, labels2 = ax2.get_legend_handles_labels()
                    handles += handles2
                    labels += labels2
                
                # Set legend in two columns
                ax.legend(handles, labels, ncol=2)
                ax.grid(True, alpha=0.6)
                plt.show()
        else:
            # Create a green colormap for the simulation lines
            unique_simulations = combined_df['Simulation'].unique()
            n_sim = len(unique_simulations)
            cmap = plt.get_cmap('Greens')
            color_map = {sim: cmap(i / max(1, n_sim - 1)) for i, sim in enumerate(unique_simulations)}
            
            # Plot each energy column separately for each simulation
            for column in energy_columns:
                for simulation_label, group_df in combined_df.groupby('Simulation'):
                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.plot(
                        group_df['Date/Time'],
                        group_df[column] / (area * 3.6e6),
                        label=f"{simulation_label} (simulation)",
                        linestyle='-',
                        alpha=alpha,
                        color=color_map[simulation_label]
                    )
                    # Overlay the metered profile if available
                    if metered_df is not None:
                        ax.plot(
                            metered_df['Timestamp'],
                            metered_df[metered_column] / (area * 3.6e6),
                            label='Metered',
                            linestyle='-',
                            linewidth=1,
                            color='black',
                            alpha=0.8
                        )
                    
                    # Add a twin axis for outdoor temperature if available
                    if metered_df is not None and 'T_out' in metered_df.columns:
                        ax2 = ax.twinx()
                        ax2.plot(
                            metered_df['Timestamp'],
                            metered_df['T_out'],
                            label='Outdoor Temp',
                            linestyle=':',
                            linewidth=1,
                            color='tab:blue'
                        )
                        ax2.set_ylim(-10, 30)
                        ax2.set_ylabel('Outdoor Temperature (°C)')
                    
                    ax.set_xlim(start_date, end_date)
                    ax.xaxis.set_major_locator(MaxNLocator(num_xticks))
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Energy Use (kWh·m$^{-2}$·y$^{-1}$)')
                    ax.set_title(f'{column} for {simulation_label}')
                    
                    # Combine legend handles
                    handles, labels = ax.get_legend_handles_labels()
                    if metered_df is not None and 'T_out' in metered_df.columns:
                        handles2, labels2 = ax2.get_legend_handles_labels()
                        handles += handles2
                        labels += labels2
                    ax.legend(handles, labels, ncol=2)
                    
                    ax.grid(True, alpha=0.6)
                    plt.show()
    else:
        if verbose:
            print("No data to plot.")

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
def breakdown_annual(parent_folders, metered_csv, start_date, end_date, area, legend_mapping, verbose=True):
    """
    Processes simulation data from multiple parent folders and metered data from a CSV file,
    computes the annual Energy Use Intensity (EUI) for electricity, heating, and cooling,
    and then displays a grouped bar chart comparing simulation results (averaged across runs
    in each folder) versus metered results.
    
    Simulation energy columns are assumed to be:
      - Electricity:Facility [J](Hourly)
      - DistrictHeatingWater:Facility [J](Hourly)
      - DistrictCooling:Facility [J](Hourly)
      
    Optionally, the simulation output may also include:
      - Heating:Electricity [J](Hourly)
      - Cooling:Electricity [J](Hourly)
    which will be used to adjust the resulting energy values.
    
    Metered data is assumed to have columns:
      - Main Electricity
      - Main Heating
      - Main Cooling
      
    Parameters:
      parent_folders: list of folder paths, where each folder contains subfolders with simulation files.
      metered_csv: path to the metered data CSV file.
      start_date: start date (inclusive) as a string/datetime.
      end_date: end date (exclusive) as a string/datetime.
      area: area value used for normalization.
      legend_mapping: dictionary mapping each parent folder's basename to a friendly label for the legend.
      verbose: if True, prints processing details.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    simulation_results = []
    sim_start = pd.to_datetime(start_date)
    sim_end = pd.to_datetime(end_date)
    
    # Process simulation files from each provided parent folder.
    for parent_folder in parent_folders:
        # Use the folder’s basename as the key for legend_mapping.
        folder_key = os.path.basename(parent_folder)
        friendly_name = legend_mapping.get(folder_key, folder_key)
        if verbose:
            print(f"Processing parent folder: {parent_folder} as '{friendly_name}'")
        # Loop over each subfolder within the parent folder.
        for subfolder in os.listdir(parent_folder):
            full_path = os.path.join(parent_folder, subfolder)
            if os.path.isdir(full_path):
                energy_file = None
                # Look for the simulation results file.
                for file in os.listdir(full_path):
                    if file == "eplusmtr.csv":
                        energy_file = os.path.join(full_path, file)
                        break
                if energy_file:
                    if verbose:
                        print(f"  Processing simulation file: {energy_file}")
                    try:
                        # Assume process_energy_data is defined elsewhere.
                        df = process_energy_data(energy_file, verbose=verbose)
                        # Filter data by the provided date range.
                        df = df[(df['Date/Time'] >= sim_start) & (df['Date/Time'] < sim_end)]
                        df = df.sort_values(by='Date/Time')
                        
                        # Compute annual totals in Joules from the core simulation outputs.
                        sim_electricity_total_J = df['Electricity:Facility [J](Hourly)'].sum()
                        sim_heating_total_J = df['DistrictHeatingWater:Facility [J](Hourly)'].sum()
                        sim_cooling_total_J = df['DistrictCooling:Facility [J](Hourly)'].sum()
                        
                        # Optionally include additional electricity contributions if the columns exist.
                        optional_heating_elec = (df['Heating:Electricity [J](Hourly)'].sum()
                                                 if 'Heating:Electricity [J](Hourly)' in df.columns
                                                 else 0)
                        optional_cooling_elec = (df['Cooling:Electricity [J](Hourly)'].sum()
                                                 if 'Cooling:Electricity [J](Hourly)' in df.columns
                                                 else 0)
                        
                        # Adjust the annual energy values:
                        # - Electricity is reduced by the electricity used for heating and cooling.
                        # - Heating and Cooling are increased by their respective electricity components.
                        sim_electricity = (sim_electricity_total_J - optional_heating_elec - optional_cooling_elec) / (area * 3.6e6)
                        sim_heating = (sim_heating_total_J + optional_heating_elec) / (area * 3.6e6)
                        sim_cooling = (sim_cooling_total_J + optional_cooling_elec) / (area * 3.6e6)
                        
                        simulation_results.append({
                            'ParentFolder': friendly_name,
                            'Simulation': subfolder,
                            'Electricity': sim_electricity,
                            'Heating': sim_heating,
                            'Cooling': sim_cooling,
                        })
                    except Exception as e:
                        if verbose:
                            print(f"    Error processing {subfolder} in {parent_folder}: {e}")
                else:
                    if verbose:
                        print(f"  No eplusmtr.csv found in {full_path}")
    
    # Process the metered data.
    metered_results = {}
    if metered_csv and os.path.exists(metered_csv):
        try:
            # Assume process_metering_data is defined elsewhere.
            met_df = process_metering_data(metered_csv, verbose=verbose)
            met_df = met_df[(met_df['Timestamp'] >= sim_start) & (met_df['Timestamp'] < sim_end)]
            met_df = met_df.sort_values(by='Timestamp')
            # Sum each metered energy column (assumed already in kWh) and normalize.
            met_electricity = met_df['Main Electricity'].sum() / area
            met_heating = met_df['Main Heating'].sum() / area
            met_cooling = met_df['Main Cooling'].sum() / area
            metered_results = {
                'Electricity': met_electricity,
                'Heating': met_heating,
                'Cooling': met_cooling
            }
        except Exception as e:
            if verbose:
                print(f"Error processing metered data: {e}")
    else:
        if verbose:
            print(f"Metered CSV file {metered_csv} not found. Skipping metered data.")
    
    # Create a DataFrame of the simulation results.
    sim_df = pd.DataFrame(simulation_results)
    if sim_df.empty:
        print("No simulation data found.")
        return
    
    # Group by the friendly parent folder label and compute the mean and standard deviation for each energy type.
    grouped = sim_df.groupby('ParentFolder').agg({
        'Electricity': ['mean', 'std'],
        'Heating': ['mean', 'std'],
        'Cooling': ['mean', 'std']
    }).reset_index()
    # Flatten the MultiIndex columns.
    grouped.columns = ['ParentFolder', 'Electricity_mean', 'Electricity_std',
                       'Heating_mean', 'Heating_std', 'Cooling_mean', 'Cooling_std']
    
    # Define energy types.
    energy_types = ['Electricity', 'Heating', 'Cooling']
    x = np.arange(len(energy_types))
    
    # Total number of bars in each energy-type group is one per simulation group plus one for metered data.
    n_groups = len(grouped)  # number of simulation groups
    n_bars = n_groups + 1
    bar_width = 0.8 / n_bars
    # Create evenly spaced offsets so that the bars for a given energy type are centered.
    offsets = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, n_bars)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot simulation bars for each parent folder.
    for i, (_, row) in enumerate(grouped.iterrows()):
        sim_values = [row[f"{et}_mean"] for et in energy_types]
        sim_errors = [row[f"{et}_std"] for et in energy_types]
        ax.bar(x + offsets[i],
               sim_values,
               width=bar_width,
               yerr=sim_errors,
               capsize=5,
               label=row['ParentFolder'])
    
    # Plot the metered data bar (using the last offset).
    metered_values = [metered_results.get(et, 0) for et in energy_types]
    ax.bar(x + offsets[-1],
           metered_values,
           width=bar_width,
           label='Metered')
    
    ax.set_ylabel('EUI (kWh·m$^{-2}$·y$^{-1}$)')
    ax.set_title('Annual EUI by Group')
    ax.set_xticks(x)
    ax.set_xticklabels(energy_types)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    if verbose:
        print("\nSimulation results (kWh·m⁻²·y⁻¹):")
        print(sim_df)
def breakdown_monthly(parent_folders, metered_csv, start_date, end_date, area, legend_mapping, verbose=True):
    """
    Processes simulation data from multiple parent folders and metered data from a CSV file,
    computes the monthly aggregated Energy Use Intensity (EUI) for electricity, heating, and cooling,
    and then displays grouped bar charts comparing simulation (mean ± 1 standard deviation)
    versus metered results for each month.
    
    Simulation energy columns used here:
      - Electricity:Facility [J](Hourly)
      - DistrictHeatingWater:Facility [J](Hourly)
      - DistrictCooling:Facility [J](Hourly)
      
    Metered data is assumed to have columns:
      - Main Electricity
      - Main Heating
      - Main Cooling
      
    Conversion:
      - Simulation data in Joules is converted to kWh and normalized by area.
      - Metered data (assumed in kWh) is normalized by area.
      
    The resulting unit is kWh·m⁻²·month⁻¹.
    
    Parameters:
      parent_folders: list of paths to parent folders, each containing simulation subfolders.
      metered_csv: path to the metered data CSV file.
      start_date: start date (inclusive) as a string or datetime.
      end_date: end date (exclusive) as a string or datetime.
      area: normalization area value.
      legend_mapping: dictionary mapping each parent folder's basename to a friendly label.
      verbose: if True, prints debugging information.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    simulation_data = []   # To hold monthly data for each simulation run along with group labels
    sim_start = pd.to_datetime(start_date)
    sim_end = pd.to_datetime(end_date)
    
    # Process each parent folder.
    for parent_folder in parent_folders:
        # Use the basename of the folder as key to look up a friendly name.
        folder_key = os.path.basename(parent_folder)
        group_label = legend_mapping.get(folder_key, folder_key)
        if verbose:
            print(f"Processing parent folder: {parent_folder} as '{group_label}'")
        for subfolder in os.listdir(parent_folder):
            full_path = os.path.join(parent_folder, subfolder)
            if os.path.isdir(full_path):
                energy_file = None
                for file in os.listdir(full_path):
                    if file == "eplusmtr.csv":
                        energy_file = os.path.join(full_path, file)
                        break
                if energy_file:
                    if verbose:
                        print(f"  Processing simulation file: {energy_file}")
                    try:
                        df = process_energy_data(energy_file, verbose=verbose)
                        # Filter the data within the date range.
                        df = df[(df['Date/Time'] >= sim_start) & (df['Date/Time'] < sim_end)]
                        df = df.sort_values(by='Date/Time')

                        # Build the aggregation dictionary, adding optional columns if present.
                        agg_dict = {
                            'Electricity:Facility [J](Hourly)': 'sum',
                            'DistrictHeatingWater:Facility [J](Hourly)': 'sum',
                            'DistrictCooling:Facility [J](Hourly)': 'sum',
                        }
                        if 'Heating:Electricity [J](Hourly)' in df.columns:
                            agg_dict['Heating:Electricity [J](Hourly)'] = 'sum'
                        if 'Cooling:Electricity [J](Hourly)' in df.columns:
                            agg_dict['Cooling:Electricity [J](Hourly)'] = 'sum'

                        # Group by month using the built dictionary.
                        monthly_df = df.groupby(pd.Grouper(key='Date/Time', freq='M')).agg(agg_dict)

                        # Convert aggregated Joules to kWh per m² per month,
                        # using .get() to supply 0 if optional columns are missing.
                        monthly_df['Electricity'] = (
                            monthly_df['Electricity:Facility [J](Hourly)'] -
                            monthly_df.get('Heating:Electricity [J](Hourly)', 0) -
                            monthly_df.get('Cooling:Electricity [J](Hourly)', 0)
                        ) / (area * 3.6e6)

                        monthly_df['Heating'] = (
                            monthly_df['DistrictHeatingWater:Facility [J](Hourly)'] +
                            monthly_df.get('Heating:Electricity [J](Hourly)', 0)
                        ) / (area * 3.6e6)

                        monthly_df['Cooling'] = (
                            monthly_df['DistrictCooling:Facility [J](Hourly)'] +
                            monthly_df.get('Cooling:Electricity [J](Hourly)', 0)
                        ) / (area * 3.6e6)

                        # Keep only the calculated columns.
                        monthly_df = monthly_df[['Electricity', 'Heating', 'Cooling']]
                        # Tag this simulation run with the group label and simulation id.
                        monthly_df['Group'] = group_label
                        monthly_df['Simulation'] = subfolder
                        # Reset the index and rename the grouping column to 'Month'.
                        monthly_df = monthly_df.reset_index().rename(columns={'Date/Time': 'Month'})
                        simulation_data.append(monthly_df)
                    except Exception as e:
                        if verbose:
                            print(f"    Error processing {subfolder} in {parent_folder}: {e}")
                else:
                    if verbose:
                        print(f"  No eplusmtr.csv found in {full_path}")
    
    # Combine simulation data and compute monthly mean and std for each group.
    if simulation_data:
        sim_all = pd.concat(simulation_data, ignore_index=True)
        sim_summary = sim_all.groupby(['Month', 'Group']).agg({
            'Electricity': ['mean', 'std'],
            'Heating': ['mean', 'std'],
            'Cooling': ['mean', 'std']
        }).reset_index()
        sim_summary.columns = ['Month', 'Group',
                               'Electricity_mean', 'Electricity_std',
                               'Heating_mean', 'Heating_std',
                               'Cooling_mean', 'Cooling_std']
    else:
        sim_summary = pd.DataFrame()
    
    # Process metered data.
    metered_summary = None
    if metered_csv and os.path.exists(metered_csv):
        try:
            metered_df = process_metering_data(metered_csv, verbose=verbose)
            metered_df = metered_df[(metered_df['Timestamp'] >= sim_start) & (metered_df['Timestamp'] < sim_end)]
            metered_df = metered_df.sort_values(by='Timestamp')
            
            monthly_metered = metered_df.groupby(pd.Grouper(key='Timestamp', freq='M')).agg({
                'Main Electricity': 'sum',
                'Main Heating': 'sum',
                'Main Cooling': 'sum'
            })
            monthly_metered['Electricity'] = monthly_metered['Main Electricity'] / area
            monthly_metered['Heating'] = monthly_metered['Main Heating'] / area
            monthly_metered['Cooling'] = monthly_metered['Main Cooling'] / area
            monthly_metered = monthly_metered[['Electricity', 'Heating', 'Cooling']]
            metered_summary = monthly_metered.reset_index().rename(columns={'Timestamp': 'Month'})
        except Exception as e:
            if verbose:
                print(f"Error processing metered data: {e}")
    else:
        if verbose:
            print(f"Metered CSV file {metered_csv} not found. Skipping metered data.")
    
    # Plot grouped bar charts for each energy type.
    energy_types = ['Electricity', 'Heating', 'Cooling']
    for energy in energy_types:
        if sim_summary.empty:
            print("No simulation data available to plot.")
            continue
        
        # Pivot the simulation summary so that each group is a separate column.
        pivot_means = sim_summary.pivot(index='Month', columns='Group', values=f'{energy}_mean')
        pivot_std = sim_summary.pivot(index='Month', columns='Group', values=f'{energy}_std')
        
        # Get sorted months and list of simulation groups.
        months = pivot_means.index.sort_values()
        sim_groups = list(pivot_means.columns)
        n_sim_groups = len(sim_groups)
        
        # Determine total number of bars: simulation groups plus metered (if available).
        total_bars = n_sim_groups + 1 if metered_summary is not None and not metered_summary.empty else n_sim_groups
        bar_width = 0.8 / total_bars
        
        # Compute offsets so that the bars for each month are centered.
        offsets = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, total_bars)
        x = np.arange(len(months))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot simulation bars for each group.
        for i, group in enumerate(sim_groups):
            means = pivot_means[group].reindex(months)
            stds = pivot_std[group].reindex(months)
            ax.bar(x + offsets[i],
                   means,
                   width=bar_width,
                   yerr=stds,
                   capsize=5,
                   label=group)
        
        # Plot the metered data as an extra set of bars.
        if metered_summary is not None and not metered_summary.empty:
            metered_months = metered_summary.set_index('Month').reindex(months)
            metered_vals = metered_months[energy].fillna(0)
            ax.bar(x + offsets[-1],
                   metered_vals,
                   width=bar_width,
                   label='Metered')
        
        # Format the x-axis with month labels.
        month_labels = [m.strftime('%b %Y') for m in months]
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels, rotation=45, ha='right')
        
        ax.set_xlabel('Month')
        ax.set_ylabel(f'EUI ({energy}) (kWh·m$^{{-2}}$·month$^{{-1}}$)')
        ax.set_title(f'Monthly EUI ({energy}) by Group')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print("\nSimulation monthly summary:")
        print(sim_summary)
        if metered_summary is not None:
            print("\nMetered monthly summary:")
            print(metered_summary)

def display_monthly_energy_use_stacked(parent_folder, metered_csv, start_date, end_date, area, verbose=True):
    """
    For each simulation file (found as a subfolder with an "eplusmtr.csv" file), the function:
      - Reads the hourly simulation data.
      - Aggregates monthly energy use for:
          • Electricity: using the column "Electricity:Facility [J](Hourly)"
          • Heating: using the column "DistrictHeatingWater:Facility [J](Hourly)"
          • Cooling (if available): using "DistrictCooling:Facility [J](Hourly)"
      - Converts from Joules to kWh and normalizes by area.
    Metered data is processed similarly using:
          • "Main Electricity", "Main Heating", and (if available) "Main Cooling"
      (Here metered values are assumed to be in kWh and are normalized by area.)
    
    Then, for each month a stacked bar chart is produced:
      - Each simulation file is plotted as a separate column (bar) in the month.
        Their stacked segments (from bottom to top: Electricity, Heating, Cooling) are plotted
        with base colors (grey, #de0a26, #00619bff) and with alpha starting at 0.8 for the first file,
        then decreasing by 0.1 for subsequent files.
      - A final column is added for the metered data, using fixed (darker) colors with no transparency.
    """
    # --- Process simulation files ---
    sim_data_frames = []
    for subfolder in os.listdir(parent_folder):
        full_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(full_path):
            energy_file = None
            for file in os.listdir(full_path):
                if file == "eplusmtr.csv":
                    energy_file = os.path.join(full_path, file)
                    break
            if energy_file:
                if verbose:
                    print(f"Processing simulation file: {energy_file}")
                try:
                    df = process_energy_data(energy_file, verbose=verbose)
                    df = df[(df['Date/Time'] >= pd.to_datetime(start_date)) & 
                            (df['Date/Time'] < pd.to_datetime(end_date))]
                    df = df.sort_values(by='Date/Time')
                    # Convert energy from Joules to kWh/m²
                    df['Electricity_kWh'] = df['Electricity:Facility [J](Hourly)'] / (area * 3.6e6)
                    df['Heating_kWh'] = df['DistrictHeatingWater:Facility [J](Hourly)'] / (area * 3.6e6)
                    # Cooling is optional
                    if 'DistrictCooling:Facility [J](Hourly)' in df.columns:
                        df['Cooling_kWh'] = df['DistrictCooling:Facility [J](Hourly)'] / (area * 3.6e6)
                    else:
                        df['Cooling_kWh'] = 0
                    # Group by month
                    monthly = df.groupby(pd.Grouper(key='Date/Time', freq='M')).agg({
                        'Electricity_kWh': 'sum',
                        'Heating_kWh': 'sum',
                        'Cooling_kWh': 'sum'
                    }).reset_index().rename(columns={'Date/Time': 'Month'})
                    monthly['Simulation'] = subfolder
                    sim_data_frames.append(monthly)
                except Exception as e:
                    if verbose:
                        print(f"Error processing {subfolder}: {e}")
            else:
                if verbose:
                    print(f"No eplusmtr.csv found in {full_path}")
    if not sim_data_frames:
        print("No simulation data found.")
        return
    sim_df = pd.concat(sim_data_frames, ignore_index=True)
    
    # --- Process metered data ---
    try:
        met_df = process_metering_data(metered_csv, verbose=verbose)
        met_df = met_df[(met_df['Timestamp'] >= pd.to_datetime(start_date)) & 
                        (met_df['Timestamp'] < pd.to_datetime(end_date))]
        met_df = met_df.sort_values(by='Timestamp')
        # For metered data, assume values are in kWh, so simply normalize by area
        met_df['Electricity_kWh'] = met_df['Main Electricity'] / area
        met_df['Heating_kWh'] = met_df['Main Heating'] / area
        if 'Main Cooling' in met_df.columns:
            met_df['Cooling_kWh'] = met_df['Main Cooling'] / area
        else:
            met_df['Cooling_kWh'] = 0
        met_monthly = met_df.groupby(pd.Grouper(key='Timestamp', freq='M')).agg({
            'Electricity_kWh': 'sum',
            'Heating_kWh': 'sum',
            'Cooling_kWh': 'sum'
        }).reset_index().rename(columns={'Timestamp': 'Month'})
    except Exception as e:
        if verbose:
            print(f"Error processing metered data: {e}")
        met_monthly = pd.DataFrame()
    
    # --- Determine unique months and simulation file order ---
    months = pd.to_datetime(sorted(set(sim_df['Month']).union(
        set(met_monthly['Month']) if not met_monthly.empty else set(sim_df['Month'])
    )))
    sim_files = sorted(sim_df['Simulation'].unique())
    # Assign alpha values for simulation files: first file gets 0.8, then 0.7, etc.
    alphas = {sim: max(0, 0.8 - 0.1*i) for i, sim in enumerate(sim_files)}
    
    # Define base colors (for both simulation and metered):
    base_colors = {
        'Electricity': 'grey',       # bottom segment
        'Heating': "#de0a26",         # middle segment
        'Cooling': "#00619bff"        # top segment
    }
    # For metered, use these colors with full opacity (they are the darkest)
    met_colors = {k: v for k, v in base_colors.items()}
    
    # --- Plot for each month ---
    for month in months:
        # Filter simulation data for the current month
        month_sim = sim_df[sim_df['Month'] == month]
        # For each simulation file, extract the values (or assume zero if missing)
        sim_electricity = []
        sim_heating = []
        sim_cooling = []
        for sim in sim_files:
            row = month_sim[month_sim['Simulation'] == sim]
            if not row.empty:
                elec = row.iloc[0]['Electricity_kWh']
                heat = row.iloc[0]['Heating_kWh']
                cool = row.iloc[0]['Cooling_kWh']
            else:
                elec, heat, cool = 0, 0, 0
            sim_electricity.append(elec)
            sim_heating.append(heat)
            sim_cooling.append(cool)
        
        # Get metered data for the month (if available)
        if not met_monthly.empty:
            met_row = met_monthly[met_monthly['Month'] == month]
            if not met_row.empty:
                met_electricity = met_row.iloc[0]['Electricity_kWh']
                met_heating = met_row.iloc[0]['Heating_kWh']
                met_cooling = met_row.iloc[0]['Cooling_kWh']
            else:
                met_electricity, met_heating, met_cooling = 0, 0, 0
        else:
            met_electricity, met_heating, met_cooling = 0, 0, 0
        
        # Create one bar per simulation file and one for metered
        n_bars = len(sim_files) + 1  # last bar is metered
        x_positions = np.arange(n_bars)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot simulation bars (each with its own alpha)
        for i, sim in enumerate(sim_files):
            bottom = 0
            alpha_val = alphas[sim]
            # Electricity segment
            elec_val = sim_electricity[i]
            ax.bar(x_positions[i], elec_val, width=0.8, bottom=bottom,
                   color=base_colors['Electricity'], alpha=alpha_val)
            bottom += elec_val
            # Heating segment
            heat_val = sim_heating[i]
            ax.bar(x_positions[i], heat_val, width=0.8, bottom=bottom,
                   color=base_colors['Heating'], alpha=alpha_val)
            bottom += heat_val
            # Cooling segment
            cool_val = sim_cooling[i]
            ax.bar(x_positions[i], cool_val, width=0.8, bottom=bottom,
                   color=base_colors['Cooling'], alpha=alpha_val)
        
        # Plot metered bar (using fixed, dark colors, full opacity)
        bottom = 0
        ax.bar(x_positions[-1], met_electricity, width=0.8, bottom=bottom,
               color=met_colors['Electricity'], alpha=1.0)
        bottom += met_electricity
        ax.bar(x_positions[-1], met_heating, width=0.8, bottom=bottom,
               color=met_colors['Heating'], alpha=1.0)
        bottom += met_heating
        ax.bar(x_positions[-1], met_cooling, width=0.8, bottom=bottom,
               color=met_colors['Cooling'], alpha=1.0)
        
        # Label x-axis: simulation file names and "Metered" as the last label
        x_labels = sim_files + ["Metered"]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylabel("Energy Use (kWh·m⁻²·month⁻¹)")
        ax.set_title(f"Stacked Energy Use for {month.strftime('%b %Y')}")
        ax.grid(True, axis='y', alpha=0.6)
        plt.tight_layout()
        plt.show()

def aggregate_annual_energy(df, area, start_date, end_date, verbose=False):
    """
    Aggregates annual energy data from a DataFrame, taking into account optional columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with energy data.
        area (float): Total area used to normalize the energy values.
        start_date (str or pd.Timestamp): Start date for filtering the data.
        end_date (str or pd.Timestamp): End date (exclusive) for filtering the data.
        verbose (bool): If True, prints additional processing messages.
        
    Returns:
        dict: Annual aggregated values in kWh/m² for Electricity, Heating, and Cooling.
    """
    # Filter rows within the desired date range and sort by date.
    df = df[(df['Date/Time'] >= pd.to_datetime(start_date)) &
            (df['Date/Time'] < pd.to_datetime(end_date))]
    df = df.sort_values(by='Date/Time')
    
    # Calculate the total Joules for the mandatory columns.
    total_elec_fac = df['Electricity:Facility [J](Hourly)'].sum()
    total_heat_water = df['DistrictHeatingWater:Facility [J](Hourly)'].sum()
    # For District Cooling, use 0 if the column doesn't exist.
    total_cool_fac = df['DistrictCooling:Facility [J](Hourly)'].sum() if 'DistrictCooling:Facility [J](Hourly)' in df.columns else 0
    
    # For optional columns, sum if present, otherwise default to 0.
    total_heat_elec = df['Heating:Electricity [J](Hourly)'].sum() if 'Heating:Electricity [J](Hourly)' in df.columns else 0
    total_cool_elec = df['Cooling:Electricity [J](Hourly)'].sum() if 'Cooling:Electricity [J](Hourly)' in df.columns else 0
    
    # Compute derived end uses.
    # Electricity is net facility consumption minus the dedicated heating and cooling electricity.
    electricity_kWh_m2 = (total_elec_fac - total_heat_elec - total_cool_elec) / (area * 3.6e6)
    
    # Heating includes district heating plus heating electricity (if present).
    heating_kWh_m2 = (total_heat_water + total_heat_elec) / (area * 3.6e6)
    
    # Cooling includes district cooling plus cooling electricity (if present).
    cooling_kWh_m2 = (total_cool_fac + total_cool_elec) / (area * 3.6e6)
    
    if verbose:
        print("Annual aggregation complete:")
        print(f"  Electricity: {electricity_kWh_m2:.3f} kWh/m²/year")
        print(f"  Heating:     {heating_kWh_m2:.3f} kWh/m²/year")
        print(f"  Cooling:     {cooling_kWh_m2:.3f} kWh/m²/year")
    
    return {
        'Electricity': electricity_kWh_m2,
        'Heating': heating_kWh_m2,
        'Cooling': cooling_kWh_m2,
    }

def overview_annual(parent_folders, metered_csv, start_date, end_date, area, legend_mapping, verbose=True):
    """
    Processes simulation data from multiple parent folders (each containing subfolders with 'eplusmtr.csv')
    and metered data from a CSV file. For each simulation file, it aggregates the hourly energy use 
    over the entire year (from start_date to end_date), converts the energy from Joules to kWh/m² (for 
    simulations) or normalizes metered kWh by area, and then produces a stacked bar chart.

    For the simulation data, the energy is aggregated per parent folder (i.e. group) by averaging across
    its simulation runs. Each bar (one per group) is plotted with three stacked segments:
      Bottom: Electricity (from "Electricity:Facility [J](Hourly)")
      Middle: Heating (from "DistrictHeatingWater:Facility [J](Hourly)")
      Top: Cooling (if available, from "DistrictCooling:Facility [J](Hourly)"; otherwise 0)

    The simulation bars use a constant transparency (alpha=0.8). A final bar is added for the metered data 
    (using dark, fixed colors: grey for Electricity, "#de0a26" for Heating, and "#00619bff" for Cooling).

    The resulting unit is kWh·m⁻²·year⁻¹.

    Parameters:
      parent_folders : list
          A list of folder paths. Each folder represents a simulation group and contains subfolders
          with simulation output files.
      metered_csv : str
          Path to the metered data CSV file.
      start_date : str or datetime
          Start date (inclusive).
      end_date : str or datetime
          End date (exclusive).
      area : float
          Area value used to normalize the energy use.
      legend_mapping : dict
          Dictionary mapping each parent folder’s basename to a user‐friendly name.
      verbose : bool, optional
          If True, prints additional output for debugging.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Dictionary to hold simulation results for each group.
    group_results = {}

    # Loop over each parent folder.
    for parent_folder in parent_folders:
        # Use the basename as the key for lookup in the legend mapping.
        folder_key = os.path.basename(parent_folder)
        group_label = legend_mapping.get(folder_key, folder_key)
        if verbose:
            print(f"Processing parent folder: {parent_folder} as '{group_label}'")
        # Iterate through subfolders (each expected to contain an "eplusmtr.csv" file)
        for subfolder in os.listdir(parent_folder):
            full_path = os.path.join(parent_folder, subfolder)
            if os.path.isdir(full_path):
                energy_file = None
                for file in os.listdir(full_path):
                    if file == "eplusmtr.csv":
                        energy_file = os.path.join(full_path, file)
                        break
                if energy_file:
                    if energy_file:
                        if verbose:
                            print(f"Processing simulation file: {energy_file}")
                        try:
                            df = process_energy_data(energy_file, verbose=verbose)
                            annual_results = aggregate_annual_energy(df, area, start_date, end_date, verbose=verbose)
                            
                            # Append the annual results to your grouped results
                            if group_label not in group_results:
                                group_results[group_label] = []
                            group_results[group_label].append(annual_results)
                        except Exception as e:
                            if verbose:
                                print(f"Error processing {subfolder} in {parent_folder}: {e}")
                    else:
                        if verbose:
                            print(f"No eplusmtr.csv found in {full_path}")

    if not group_results:
        print("No simulation data found.")
        return

    # Aggregate (average) the simulation results for each group.
    aggregated_results = []
    for group, results in group_results.items():
        df_group = pd.DataFrame(results)
        avg_electricity = df_group['Electricity'].mean()
        avg_heating = df_group['Heating'].mean()
        avg_cooling = df_group['Cooling'].mean()
        aggregated_results.append({
            'Group': group,
            'Electricity': avg_electricity,
            'Heating': avg_heating,
            'Cooling': avg_cooling
        })

    sim_df = pd.DataFrame(aggregated_results)
    sim_groups = sim_df['Group'].tolist()

    # Process metered data.
    try:
        met_df = process_metering_data(metered_csv, verbose=verbose)
        met_df = met_df[(met_df['Timestamp'] >= pd.to_datetime(start_date)) &
                        (met_df['Timestamp'] < pd.to_datetime(end_date))]
        met_df = met_df.sort_values(by='Timestamp')
        # Assume metered values are in kWh; normalize by area.
        met_electricity = met_df['Main Electricity'].sum() / area
        met_heating = met_df['Main Heating'].sum() / area
        if 'Main Cooling' in met_df.columns:
            met_cooling = met_df['Main Cooling'].sum() / area
        else:
            met_cooling = 0
        metered_results = {
            'Electricity': met_electricity,
            'Heating': met_heating,
            'Cooling': met_cooling
        }
    except Exception as e:
        if verbose:
            print(f"Error processing metered data: {e}")
        metered_results = {'Electricity': 0, 'Heating': 0, 'Cooling': 0}

    # Define base colors for the stacked segments.
    base_colors = {
        'Electricity': 'grey',       # bottom segment
        'Heating': "#de0a26",         # middle segment
        'Cooling': "#00619bff"        # top segment
    }
    # Metered bar uses full opacity.
    met_colors = {k: v for k, v in base_colors.items()}

    # Prepare x positions: one bar per simulation group plus one for metered data.
    n_bars = len(sim_groups) + 1
    x_positions = np.arange(n_bars)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked bars for each simulation group.
    for i, group in enumerate(sim_groups):
        row = sim_df[sim_df['Group'] == group]
        if not row.empty:
            elec_val = row.iloc[0]['Electricity']
            heat_val = row.iloc[0]['Heating']
            cool_val = row.iloc[0]['Cooling']
        else:
            elec_val, heat_val, cool_val = 0, 0, 0

        bottom = 0
        alpha_val = 0.8  # constant alpha for aggregated group bars
        # Bottom segment: Electricity.
        ax.bar(x_positions[i], elec_val, width=0.8, bottom=bottom,
               color=base_colors['Electricity'], alpha=alpha_val)
        bottom += elec_val
        # Middle segment: Heating.
        ax.bar(x_positions[i], heat_val, width=0.8, bottom=bottom,
               color=base_colors['Heating'], alpha=alpha_val)
        bottom += heat_val
        # Top segment: Cooling.
        ax.bar(x_positions[i], cool_val, width=0.8, bottom=bottom,
               color=base_colors['Cooling'], alpha=alpha_val)

    # Plot the metered data in the last position with full opacity.
    bottom = 0
    ax.bar(x_positions[-1], metered_results['Electricity'], width=0.8, bottom=bottom,
           color=met_colors['Electricity'], alpha=1.0)
    bottom += metered_results['Electricity']
    ax.bar(x_positions[-1], metered_results['Heating'], width=0.8, bottom=bottom,
           color=met_colors['Heating'], alpha=1.0)
    bottom += metered_results['Heating']
    ax.bar(x_positions[-1], metered_results['Cooling'], width=0.8, bottom=bottom,
           color=met_colors['Cooling'], alpha=1.0)

    # Create a legend for the energy types.
    legend_elements = [
        Patch(facecolor=base_colors['Electricity'], label='Electricity'),
        Patch(facecolor=base_colors['Heating'], label='Heating'),
        Patch(facecolor=base_colors['Cooling'], label='Cooling')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Label the x-axis using the simulation group names and add a "Metered" bar.
    x_labels = sim_groups + ["Metered"]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Energy Use (kWh·m⁻²·year⁻¹)")
    ax.set_title("Annual EUI")
    ax.grid(True, axis='y', alpha=0.6)
    plt.tight_layout()
    plt.show()

    if verbose:
        print("\nAggregated Simulation Results (kWh·m⁻²·year⁻¹):")
        print(sim_df)
        print("\nMetered Aggregated Results (kWh·m⁻²·year⁻¹):")
        print(metered_results)
