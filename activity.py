import numpy as np
import pandas as pd
import ruptures as rpt
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib as mpl


# --- Classify the data in activity/rest ---

def classify_signal(list_of_signals, list_of_signals_time, list_of_bkps_pred, threshold = 1.5):

    df = list()   

    for k_hour, (hour, hour_time, bkps_pred) in enumerate(
        zip(list_of_signals, list_of_signals_time, list_of_bkps_pred), start=1
        ):

        for start, end in rpt.utils.pairwise([0] + bkps_pred):
            start_time, end_time = hour_time[start], hour_time[min(end, hour_time.shape[0]-1)]
            SD_p = np.std(hour[start:end])

            res = dict(
                    hour= k_hour,
                    start_time=start_time,
                    end_time=end_time,
                    duration=(end_time - start_time),
                    SD_p=SD_p, 
                )
            df.append(res)

    df = pd.DataFrame(df)
    df["day_or_night"] = np.where((((df.hour -1 ) ) % 24) < 12, "night", "day") 
    df["day_or_night_number"] = ((((df.hour - 1) - 12) // 24) + 1)
    df["activity"] = np.where(df.SD_p > threshold, True, False)

    return df



# --- visualization ---

def distribution_SD(df, threshold = 1.5):
    #changing the value of threshold will actualize the value at which a segment is classified as 'activity' or 'rest'
    df["activity"] = np.where(df.SD_p > threshold, True, False)

    ddf = df.loc[df['day_or_night'] == 'day']
    ndf = df.loc[df['day_or_night'] == 'night']
   

    plt.axvline(threshold , color='0.8', linestyle='--')

    plt.hist(ndf['SD_p'], bins=np.arange(0,15,0.1), alpha = 0.6, label='$\sigma_p$ during night', color = '#2B2C5A')
    plt.hist(ddf['SD_p'], bins=np.arange(0,15,0.1), alpha = 0.6, label='$\sigma_p$ during day', color = '#FADB33')
    plt.xlabel('$\sigma_p$')
    plt.ylabel('occurence')

    plt.legend(loc='upper right')
    plt.show()

    return df


def timeline(df, binary = True):
    unique_rows = df.drop_duplicates(subset=['day_or_night', 'day_or_night_number'])

    last_row_day_or_night_number = df.iloc[df.index[-1] - 1]['day_or_night_number']
    last_row_day_or_night = df.iloc[df.index[-1] - 1]['day_or_night']
    if last_row_day_or_night == 'day':
        num_subplots = last_row_day_or_night_number * 2
    else :
        num_subplots = last_row_day_or_night_number * 2 + 1

    fig, axs = plt.subplots(nrows=num_subplots, ncols=1, figsize=(40,8), layout='constrained')
    hour_12_signal = []
    i=0

    if binary == True:
        for index, row in unique_rows.iterrows():
            hour_12_signal.append(df.groupby(['day_or_night', 'day_or_night_number']).get_group((row['day_or_night'],row['day_or_night_number'])))

            ax = axs[i]
            ax.set_title(row['day_or_night'] + ' ' + str(row['day_or_night_number']))
            ax.set_xlim(hour_12_signal[i].iloc[0]['start_time'], hour_12_signal[i].iloc[0]['start_time']+ 3600*12)
            hour_12_xticks = [hour_12_signal[i].iloc[0]['start_time']+3600*k for k in range(13)]
            ax.set_xticks(hour_12_xticks, np.arange(13))
            ax.set_yticks([])
            
            for indexax, rowax in hour_12_signal[i].iterrows():
                # Set the color based on activity
                color = mpl.colors.to_hex((0.9,0.2,0.1)) if rowax['activity'] else mpl.colors.to_hex((0.68,0.83,0.94))

                ax.plot([rowax['start_time'], rowax['end_time']], [0, 0], color=color, lw=100)
            i+=1

    elif binary == False:
        for index, row in unique_rows.iterrows():
            hour_12_signal.append(df.groupby(['day_or_night', 'day_or_night_number']).get_group((row['day_or_night'],row['day_or_night_number'])))

            ax = axs[i]
            ax.set_title(row['day_or_night'] + ' ' + str(row['day_or_night_number']))
            ax.set_xlim(hour_12_signal[i].iloc[0]['start_time'], hour_12_signal[i].iloc[0]['start_time']+ 3600*12)
            hour_12_xticks = [hour_12_signal[i].iloc[0]['start_time']+3600*k for k in range(13)]
            ax.set_xticks(hour_12_xticks, np.arange(13))
            ax.set_yticks([])
            
            c1=np.array(mpl.colors.to_rgb((0.68,0.83,0.94))) #blue
            c2=np.array(mpl.colors.to_rgb((1,0,0))) #red
            c3=np.array(mpl.colors.to_rgb((0,0,0))) #black
            for indexax, rowax in hour_12_signal[i].iterrows():          

                v = rowax['SD_p']/10 
                if v <= 1:
                    color = mpl.colors.to_hex((1-v)*c1 + v*c2)
                elif v <= 2:
                    color = mpl.colors.to_hex((1-(v-1))*c2 + (v-1)*c3)
                else :
                    print(v, ' : outlier')
                    color = mpl.colors.to_hex((0,0,0))

                ax.plot([rowax['start_time'], rowax['end_time']], [0, 0], color=color, lw=100)
            i+=1
    else :
        raise ValueError("binary must be True or False")
    


# --- features calculated on the previously created dataframe ---

def activity_rest_features(df):
    
    # --- Total duration (sec) ---
    sum_duration_activity_day = df.loc[(df['activity'] == True) & (df['day_or_night'] == 'day'), 'duration'].sum()
    sum_duration_rest_day = df.loc[(df['activity'] == False) & (df['day_or_night'] == 'day'), 'duration'].sum()
    sum_duration_activity_night = df.loc[(df['activity'] == True) & (df['day_or_night'] == 'night'), 'duration'].sum()
    sum_duration_rest_night = df.loc[(df['activity'] == False) & (df['day_or_night'] == 'night'), 'duration'].sum()

    # --- Average duration during 1 day / 1 night (sec) ---
    unique_hours = df.drop_duplicates(subset=['hour', 'day_or_night'])

    n_hours_day = unique_hours.loc[unique_hours['day_or_night'] == 'day'].shape[0]
    n_hours_night = unique_hours.loc[unique_hours['day_or_night'] == 'night'].shape[0]

    avg_sum_1d_duration_activity_day = (sum_duration_activity_day*12)/n_hours_day
    avg_sum_1d_duration_rest_day = (sum_duration_rest_day*12)/n_hours_day
    avg_sum_1n_duration_activity_night = (sum_duration_activity_night*12)/n_hours_night
    avg_sum_1n_duration_rest_night = (sum_duration_rest_night*12)/n_hours_night

    # --- Average duration of one segment (sec) ---
    avg_segment_duration_activity_day = df.loc[(df['activity'] == True) & (df['day_or_night'] == 'day'), 'duration'].mean()
    avg_segment_duration_rest_day = df.loc[(df['activity'] == False) & (df['day_or_night'] == 'day'), 'duration'].mean()
    avg_segment_duration_activity_night = df.loc[(df['activity'] == True) & (df['day_or_night'] == 'night'), 'duration'].mean()
    avg_segment_duration_rest_night = df.loc[(df['activity'] == False) & (df['day_or_night'] == 'night'), 'duration'].mean()

    

    sum_duration_list = [sum_duration_activity_day, sum_duration_rest_day, sum_duration_activity_night, sum_duration_rest_night]
    avg_sum_1_list = [avg_sum_1d_duration_activity_day, avg_sum_1d_duration_rest_day, avg_sum_1n_duration_activity_night, avg_sum_1n_duration_rest_night]
    avg_sum_1_list_pourcentage = [np.round(x/(12*36), 1) for x in avg_sum_1_list]
    avg_segment_duration_list = [avg_segment_duration_activity_day, avg_segment_duration_rest_day, avg_segment_duration_activity_night, avg_segment_duration_rest_night]

    print('duration activity day ' , sum_duration_list[0], ' - duration rest day ' , sum_duration_list[1], ' - duration activity night ' , sum_duration_list[2], ' - duration rest night ' , sum_duration_list[3], ' - duration activity day ' , sum_duration_list[0])
    print('\n avg duration activty day ', avg_sum_1_list[0], ' (',avg_sum_1_list_pourcentage[0], '%)', ' - avg duration rest day ', avg_sum_1_list[1], ' (',avg_sum_1_list_pourcentage[1], '%)',  ' - avg duration activty night ', avg_sum_1_list[2], ' (',avg_sum_1_list_pourcentage[2], '%)', ' - avg duration rest night ', avg_sum_1_list[3], ' (',avg_sum_1_list_pourcentage[3], '%)')
    print('\n avg segment activty day ', avg_segment_duration_list[0], ' - avg segment rest day ', avg_segment_duration_list[1], ' - avg segment activty night ', avg_segment_duration_list[2], ' - avg segment rest night ', avg_sum_1_list[3])
    return sum_duration_list, avg_sum_1_list, avg_sum_1_list_pourcentage, avg_segment_duration_list


def SD_features(df):
    # Calculate statistics for the 'SD_p' column

    #don't take the outliers in consideration
    df_no_outl = df.loc[df['SD_p'] <= 15]

    column_stats = df_no_outl['SD_p'].describe()

    # Extract the desired statistics
    mean = column_stats['mean']
    median = column_stats['50%']
    minimum = column_stats['min']
    maximum = column_stats['max']
    q1 = column_stats['25%']
    q2 = column_stats['50%']
    q3 = column_stats['75%']

    succ_diff = [(i-j)**2 for (i,j) in rpt.utils.pairwise(df_no_outl['SD_p'].to_numpy())]
    rMSSD = sqrt(sum(succ_diff) / len(succ_diff)) 

    print('mean ', mean, '\nminimum ', minimum, '\n25% ', q1,'\n50% ', q2, '\n75% ', q3, '\nRMSSD ', rMSSD)
    return mean, minimum, q1, q2, q3, rMSSD