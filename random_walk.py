import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import splrep, splev

def generate_list_of_int(row):
    k = round(row['duration'])
    value = 1 if row['activity'] else -1
    return [value] * k


def generate_alternating_list(signal, bkps):

    spl = splrep(x=np.arange(signal.shape[0]), y=signal, task=-1, t=bkps[:-1], k=1)
    approx = splev(np.arange(signal.shape[0]), spl)

    length = len(signal)
    if length <= len(bkps):
        raise ValueError("The length of the signal must be greater than the number of bkps.")
    elif length <= bkps[-1]:
        raise ValueError("The index of the last bkps must be lower than the length of the signal.")

    alternating_list = []
    counting_list = []
    count=0
    alternating = True if signal[1]>0 else False
    for i in range(length):
        # If the current index is in the list of indexes, switch the value
        alternating_list.append(alternating)
        counting_list.append(count)
        count += 1
        if i == length-1:
            pass
        elif i in bkps:
            if (alternating == True and approx[i]>approx[i+1]) or (alternating == False and approx[i]<approx[i+1]):
                alternating = not alternating
                count=1
            else:
                pass
    return alternating_list, counting_list


    
def features_random_walk_day_night(df, day_or_night, day_or_night_number, plot = False):
    day_df = df[df['day_or_night'] == 'day']
    number_days = day_df['day_or_night_number'].iloc[-1]

    night_df = df[df['day_or_night'] == 'night']
    number_nights = night_df['day_or_night_number'].iloc[-1]
    
    #ERROR
    if day_or_night != 'day' and day_or_night != 'night':
        raise ValueError("'day_or_night' must be equal to 'day' or 'night'")
    if day_or_night == 'day' and day_or_night_number <1:
        raise ValueError('day_or_night_number is too low')
    if day_or_night == 'night' and day_or_night_number <0:
        raise ValueError('day_or_night_number is too low')
    if day_or_night == 'day' and day_or_night_number > number_days:
        raise ValueError('day_or_night_number is too high')
    if day_or_night == 'night' and day_or_night_number > number_nights:
        raise ValueError('day_or_night_number is too high')
    
    filtered_df = df[(df['day_or_night'] == day_or_night) & (df['day_or_night_number'] == day_or_night_number)]

    list_of_int = []
    for index, row in filtered_df.iterrows():
        list_of_int.extend(generate_list_of_int(row))
    list_of_int.insert(0, 0)

    list_random_walk = np.empty(0,)
    cumulative_sum_day = 0

    for num in list_of_int:
        cumulative_sum_day += num
        list_random_walk = np.append(list_random_walk, cumulative_sum_day)

    list_random_walk = list_random_walk/3600
    if plot == True :
        #changer les couleurs de la vraie marche aléatoire, de son approximation, changer la taille de la figure (plus grande)
        plt.figure(figsize=(10, 2))
        plt.plot(list_random_walk, label=day_or_night + str(day_or_night_number), c='lightblue')
        bkps = rpt.KernelCPD().fit(signal=np.diff(list_random_walk)).predict(pen=5e-6*np.log(list_random_walk.shape[0]))
        spl = splrep(x=np.arange(list_random_walk.shape[0]), y=list_random_walk, task=-1, t=bkps[:-1], k=1)
        approx = splev(np.arange(list_random_walk.shape[0]), spl)
        plt.plot(approx, c= 'darkblue', linestyle='--')
        plt.margins(0)
        for b in bkps[:-1]:
            plt.axvline(b, c="0.8", ls="--", lw = 1)

    activity_list, counting_list = generate_alternating_list(list_random_walk, bkps)

    df_random_walk = pd.DataFrame({'random walk': list_random_walk, 'activity': activity_list, 'counting' : counting_list})

    group_key = ( (df_random_walk['activity'] != df_random_walk['activity'].shift())).cumsum()

    merged_df_random_walk = df_random_walk.groupby(group_key).agg({
        'random walk': 'first',
        'activity': 'first',
        'counting': 'last'
    })

    return df_random_walk.loc[df_random_walk['activity'] == True].count()[0], df_random_walk.loc[df_random_walk['activity'] == False].count()[0], df_random_walk.loc[df_random_walk['activity'] == True].count()[0]/merged_df_random_walk.loc[merged_df_random_walk['activity'] == True].count()[0], df_random_walk.loc[df_random_walk['activity'] == False].count()[0]/merged_df_random_walk.loc[merged_df_random_walk['activity'] == False].count()[0], merged_df_random_walk.loc[merged_df_random_walk['activity'] == True].count()[0], merged_df_random_walk.loc[merged_df_random_walk['activity'] == False].count()[0], merged_df_random_walk.loc[merged_df_random_walk['activity'] == True, 'counting'].max()



def plot_random_walk(df):

    day_df = df[df['day_or_night'] == 'day']
    number_days = day_df['day_or_night_number'].iloc[-1]

    night_df = df[df['day_or_night'] == 'night']
    number_nights = night_df['day_or_night_number'].iloc[-1]

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 4), layout='constrained')
    
    color = '#FADB33'
    for d in range(number_days):
        # Create an empty list to store the results
        list_of_int_day = []

        # Iterate through each row and populate the 'list_of_int'
        for index, row in df[(df['day_or_night'] == 'day') & (df['day_or_night_number'] == d+1)].iterrows():
            list_of_int_day.extend(generate_list_of_int(row))

        list_of_int_day.insert(0, 0)

        list_random_walk_day = []

        cumulative_sum_day = 0

        for num in list_of_int_day:
            cumulative_sum_day += num
            list_random_walk_day.append(cumulative_sum_day)

        ax1.plot(list_random_walk_day, c=color, label='day '+str(d+1))
        ax1.set_title('day')
    
    color = '#2B2C5A'
    for n in range(number_nights+1):
        # Create an empty list to store the results
        list_of_int_night = []

        # Iterate through each row and populate the 'list_of_int'
        for index, row in df[(df['day_or_night'] == 'night') & (df['day_or_night_number'] == n)].iterrows():
            list_of_int_night.extend(generate_list_of_int(row))

        list_of_int_night.insert(0, 0)

        list_random_walk_night = []

        cumulative_sum_night = 0

        for num in list_of_int_night:
            cumulative_sum_night += num
            list_random_walk_night.append(cumulative_sum_night)

        ax2.plot(list_random_walk_night, c=color, label='night ' + str(n))
        ax2.set_title('night')
    
    ax1.legend(loc='best')
    ax2.legend(loc='best')