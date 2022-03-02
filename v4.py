import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 300)

train_df = pd.read_csv("Data/train_df_v3.csv")
test_df = pd.read_csv("Data/test_df_v3.csv")

train_df['rr'] = train_df['rating']
train_df['rating'] = train_df['rating_rival']
train_df['rating_rival'] = train_df['rr']
train_df.drop(columns=['rr'], inplace=True)

test_df['rr'] = test_df['rating']
test_df['rating'] = test_df['rating_rival']
test_df['rating_rival'] = test_df['rr']
test_df.drop(columns=['rr'], inplace=True)


#time_control_name

print(train_df['time_control_name'].nunique())

tcn_count = train_df['time_control_name'].value_counts()
tcn_percent = tcn_count.apply(lambda x: x/tcn_count.sum())
print(tcn_percent)
tcn_percent.plot(kind="bar", rot=0, xlabel='time_control_name', ylabel="Percent",
                 title="Percent of time control names used")

plt.show()

time_points = train_df.groupby(['time_control_name'])['points'].mean()
print(time_points.head())
time_points.plot(kind="bar", rot=0, xlabel='time_control_name', ylabel="Average points",
                   title="Average points by time_control_name")
plt.show()

#Seems like more points is gained by players in games with ultrablitz time control name.

#mapping
time_cn_mapping = {'blitz': 1, 'rapid': 3, 'regular': 2, 'ultrablitz': 4}
train_df['time_control_name'] = train_df['time_control_name'].map(time_cn_mapping).astype(int)
test_df['time_control_name'] = test_df['time_control_name'].map(time_cn_mapping).astype(int)

#lexicon

lexicon_points = train_df.groupby(['lexicon'])['points'].mean()
print(lexicon_points)
lexicon_points.plot(kind="bar", rot=0, xlabel='Lexicon', ylabel="Average points", title="Average points by lexicon")

plt.show()

lexicon_count = train_df['lexicon'].value_counts()
lexicon_percent = lexicon_count.apply(lambda x: x/lexicon_count.sum())
print(lexicon_percent)
lexicon_percent.plot(kind="bar", rot=0, xlabel='lexicon', ylabel="Percent",
                       title="Percent of lexicons in games")
plt.show()

#There are small differncies between average points gained in turns in dependancy on lexicon used

#mapping

lexicon_mapping = {'NWL18': 1, 'NWL20': 2, 'CSW21': 3, 'CSW19': 4}
train_df['lexicon'] = train_df['lexicon'].map(lexicon_mapping).astype(int)
test_df['lexicon'] = test_df['lexicon'].map(lexicon_mapping).astype(int)

#initial_time_seconds

print(train_df['initial_time_seconds'].nunique())
its_count = train_df['initial_time_seconds'].value_counts()
initial_time_points = train_df.groupby(['initial_time_seconds'])['points'].mean()
print(initial_time_points)
initial_time_points.plot(kind="bar", rot=0, xlabel='initial_time', ylabel="Average points",
                       title="Average points by initial time")

plt.show()

it_percent = its_count.apply(lambda x: x/its_count.sum())
print(it_percent)

it_percent.plot(kind="bar", rot=0, xlabel='initial_time', ylabel="Percent",
                       title="Percent of initial times used in games")
plt.show()

#There are small differncies between average points gained in turns depend on initial time used

#replacing small samples
train_df['initial_time_seconds'] = train_df['initial_time_seconds'].replace([1380, 120, 15, 2400, 3300, 45, 240, 1440,
                                                                             420, 1260, 30, 180, 3000, 540, 720, 2100,
                                                                             1320, 960, 1140, 1020, 1080, 1800, 840,
                                                                             780, 480, 2700, 360], 99999)

test_df['initial_time_seconds'] = test_df['initial_time_seconds'].replace([1380, 120, 15, 2400, 3300, 45, 240, 1440,
                                                                           420, 1260, 30, 180, 3000, 540, 720, 2100,
                                                                           1320, 960, 1140, 1020, 1080, 1800, 840,
                                                                           780, 480, 2700, 360], 99999)

its_count = train_df['initial_time_seconds'].value_counts()
initial_time_points = train_df.groupby(['initial_time_seconds'])['points'].mean()
print(initial_time_points.sort_values())
initial_time_points.plot(kind="bar", rot=0, xlabel='initial_time', ylabel="Average points",
                       title="Average points by initial time")

plt.show()

it_percent = its_count.apply(lambda x: x/its_count.sum())
print(it_percent)

it_percent.plot(kind="bar", rot=0, xlabel='initial_time', ylabel="Percent",
                       title="Percent of initial times used in games")
plt.show()

#mapping

i_time_mapping = {900: 1, 660: 2, 300: 3, 99999: 4, 60: 5, 1200: 6, 3600: 7, 1500: 8, 600: 9}
train_df['initial_time_seconds'] = train_df['initial_time_seconds'].map(i_time_mapping).astype(int)
test_df['initial_time_seconds'] = test_df['initial_time_seconds'].map(i_time_mapping).astype(int)

#rating_mode
print(train_df['rating_mode'].nunique())
rm_count = train_df['rating_mode'].value_counts()
print(rm_count)
rm_percent = rm_count.apply(lambda x: x/rm_count.sum())
rm_percent.plot(kind="bar", rot=0, xlabel='rating mode', ylabel="Percent",
                       title="Percent of rating modes used in games")
plt.show()

rmode_points = train_df.groupby(['rating_mode'])['points'].mean()
print(rmode_points.head())
rmode_points.plot(kind="bar", rot=0, xlabel='rating mode', ylabel="Average points",
                   title="Average points by rating mode")
plt.show()


#There are small differncies between average points gained in turns depend on rating mode used. Much more games was
# played on rated mode

#mapping

rat_mode_mapping = {'RATED': 1, 'CASUAL': 2}
train_df['rating_mode'] = train_df['rating_mode'].map(rat_mode_mapping).astype(int)
test_df['rating_mode'] = test_df['rating_mode'].map(rat_mode_mapping).astype(int)

#rating

def ratingbins(score):
    if score < 1000:
        return 'A'
    if score < 1200 and score >= 1000:
        return 'B'
    if score < 1400 and score >= 1200:
        return 'C'
    if score < 1600 and score >= 1400:
        return 'D'
    if score < 1800 and score >= 1600:
        return 'E'
    if score < 2000 and score >= 1800:
        return 'F'
    if score < 2200 and score >= 2000:
        return 'G'
    if score < 2400 and score >= 2200:
        return 'H'

train_df['rating_bins'] = tuple(map(ratingbins, train_df['rating']))
rating_points = train_df.groupby(['rating_bins'])['points'].mean()

rating_points.plot(kind="bar", rot=0, xlabel='Rating bins', ylabel="Average points",
                   title="Average points by ratings")
plt.show()

train_df.drop(columns=['rating_bins'], inplace=True)
#There is difference between points gained and raing

#move_lc_sum

print(train_df['move_lc_sum'].max())
print(train_df['move_lc_sum'].min())
def movelcsumbins(score):
    if score < 30:
        return 'A'
    if score < 40 and score >= 30:
        return 'B'
    if score < 50 and score >= 40:
        return 'C'
    if score < 60 and score >= 50:
        return 'D'
    if score < 70 and score >= 60:
        return 'E'
    if score < 80 and score >= 70:
        return 'F'
    if score < 90 and score >= 80:
        return 'G'
    if score >= 90:
        return 'H'

train_df['movelcsum_bins'] = tuple(map(movelcsumbins, train_df['move_lc_sum']))
movelcsum_points = train_df.groupby(['movelcsum_bins'])['points'].mean()

movelcsum_points.plot(kind="bar", rot=0, xlabel='Rating bins', ylabel="Average points",
                   title="Average points sum of letters used")
plt.show()

train_df.drop(columns=['movelcsum_bins'], inplace=True)



#move_vow_sum

print(train_df['move_vow_sum'].max())
print(train_df['move_vow_sum'].min())
def movelcsumbins(score):
    if score < 15:
        return 'A'
    if score < 20 and score >= 15:
        return 'B'
    if score < 25 and score >= 20:
        return 'C'
    if score < 30 and score >= 25:
        return 'D'
    if score < 35 and score >= 30:
        return 'E'
    if score < 40 and score >= 35:
        return 'F'
    if score < 45 and score >= 40:
        return 'G'
    if score >= 45:
        return 'H'

train_df['movevowsum_bins'] = tuple(map(movelcsumbins, train_df['move_vow_sum']))
movevowsum_points = train_df.groupby(['movevowsum_bins'])['points'].mean()

movevowsum_points.plot(kind="bar", rot=0, xlabel='Rating bins', ylabel="Average points",
                   title="Average points sum of vowels used")
plt.show()

train_df.drop(columns=['movevowsum_bins'], inplace=True)



#move_con_sum

print(train_df['move_con_sum'].max())
print(train_df['move_con_sum'].min())
def moveconsumbins(score):
    if score < 20:
        return 'A'
    if score < 30 and score >= 20:
        return 'B'
    if score < 40 and score >= 30:
        return 'C'
    if score < 50 and score >= 40:
        return 'D'
    if score < 60 and score >= 50:
        return 'E'
    if score < 70 and score >= 60:
        return 'F'
    if score >= 70:
        return 'G'

train_df['moveconsum_bins'] = tuple(map(moveconsumbins, train_df['move_con_sum']))
moveconsum_points = train_df.groupby(['moveconsum_bins'])['points'].mean()

moveconsum_points.plot(kind="bar", rot=0, xlabel='Rating bins', ylabel="Average points",
                   title="Average points sum of consonants used")
plt.show()

train_df.drop(columns=['moveconsum_bins'], inplace=True)

print(train_df.info())
#print(train_df.corr())

#rack_length
rack_count = train_df['rack_length'].value_counts()
rack_percent = rack_count.apply(lambda x: x/rack_count.sum())
print(rack_percent)
rack_percent.plot(kind="bar", rot=0, xlabel='Rack length', ylabel="Percent",
                  title="Percent of rack lengths")
plt.show()
rack_points = train_df.groupby(['rack_length'])['points'].mean().sort_values(ascending=False)
rack_points.plot(kind="bar", rot=0, xlabel='Rack length', ylabel="Average points",
                  title="Average points by rack length")
plt.show()


#rack vowels
vowel_points = train_df.groupby(['rack_vowels'])['points'].mean()
vowel_points. plot(kind="bar", rot=0, xlabel='Number of vowels', ylabel="Average points",
                   title="Average points gy number of vowels in rack")
plt.show()

#rack diff
diff_points = train_df.groupby(['rack_diff'])['points'].mean()
diff_points . plot(kind="bar", rot=0, xlabel='Difference between consonants and vowels in rack', ylabel="Average points",
                   title="Average points by difference between consonants and vowels in rack")
plt.show()

#rack_repeats
repeats_points = train_df.groupby(['rack_repeats'])['points'].mean()
repeats_points . plot(kind="bar", rot=0, xlabel='Number of unique letters', ylabel="Average points",
                   title="Average points by number of unique letters in rack")
plt.show()



print(train_df['score_rival'].max())
print(train_df['score_rival'].min())


def scorerivbins(score):
    if score < 100:
        return 'A'
    if score < 200 and score >= 100:
        return 'B'
    if score < 300 and score >= 200:
        return 'C'
    if score < 400 and score >= 300:
        return 'D'
    if score < 500 and score >= 400:
        return 'E'
    if score >= 500:
        return 'F'

train_df['scoreriv_bins'] = tuple(map(scorerivbins, train_df['score_rival']))
scoreriv_points = train_df.groupby(['scoreriv_bins'])['points'].mean()

scoreriv_points.plot(kind="bar", rot=0, xlabel='Rival score bins', ylabel="Average points",
                   title="Average points by rival score")
plt.show()

train_df.drop(columns=['scoreriv_bins'], inplace=True)


print(train_df['score'].max())
print(train_df['score'].min())


def scorerivbins(score):
    if score < 100:
        return 'A'
    if score < 200 and score >= 100:
        return 'B'
    if score < 300 and score >= 200:
        return 'C'
    if score < 400 and score >= 300:
        return 'D'
    if score < 500 and score >= 400:
        return 'E'
    if score >= 500:
        return 'F'

train_df['score_bins'] = tuple(map(scorerivbins, train_df['score']))
score_points = train_df.groupby(['score_bins'])['points'].mean()

score_points.plot(kind="bar", rot=0, xlabel='Score bins', ylabel="Average points",
                   title="Average points by score")
plt.show()

train_df.drop(columns=['score_bins'], inplace=True)



#rating_rival

def ratingbins(score):
    if score < 1000:
        return 'A'
    if score < 1200 and score >= 1000:
        return 'B'
    if score < 1400 and score >= 1200:
        return 'C'
    if score < 1600 and score >= 1400:
        return 'D'
    if score < 1800 and score >= 1600:
        return 'E'
    if score < 2000 and score >= 1800:
        return 'F'
    if score < 2200 and score >= 2000:
        return 'G'
    if score < 2400 and score >= 2200:
        return 'H'

train_df['ratingriv_bins'] = tuple(map(ratingbins, train_df['rating_rival']))
ratingriv_points = train_df.groupby(['ratingriv_bins'])['points'].mean()

ratingriv_points.plot(kind="bar", rot=0, xlabel='Rival rating bins', ylabel="Average points",
                   title="Average points by rival ratings")
plt.show()

train_df.drop(columns=['ratingriv_bins'], inplace=True)



train_df.drop(columns=['rack_length', 'rating_mode', 'score', 'score_rival', 'move_con_sum'], inplace=True)
test_df.drop(columns=['rack_length', 'rating_mode', 'score', 'score_rival', 'move_con_sum'], inplace=True)


print(train_df.corr())
print(test_df.info())
print(train_df.info())

train_df.to_csv('Data/train_df_v4.csv', index=False)
test_df.to_csv('Data/test_df_v4.csv', index=False)
