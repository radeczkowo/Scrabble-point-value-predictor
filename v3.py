import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
pd.set_option('display.max_columns', 25)
pd.set_option('display.max_rows', 100)
train_df = pd.read_csv("Data/turns_train.csv")
games_df = pd.read_csv("Data/games.csv")
scores_df = pd.read_csv("Data/scores.csv")
test_df = pd.read_csv("Data/turns_test.csv")
train_df = train_df.merge(games_df, on='game_id')
train_df = train_df.merge(scores_df, on='game_id')

test_df = test_df.merge(games_df, on='game_id')
test_df = test_df.merge(scores_df, on='game_id')
#print(test_df.head())
#print(train_df.head())
#print(train_df.info())
#print(train_df.describe())


def calculateletters(move, turn):
    count = 0
    move = str(move)
    for letter in move:
        if letter.isalpha():
            count = count+1
    if(turn == 20):
        count = 0
    return count

def movevowelcount(move, turn):
    counter = 0
    for letter in str(move):
        if (letter == 'E' or letter == 'A' or letter == 'O' or letter == 'I' or letter == 'U' or letter == 'Y'
                or letter == '?'):
            counter = counter+1
    if(turn == 20):
        counter = 0
    return counter

def moveconsonantcount(rack, turn):
    counter = 0
    for letter in str(rack):
        if letter == 'E' or letter == 'A' or letter == 'O' or letter == 'I' or letter == 'U' or letter == 'Y' \
                or letter == '.' or letter == '(' or letter == ')' or letter == '-':
            pass
        else:
            counter = counter+1
    if(turn == 20):
        counter = 0
    return counter


test_df = test_df.loc[test_df['turn_number'] <= 20]
train_df = train_df.loc[train_df['turn_number'] <= 20]
#print(train_df.info())

train_df['move_lc'] = tuple(map(calculateletters, train_df['move'], train_df['turn_number']))
test_df['move_lc'] = tuple(map(calculateletters, test_df['move'], test_df['turn_number']))

train_df['move_lc_sum'] = train_df['move_lc'].rolling(window=40).sum()/2
test_df['move_lc_sum'] = test_df['move_lc'].rolling(window=40).sum()/2

train_df['move_vow'] = tuple(map(movevowelcount, train_df['move'], train_df['turn_number']))
test_df['move_vow'] = tuple(map(movevowelcount, test_df['move'], test_df['turn_number']))

train_df['move_vow_sum'] = train_df['move_vow'].rolling(window=40).sum()/2
test_df['move_vow_sum'] = test_df['move_vow'].rolling(window=40).sum()/2

train_df['move_con'] = tuple(map(moveconsonantcount, train_df['move'], train_df['turn_number']))
test_df['move_con'] = tuple(map(moveconsonantcount, test_df['move'], test_df['turn_number']))

train_df['move_con_sum'] = train_df['move_con'].rolling(window=40).sum()/2
test_df['move_con_sum'] = test_df['move_con'].rolling(window=40).sum()/2


train_df.drop(columns=['move_con', 'move_vow', 'move_lc'], inplace=True)
test_df.drop(columns=['move_con', 'move_vow', 'move_lc'], inplace=True)

train_df = train_df.loc[train_df['turn_number'] >= 18]
test_df = test_df.loc[test_df['turn_number'] >= 18]
#print(train_df.info())
#print(train_df.describe())
#print(train_df.corr())


#turn_number


turn_points_df = train_df.groupby(['turn_number'])['points'].mean()
turn_points_df.plot(kind="bar", rot=0,
                    xlabel='turn number', ylabel="Average Points", title="Average points by turn number")

plt.show()

turn_number_count = train_df['turn_number'].value_counts()
turn_number_count.plot(kind="bar", rot=0,
                       xlabel='turn number', ylabel="how many times turn number get reached",
                       title="Count of total turn numbers")
plt.show()

#nickname_x

nickname_count = train_df['nickname_x'].value_counts()
print(train_df['nickname_x'].nunique())
print(nickname_count)
nickname_count_percent = nickname_count.apply(lambda x: x/nickname_count.sum())
print(nickname_count_percent)
#As expected bot turns are equal nearly 50% of total registered turns


#rack
rack_count = train_df['rack'].value_counts()
print(rack_count.tail(20))
train_df['rack_length'] = train_df['rack'].str.len()
test_df['rack_length'] = test_df['rack'].str.len()
print(train_df['rack_length'])
rack_length_by_turn = train_df.groupby(['turn_number'])['rack_length'].mean()
print(rack_length_by_turn)

rack_length_by_turn.plot(kind="bar", rot=0,
                    xlabel='turn number', ylabel="Average rack length", title="Average rack length by turn number")


rack_points = train_df.groupby(['rack'])['points'].mean().sort_values(ascending=False)
print(rack_points.head())
plt.show()

def rackvowelcount(rack):
    counter = 0
    for letter in str(rack):
        if (letter == 'E' or letter == 'A' or letter == 'O' or letter == 'I' or letter == 'U' or letter == 'Y'
                or letter == '?'):
            counter = counter+1
    return counter


def rackconsonantcount(rack):
    counter = 0
    for letter in str(rack):
        if letter == 'E' or letter == 'A' or letter == 'O' or letter == 'I' or letter == 'U' or letter == 'Y':
            pass
        else:
            counter = counter+1
    return counter


def rackrepeats(rack):
    return len(set(str(rack)))


def consonantsvowelsdiff(consonants, vowels):
    return abs(consonants-vowels)


train_df['rack_vowels'] = tuple(map(rackvowelcount, train_df['rack']))
train_df['rack_consonants'] = tuple(map(rackconsonantcount, train_df['rack']))
train_df['rack_diff'] = tuple(map(consonantsvowelsdiff, train_df['rack_consonants'], train_df['rack_vowels']))
train_df['rack_repeats'] = tuple(map(rackrepeats, train_df['rack']))

test_df['rack_vowels'] = tuple(map(rackvowelcount, test_df['rack']))
test_df['rack_consonants'] = tuple(map(rackconsonantcount, test_df['rack']))
test_df['rack_diff'] = tuple(map(consonantsvowelsdiff, test_df['rack_consonants'], test_df['rack_vowels']))
test_df['rack_repeats'] = tuple(map(rackrepeats, test_df['rack']))


vowel_points = train_df.groupby(['rack_vowels'])['points'].mean()
vowel_points. plot(kind="bar", rot=0, xlabel='Number of vowels', ylabel="Average points",
                   title="Average points gy number of vowels in rack")
plt.show()

consonants_points = train_df.groupby(['rack_consonants'])['points'].mean()
consonants_points. plot(kind="bar", rot=0, xlabel='Number of consonants', ylabel="Average points",
                   title="Average points gy number of consonants in rack")
plt.show()

repeats_points = train_df.groupby(['rack_repeats'])['points'].mean()
repeats_points . plot(kind="bar", rot=0, xlabel='Number of unique letters', ylabel="Average points",
                   title="Average points by number of unique letters in rack")
plt.show()

diff_points = train_df.groupby(['rack_diff'])['points'].mean()
diff_points . plot(kind="bar", rot=0, xlabel='Difference between consonants and vowels in rack', ylabel="Average points",
                   title="Average points by difference between consonants and vowels in rack")
plt.show()
#rack lenght is equal 7 in all turns until 19 where it begins to decrease a little.


#location
location_count = train_df['location'].value_counts()
print(location_count)
print(train_df['location'].nunique())
location_count_percent = location_count.apply(lambda x: x/location_count.sum())
print(location_count_percent)
location_df = train_df.groupby(['location'])['points'].mean().sort_values(ascending=False)
print(location_df)
#There is a difference between average points gained from different locations

#move

move_count = train_df['move'].value_counts()
print(move_count)
print(train_df['move'].nunique())
train_df['move_length'] = train_df['move'].str.len()
test_df['move_length'] = test_df['move'].str.len()
move_length_by_turn = train_df.groupby(['turn_number'])['move_length'].mean()
move_length_by_turn.plot(kind="bar", rot=0,
                    xlabel='turn number', ylabel="Average move length", title="Average move length by turn number")
print(train_df.corr())
plt.show()
move_length_by_points = train_df.groupby(['move_length'])['points'].mean()
move_length_by_points.plot(kind="bar", rot=0, xlabel='move length', ylabel="Average points",
                           title="Average points by move length")
plt.show()
move_length_rack_diff = train_df.groupby(['rack_diff'])['move_length'].mean()
move_length_rack_diff.plot(kind="bar", rot=0, xlabel='rack_diff', ylabel="move length",
                           title="Average move length by rack_diff")
plt.show()

#Definitely there is a realtion between move length and points gained


#score_x
score_points_df = train_df.groupby(['score_x'])['points'].mean()
score_points_df.plot(kind="bar", rot=0, xticks=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
                    xlabel='score', ylabel="Average Points", title="Average points by score")

plt.show()
#Seems like average points number is increasing alongside score_x

#turn_type
turn_type_count = train_df['turn_type'].value_counts()
print(turn_type_count)
turn_type_count_percent = turn_type_count.apply(lambda x: x/turn_type_count.sum())
print(turn_type_count_percent)
#Nearly 98%  of turn types are play


#first

first = train_df['first'].value_counts()
print(train_df['first'].nunique())
print(first)
first_percent = first.apply(lambda x: x/first.sum())
print(first_percent)
#As expected similar with nickname_x

#time_control_name
print(train_df['time_control_name'].nunique())
time_points = train_df.groupby(['time_control_name'])['points'].mean()
print(time_points.head())
time_points.plot(kind="bar", rot=0, xlabel='time_control_name', ylabel="Average points",
                   title="Average points by time_control_name")
plt.show()

#Seems like more points is gained by players in games with ultrablitz time control name.

#game_end_reason

print(train_df['game_end_reason'].nunique())
ger = train_df['game_end_reason'].value_counts()
ger_percent = ger.apply(lambda x: x/ger.sum())
print(ger_percent)
ger_percent.plot(kind="bar", rot=0, xlabel='game_end_reason', ylabel="Percent",
                       title="Percent of total ending reasons")
plt.show()
ger_points = train_df.groupby(['game_end_reason'])['points'].mean()
print(ger_points.head())
ger_points.plot(kind="bar", rot=0, xlabel='game_end_reason', ylabel="Average points",
                   title="Average points by game_end_reason")
plt.show()


#Most of ending rasons are standard

#winner
winner_count = train_df['winner'].value_counts()
winner_percent = winner_count.apply(lambda x: x/winner_count.sum())
print(winner_percent)
winner_percent.plot(kind="bar", rot=0, xlabel='winner', ylabel="Percent",
                       title="Percent of total wins by starting person")

plt.show()
#More games has been won by 0

#created_at
print(train_df['created_at'].nunique())
#completely unrelevant data


#lexicon

lexicon_points = train_df.groupby(['lexicon'])['points'].mean()
lexicon_points.plot(kind="bar", rot=0, xlabel='Lexicon', ylabel="Average points", title="Average points by lexicon")

plt.show()

lexicon_count = train_df['lexicon'].value_counts()
lexicon_percent = lexicon_count.apply(lambda x: x/lexicon_count.sum())
print(lexicon_percent)
lexicon_percent.plot(kind="bar", rot=0, xlabel='lexicon', ylabel="Percent",
                       title="Percent of lexicons in games")
plt.show()
#There are small differncies between average points gained in turns in dependancy on lexicon used

#initial_time_seconds

print(train_df['initial_time_seconds'].nunique())
print(train_df['initial_time_seconds'].value_counts())
initial_time_points = train_df.groupby(['initial_time_seconds'])['points'].mean()
print(initial_time_points)
initial_time_points.plot(kind="bar", rot=0, xlabel='initial_time', ylabel="Average points",
                       title="Average points by initial time")
it_percent = train_df['initial_time_seconds'].value_counts().apply(lambda x: x/train_df['initial_time_seconds'].value_counts().sum())

it_percent.plot(kind="bar", rot=0, xlabel='initial_time', ylabel="Percent",
                       title="Percent of initial times used in games")
plt.show()
#There are small differncies between average points gained in turns depend on initial time used


#increment_seconds
print(train_df['increment_seconds'].nunique())
print(train_df['increment_seconds'].value_counts())
is_percent = train_df['increment_seconds'].value_counts().apply(lambda x: x/train_df['increment_seconds'].value_counts().sum())

is_percent.plot(kind="bar", rot=0, xlabel='increment seconds', ylabel="Percent",
                       title="Percent of increment seconds used in games")
plt.show()

#Increment seconds columns is completely dominated by 

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

#max_overtime_minutes
print(train_df['max_overtime_minutes'].nunique())
mom_count = train_df['max_overtime_minutes'].value_counts()
print(mom_count)
mom_percent = mom_count.apply(lambda x: x/mom_count.sum())
mom_percent.plot(kind="bar", rot=0, xlabel='max_overtime_minutes', ylabel="Percent",
                       title="Percent of max_overtime_minutes used in games")
plt.show()
mov_points = train_df.groupby(['max_overtime_minutes'])['points'].mean()
print(mov_points.head())
mov_points.plot(kind="bar", rot=0, xlabel='max overtime minutes', ylabel="Average points",
                   title="Average points by max overtime minutes")
plt.show()

#There are small differncies between average points gained in turns depend on overtime minutes. Much more games was
# had 0 overtime minutes


#game_duration_seconds

print(train_df['game_duration_seconds'].nunique())
print(train_df['game_duration_seconds'].value_counts())
print(train_df['game_duration_seconds'].min())
print(train_df['game_duration_seconds'].max())

def gamedurationbins(gameduration):
    if gameduration < 500:
        return 'A'
    if gameduration < 1000 and gameduration >= 500:
        return 'B'
    if gameduration < 1500 and gameduration >= 1000:
        return 'C'
    if gameduration < 2000 and gameduration >= 1500:
        return 'D'
    if gameduration < 2500 and gameduration >= 2000:
        return 'E'
    if gameduration < 3000 and gameduration >= 2500:
        return 'F'
    if gameduration < 3500 and gameduration >= 3000:
        return 'G'
    if gameduration < 4000 and gameduration >= 3500:
        return 'H'

train_df['gdur_bins'] = tuple(map(gamedurationbins, train_df['game_duration_seconds']))
mov_points = train_df.groupby(['gdur_bins'])['points'].mean()
print(mov_points.head(10))
mov_points.plot(kind="bar", rot=0, xlabel='Game duration bins', ylabel="Average points",
                   title="Average points by game duration")
plt.show()
#There are small differncies between average points gained in turns depend on game duration.

#nickname_y
#same as nickname_x
train_df.drop(columns=['gdur_bins'], inplace=True)

#score_y

scorey_points_df = train_df.groupby(['score_y'])['points'].mean()
scorey_points_df.plot(kind="bar", rot=0, xticks=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
                    xlabel='score_y', ylabel="Average Points", title="Average points by score_y")

plt.show()
#Seems like average points number is increasing alongside score_y

#rating

#deleting the 'y' from rating values
df_rating = train_df['rating'].str.split(pat="?", expand=True)
train_df['rating'] = pd.to_numeric(df_rating[0], downcast='integer')
df_rating = test_df['rating'].str.split(pat="?", expand=True)
test_df['rating'] = pd.to_numeric(df_rating[0], downcast='integer')


#print(train_df['rating'].nunique())
#print(train_df['rating'].value_counts())
#print(train_df['rating'].nunique())
#print(train_df['rating'].value_counts())
#print(train_df['rating'].min())
#print(train_df['rating'].max())


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
#print(rating_points.head(10))

rating_points.plot(kind="bar", rot=0, xlabel='Rating bins', ylabel="Average points",
                   title="Average points by ratings")
plt.show()

train_df.drop(columns=['rating_bins'], inplace=True)
#There is difference between points gained and raing



#cunt of total letters used for move

train_df['score_rival'] = train_df['score_x'].shift(2)
train_df['score'] = train_df['score_x'].shift(4)
test_df['score_rival'] = test_df['score_x'].shift(2)
test_df['score'] = test_df['score_x'].shift(4)

#droping NA test columns
test_df.drop(columns=['game_end_reason', 'move', 'winner', 'game_duration_seconds', 'score_y', 'location', 'score_x',
                      'turn_type'], inplace=True)
train_df.drop(columns=['game_end_reason', 'move', 'winner', 'game_duration_seconds', 'score_y', 'location', 'score_x',
                       'turn_type'], inplace=True)


train_df = train_df.loc[train_df['turn_number'] == 20]
test_df = test_df.loc[test_df['turn_number'] == 20]



#print(train_df.info())
#print(train_df.head(10))
#print(train_df.corr())

train_regression = train_df[['score', 'rating', 'score_rival', 'move_lc_sum']].dropna()
X = train_regression[['score', 'score_rival', 'move_lc_sum']]
y = train_regression['rating']
regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_)

def calculatena(x1, x2, x3):
    return regr.predict([[x1, x2, x3]])

#filling rating NA values
nans = train_df[['score', 'rating', 'score_rival', 'move_lc_sum']][np.isnan(train_df['rating'])]
new_values = tuple(map(calculatena, nans['score'], nans['score_rival'], nans['move_lc_sum']))
train_df['rating'][np.isnan(train_df['rating'])] = new_values
nans = test_df[['score', 'rating', 'score_rival', 'move_lc_sum']][np.isnan(test_df['rating'])]
new_values = tuple(map(calculatena, nans['score'], nans['score_rival'], nans['move_lc_sum']))
test_df['rating'][np.isnan(test_df['rating'])] = new_values


train_df['rating_rival'] = train_df.groupby(['game_id'])['rating'].transform(np.roll, shift=1)
test_df['rating_rival'] = test_df.groupby(['game_id'])['rating'].transform(np.roll, shift=1)


#Droping the players whos didn;t make 20th turn
train_df = train_df.loc[train_df['nickname_x'] != train_df['nickname_y']]
test_df = test_df.loc[test_df['nickname_x'] != test_df['nickname_y']]


#droping other columns
test_df.drop(columns=['nickname_x', 'turn_number', 'rack', 'created_at', 'nickname_y', 'move_length',
                      'increment_seconds', 'max_overtime_minutes', 'first', 'rack_consonants'], inplace=True)
train_df.drop(columns=['nickname_x', 'turn_number', 'rack', 'created_at', 'nickname_y', 'move_length',
                       'increment_seconds', 'max_overtime_minutes', 'first', 'rack_consonants'], inplace=True)

print(train_df.info())
print(test_df.info())
print(train_df.head())
print(test_df.head())
print(train_df.corr())


train_df.to_csv('Data/train_df_v3.csv', index=False)
test_df.to_csv('Data/test_df_v3.csv', index=False)