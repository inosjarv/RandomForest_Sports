import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
class DataProcessor(object):
    
    def __init__(self, filename):
        self.filename = filename

    def run(self):
        df = pd.read_csv(self.filename)
        length = len(df)

        df_len = df.shape[0]

        home_info = df[["League", "Home Team", "S.1", "EX.1", "EN.1", "F.1", "I.1", "R.1", "Away Team",
                        "S", "EX", "EN", "F", "I", "R", "Results.1"]]
        away_info = df[["League", "Away Team", "S", "EX", "EN", "F", "I", "R", "Home Team",
                        "S.1", "EX.1", "EN.1", "F.1", "I.1", "R.1", "Results"]]

        home_info = home_info.rename(index=str, columns={"League": "League", "Home Team": "Team","S.1": "S_own", 
                                            "EX.1": "EX_own", "EN.1": "EN_own", "F.1": "F_own", "I.1": "I_own",
                                            "R.1": "R_own", "Away Team":"Opponent Team", "S": "S_opp", "EX": "EX_opp", "EN": "EN_opp",
                                            "F": "F_opp", "I":"I_opp", "R": "R_opp", "Results.1": "Result"})
        away_info = away_info.rename(index=str, columns={"League":"League", "Away Team": "Team","S": "S_own", 
                                            "EX": "EX_own", "EN": "EN_own", "F": "F_own", "I": "I_own",
                                            "R": "R_own","Home Team":"Opponent Team", "S.1": "S_opp", "EX.1": "EX_opp", "EN.1": "EN_opp",
                                            "F.1": "F_opp", "I.1":"I_opp", "R.1": "R_opp", "Results": "Result"})

        df = home_info.append(away_info,ignore_index=True)


        cols_to_norm = ['S_own', 'EX_own', 'EN_own', 'F_own', 'I_own',
            'R_own', 'S_opp', 'EX_opp', 'EN_opp', 'F_opp', 'I_opp','R_opp']
        

        max_val = {}
        def max_dict(df, column):
            max_val[column] = df[column].max()
            return max_val
            
        for column in cols_to_norm:
            max_dict(df, column)

        df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x/x.max()))


        teams = df['Team'].drop_duplicates().values.tolist()
        league =df['League'].drop_duplicates().values.tolist()
        result = df['Result'].drop_duplicates().values.tolist()

        index_team = np.arange(0, len(teams),dtype=int)
        d_team = {k:v for k,v in zip(teams,index_team)}
        
        index_league = np.arange(0, len(league), dtype=int)
        d_league = {k:v for k,v in zip(league,index_league)}

        index_result = np.arange(0, len(result), dtype=int)
        d_result = {k:v for k,v in zip(result,index_result)}

        df['Result'] = df['Result'].replace(d_result)
        df['Team'] = df['Team'].replace(d_team)
        df['Opponent Team'] = df['Opponent Team'].replace(d_team)
        df['League'] = df['League'].replace({k:v for k,v in zip(league,index_league)})

        length = len(df['League'])

        ones = [1] * length
        zeros = [0] * length
        home_situation = ones + zeros

        home_situation = pd.DataFrame(home_situation)

        df["Home Situation"] = home_situation

        df = df[['League', 'Team', 'S_own', 'EX_own', 'EN_own', 'F_own', 'I_own',
            'R_own', 'Opponent Team', 'S_opp', 'EX_opp', 'EN_opp', 'F_opp', 'I_opp',
            'R_opp','Home Situation' ,'Result']]

        count_rows = pd.Series.value_counts(df['Team'], sort=False)
        max_num_row = max(count_rows)

        times_list=list()
        for index in count_rows:
            times = np.ceil(max_num_row/index)
            times_list.append(times)

        times_list = np.array(times_list, dtype=int)

        column_names = df.columns
        len(count_rows)
		
        filename = 'team_dict'
        outfile = open(filename, 'wb')
        pickle.dump(d_team, outfile)
        outfile.close()

        league_file = "league_dict"
        outfile = open(league_file, 'wb')
        pickle.dump(d_league, outfile)
        outfile.close()

        

        
        max_file = "max_dict"
        outfile = open(max_file, 'wb')
        pickle.dump(max_val, outfile)
        outfile.close()

        new_dataframe = pd.DataFrame(columns=column_names)

        flag = 0
        for i in range(len(count_rows)):
            temp_df = df['Team'] == flag
            new_dataframe = new_dataframe.append(pd.concat([df[temp_df]]*times_list[i], ignore_index=True))
            flag+=1

        return new_dataframe

if __name__ == "__main__":
    csv_file = DataProcessor(filename='data.csv')
    csv_file.run()

