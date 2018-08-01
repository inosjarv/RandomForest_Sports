import pandas as pd
import pickle


class RowProcessor(object):
    
    def __init__(self, filename):
        self.filename = filename

    def run(self):
        row = pd.read_csv(self.filename)
        #row = data[2:3]
        row = row[["League", "Home Team", "S.1", "EX.1", "EN.1", "F.1", "I.1", "R.1", "Away Team",
                                "S", "EX", "EN", "F", "I", "R"]]

        row = row.rename(index=str, columns={"League": "League", "Home Team": "Team","S.1": "S_own", 
                                                    "EX.1": "EX_own", "EN.1": "EN_own", "F.1": "F_own", "I.1": "I_own",
                                                    "R.1": "R_own", "Away Team":"Opponent Team", "S": "S_opp", "EX": "EX_opp", 
                                                    "EN": "EN_opp","F": "F_opp", "I":"I_opp", "R": "R_opp"})


        max_dict = {}
        max_file = open('max_dict', 'rb')
        max_dict = pickle.load(max_file)
        max_file.close()

        def preprocess_row(dataframe, column_name, dictionary_file):
            temp_dict = {}
            temp_file = open(dictionary_file, 'rb')
            temp_dict = pickle.load(temp_file)
            temp_file.close()
            dataframe[column_name] = temp_dict[dataframe.iloc[0][column_name]]
            return dataframe[column_name]

        preprocess_row(row, 'Team', "team_dict")
        preprocess_row(row, 'Opponent Team', 'team_dict')
        preprocess_row(row, 'League', 'league_dict')


        def normalize(df, column, dictionary):
            df[column] = df.iloc[0][column] / dictionary[column]
            return df[column]
        
        cols_to_norm = ['S_own', 'EX_own', 'EN_own', 'F_own', 'I_own',
                    'R_own', 'S_opp', 'EX_opp', 'EN_opp', 'F_opp', 'I_opp','R_opp']

        cols_to_norm = ['S_own', 'EX_own', 'EN_own', 'F_own', 'I_own','R_own', 'S_opp', 'EX_opp', 'EN_opp', 'F_opp', 'I_opp','R_opp']

        row["Home Situation"] = 1

        for column in cols_to_norm:
            normalize(row, column, max_dict)

        return row
