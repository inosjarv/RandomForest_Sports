from preprocessing import *
from random_forest import *
from row_preprocess import *
import sys
if __name__ == "__main__":
    predict = sys.argv[1:]
    if (len(sys.argv)==1) and (not predict):
        csv_file = DataProcessor(filename='data.csv')
        dataframe = csv_file.run()
    else:
        csv_file = RowProcessor(filename="test.csv")
        dataframe = csv_file.run()

    nn = Randomforest(dataframe)
    nn.run(predict)