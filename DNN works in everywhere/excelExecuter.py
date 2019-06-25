import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


'''
example:

pd.read_excel(io, sheetname=0,header=0,skiprows=None,index_col=None,names=None,
                arse_cols=None,date_parser=None,na_values=None,thousands=None, 
                convert_float=True,has_index_names=None,converters=None,dtype=None,
                true_values=None,false_values=None,engine=None,squeeze=False,**kwds)

'''



'''
inputFile = "C:/Users/guosu/Desktop/python/Nerve network/FX/New folder/"+inputFileName+".xlsx"
outputFile = "C:/Users/guosu/Desktop/python/Nerve network/FX/New folder/"+inputFileName+" copy.xlsx"
'''

def data_process():
    df = pd.read_csv("browsing.csv")   # set the targeted file location

    factor = df.drop(df.columns[len(df.columns)-1],axis=1).values # get inputs
    target = df[df.columns[len(df.columns)-1]].values # get labels

    #print("data is ",data, "factor is ",factor, "target is ",target)
    return factor,target
