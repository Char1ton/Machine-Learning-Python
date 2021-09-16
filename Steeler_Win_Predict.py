import pandas as pd
import numpy
from pandas import set_option
from matplotlib import pyplot
from numpy import set_printoptions
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit,cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier



#Prepare Data----------------------------------------------------------------------
def PrepareData():

    filename = 'Steelers.csv'
    names = ['week', 'day', 'date', 'time', 'boxscore','W/L', 'NaN', 
         'record','at', 'Opp', 'SteelScore', 'OppScore','1stD',
         'TotYd','PassYd','RushYd','OppTurn','Opp1stD','OppTotYd','OppPassYd'
         ,'OppRushYd','Turn','ExOff','ExDef','ExSpt']
    data = pd.read_csv(filename, names=names).drop(columns=['day','date','time',
                                                        'boxscore','at','week'
                                                        ,'NaN','record','Opp',
                                                        'ExOff','ExDef','ExSpt','OppScore','SteelScore'])
    newnames=['W/L','1stD',
         'TotYd','PassYd','RushYd','OppTurn','Opp1stD','OppTotYd','OppPassYd'
         ,'OppRushYd','Turn']

    dKey={'W':1,'L':0}
    data['W/L']=data['W/L'].map(dKey)
    data=data.fillna(0)
    
    return data, newnames
#-------------------------------------------------------------------------------------------

#Data Visualization-------------------------------------------------

def DataVisualization(data,newnames):
    
    correlations = data.corr()
    
    data.plot(kind='density', subplots=True, layout=(3,5), sharex=False)

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0,11,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(newnames)
    ax.set_yticklabels(newnames)
    pyplot.show()

#-------------------------------------------------------------

#ML Test-------------------------------------------------

def MachineLearn(data):
    array=data.values
    X=array[:,1:11]
    Y=array[:,0]

    test_size=0.25
    seed = 10

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=seed)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)
    
    return model

#-----------------------------------------------------

#Main-----------------------------------------

def Main():

    data, newnames =PrepareData()

    model=MachineLearn(data)
    
    new_input=pd.read_csv('SteelerSingleGamePredict.csv',names=['1stD',
         'TotYd','PassYd','RushYd','OppTurn','Opp1stD','OppTotYd','OppPassYd'
         ,'OppRushYd','Turn'])
    
    new_output=model.predict(new_input)

    print('1 is a predicted Win, 0 is a predicted Loss:',new_output)
    
if __name__=="__main__":
    Main()



