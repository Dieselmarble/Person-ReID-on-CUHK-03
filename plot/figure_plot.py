import numpy as np
import scipy.io as sio
from scipy.io import savemat, loadmat
#from cmc_plot import cmc_figure
from map_plot import map_figure

#------------------------------------rank-10 CMC plot-------------------------------------
#euclid_rankk=np.load('euclid_rankk.npy')
#corre_rankk=np.load('corre_rankk.npy')
#cheby_rankk=np.load('cheby_rankk.npy')
#manhat_rankk=np.load('manhat_rankk.npy')
#minkow_rankk=np.load('minkow_rankk.npy')
#cosine_rankk=np.load('cosine_rankk.npy')
#sqeuclid_rankk=np.load('sqeuclid_rankk.npy')
#
#rankk_dict={'euclid':euclid_rankk.T,'correl':corre_rankk.T,'cheby':cheby_rankk.T,'manhattan':manhat_rankk.T,'cosine':cosine_rankk.T,'minkow':minkow_rankk.T,'sq.euclid':sqeuclid_rankk.T}
#sio.savemat('rankk.mat',rankk_dict)
#mat = loadmat('rankk.mat')
#keys = ['euclid', "correl", 'cheby', 'manhattan', 'cosine', 'minkow', "sq. euclid"]
#methods = {"euclid": mat['euclid'],
#           "correl": mat['correl'],
#           "cheby": mat['cheby'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkow": mat['minkow'],
#           "sq. euclid" : mat['sq.euclid']}
#cmc_figure(methods, keys=keys)



#----------------------------------rank-100 CMC plot--------------------------------------
#euclid_rankk_100=np.load('euclid_rankk_100.npy')
#corre_rankk_100=np.load('corre_rankk_100.npy')
#cheby_rankk_100=np.load('cheby_rankk_100.npy')
#manhat_rankk_100=np.load('manhat_rankk_100.npy')
#minkow_rankk_100=np.load('minkow_rankk_100.npy')
#cosine_rankk_100=np.load('cosine_rankk_100.npy')
#sqeuclid_rankk_100=np.load('sqeuclid_rankk_100.npy')
#
#rankk_dict_100={'euclid':euclid_rankk_100.T,'correl':corre_rankk_100.T,'cheby':cheby_rankk_100.T,'manhattan':manhat_rankk_100.T,'cosine':cosine_rankk_100.T,'minkow':minkow_rankk_100.T,'sq.euclid':sqeuclid_rankk_100.T}
#sio.savemat('rankk_100.mat',rankk_dict_100)
#mat = loadmat('rankk_100.mat')
#keys = ['euclidean', "correlation", 'chebyshev', 'manhattan', 'cosine', 'minkowski', "sq. euclidean"]
#methods = {"euclidean": mat['euclid'],
#           "correlation": mat['correl'],
#           "chebyshev": mat['cheby'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkowski": mat['minkow'],
#           "sq. euclidean" : mat['sq.euclid']}
#cmc_figure(methods, keys=keys)



# -------------------------------rankk mean CMC plot--------------------------------------
#euclid_mean_cmc=np.load('euclid_mean_cmc.npy')
#corre_mean_cmc=np.load('corre_mean_cmc.npy')
#cheby_mean_cmc=np.load('cheby_mean_cmc.npy')
#manhat_mean_cmc=np.load('manhat_mean_cmc.npy')
#minkow_mean_cmc=np.load('minkow_mean_cmc.npy')
#cosine_mean_cmc=np.load('cosine_mean_cmc.npy')
#sqeuclid_mean_cmc=np.load('sqeuclid_mean_cmc.npy')
#rankk_dict_mean_cmc={'euclidean':euclid_mean_cmc.T,'correlation':corre_mean_cmc.T,'chebyshev':cheby_mean_cmc.T,'manhattan':manhat_mean_cmc.T,'minkowski':minkow_mean_cmc.T,'cosine':cosine_mean_cmc.T,'sq.euclid':sqeuclid_mean_cmc.T}
#sio.savemat('rankk_mean_cmc.mat',rankk_dict_mean_cmc)
#mat = loadmat('rankk_mean_cmc.mat')
#keys = ['euclidean', "correlation", 'chebyshev', 'manhattan', 'cosine', 'minkowski', "sq. euclidean"]
#methods = {"euclidean": mat['euclidean'],
#           "correlation": mat['correlation'],
#           "chebyshev": mat['chebyshev'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkowski": mat['minkowski'],
#           "sq. euclidean" : mat['sq.euclid']}
#cmc_figure(methods, keys=keys)



# -------------------------------rankk mean CMC plot--------------------------------------
#euclid_mean_cmc_100=np.load('euclid_mean_cmc_100.npy')
#corre_mean_cmc_100=np.load('corre_mean_cmc_100.npy')
#cheby_mean_cmc_100=np.load('cheby_mean_cmc_100.npy')
#manhat_mean_cmc_100=np.load('manhat_mean_cmc_100.npy')
#minkow_mean_cmc_100=np.load('minkow_mean_cmc_100.npy')
#cosine_mean_cmc_100=np.load('cosine_mean_cmc_100.npy')
#sqeuclid_mean_cmc_100=np.load('sqeuclid_mean_cmc_100.npy')
#euclid_mean_cmc_100 = np.reshape(euclid_mean_cmc_100,(100,1))
#corre_mean_cmc_100 = np.reshape(corre_mean_cmc_100,(100,1))
#cheby_mean_cmc_100 = np.reshape(cheby_mean_cmc_100,(100,1))
#manhat_mean_cmc_100 = np.reshape(manhat_mean_cmc_100,(100,1))
#minkow_mean_cmc_100 = np.reshape(minkow_mean_cmc_100,(100,1))
#cosine_mean_cmc_100 = np.reshape(cosine_mean_cmc_100,(100,1))
#sqeuclid_mean_cmc_100 = np.reshape(sqeuclid_mean_cmc_100,(100,1))
#rankk_dict_mean_cmc_100={'euclidean':euclid_mean_cmc_100,'correlation':cosine_mean_cmc_100,'chebyshev':cheby_mean_cmc_100,'manhattan':manhat_mean_cmc_100,'minkowski':minkow_mean_cmc_100,'cosine':cosine_mean_cmc_100,'sq.euclid':sqeuclid_mean_cmc_100}
#
#sio.savemat('rankk_mean_cmc_100.mat',rankk_dict_mean_cmc_100)
#mat = loadmat('rankk_mean_cmc_100.mat')
#keys = ['euclidean', "correlation", 'chebyshev', 'manhattan', 'cosine', 'minkowski', "sq. euclidean"]
#methods = {"euclidean": mat['euclidean'],
#           "correlation": mat['correlation'],
#           "chebyshev": mat['chebyshev'],
#           "manhattan": mat['manhattan'],
#           "cosine": mat['cosine'],
#           "minkowski": mat['minkowski'],
#           "sq. euclidean" : mat['sq.euclid']}
#cmc_figure(methods, keys=keys)



# --------------------------------------mAP plot---------------------------------------
#euclid_11map=np.load('euclid_11map.npy')
#corre_11map=np.load('corre_11map.npy')
#cheby_11map=np.load('cheby_11map.npy')
#manhat_11map=np.load('manhat_11map.npy')
#minkow_11map=np.load('minkow_11map.npy')
lmnn_11map=np.load('lmnn_11map.npy')
cosine_11map=np.load('cosine_11map.npy')
#sqeuclid_11map=np.load('sqeuclid_11map.npy')
#original_map=np.load('original_map.npy')

lmnn_11map = np.reshape(lmnn_11map,(11,1))
cosine_11map = np.reshape(cosine_11map,(11,1))
mmc_11map = np.array([0.58498,0.5318,0.483,0.4343,0.3768,0.3318,0.2877,0.2588,0.2142,0.1964,0.18158])

rankk_dict_11map={'cosine':cosine_11map,'lmnn':lmnn_11map,'mmc': mmc_11map}
sio.savemat('11map.mat',rankk_dict_11map)
mat = loadmat('11map.mat')
keys = ['cosine', 'lmnn', 'mmc']
methods = {'lmnn': mat['lmnn'],
           'mmc': mat['mmc'],
           "cosine": mat['cosine'],}

map_figure(methods, original_map, keys=keys)