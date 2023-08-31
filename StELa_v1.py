###################################################################################################
####################################### The StELa Algorithm #######################################
# Secondary sTructural Ensembles with machine LeArning
# For Characterizing Secondary Structure in Local Protein Regions

# Authored by Amanda C. Macke & Ruxandra I. Dima
# In collaboration with Maria S. Kelly, Jamie Rowley, Jacob E. Stump & Vageesha Hearth
# University of Cincinnati - Cincinnati, OH 
# Last Edited: 08/31/2023
###################################################################################################
# Python 3.8 & Anaconda (https://www.anaconda.com/)

# Load All the necessary Libraries used
import os
import csv
import sys
from collections import Counter

import numpy as np
from numpy import savetxt

import pandas as pd
from pandas.core.reshape.concat import concat
# For Visualizing
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import gaussian_kde

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as SCH
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn.metrics import silhouette_score

sys.setrecursionlimit(100000)

# You will want to change the following to best suit your system
# The Angle Headers can be found in the RAMA.xgv input files!
###################################################################################################
# The following function is used to load the Angle Files computed from GROMACS rama function and saved to .xvg files
# To simplify the python script, the GROMACS header was removed with the following bash line in a terminal
# sed -n '35, 490000p' rama-loop-a15-01-MonC.xvg > rama-loop-a15-01-nohead.xvg
###################################################################################################
# print('    ~~~ You can count the number of angles in the RAMA.xvg input and divide the number of lines by the number of angles ~~~')

# Alpha helices have to have 4 helical residues in a row
# Beta sheets have to have 2 beta residues in a row
# There has to be 2 residues inbetween independent secondary structures

###################################################################################################
print('\nDear User, ')
print('Hello, from the Dima Lab at the University of Cincinnati!\n')
###################################################################################################
# Some Parameters needed for your system
###################################################################################################
# Need to set the Random State for Consistency
STATE=701896

print('---------- Extracting System Info from Provided StELa_Input.csv ----------')

StELa_Input=pd.read_csv('./StELa_Input.csv', sep=',', header=None)
print('\n---------- Protein Region Information ----------')
SETUP=StELa_Input.iloc[0,1]
print('Provided System Name:',SETUP)

PROTEIN_HEAD=(StELa_Input.iloc[1,1]).split(', ')
# Add "frame_index" to be used as a column header
PROTEIN_HEAD_FRAME=['FRAME_INDEX']+PROTEIN_HEAD

TOTAL_ANGLES=int(StELa_Input.iloc[2,1])
# Setting Counters
NTERM=0
CTERM=TOTAL_ANGLES

print('\n---------- Trajectory Information ----------')
NO_TRAJ=int(StELa_Input.iloc[3,1])
print('Indicated No. Traj: {0}'.format(NO_TRAJ))

TRAJ_FRAMES=int(StELa_Input.iloc[4,1])
print('Indicated No. Sampled Frames: {0}'.format(TRAJ_FRAMES))

###################################################################################################
# Function used to read GROMACS rama.xvg files
###################################################################################################
def LOAD_ANGLES (PHI, PSI, LABEL, STRUCT):

	with open('./{0}.xvg'.format(STRUCT)) as f:
		for line in f:
			cols=line.split()
			
			if len (cols) == 3:
				PHI.append(float(cols[0]))
				PSI.append(float(cols[1]))
				LABEL.append(str(cols[2]))
	
	# Note: 19 Total Residues - the first and last residues because they don't have angles
	# Note: Frame1 - Lines 1 to 19 
	PHI=np.array(PHI)
	PSI=np.array(PSI)

print('---------- Importing Data from Trajectories ----------')
# Load Angles & Combine Alpha Helices & Loops
PHI_H15_L, PSI_H15_L, LABEL_H15_L=[],[],[]
for i in range(1, NO_TRAJ+1):
    LOAD_ANGLES (PHI_H15_L, PSI_H15_L, LABEL_H15_L, 'rama-{0}-{1}-nohead'.format(SETUP, i))

print('~ Angles Extracted from XVG ~')
x=np.array(PHI_H15_L)
y=np.array(PSI_H15_L)

##################################################################################################################################################
# Combine PHI & PSI angles into the same numpy array
angles=np.concatenate([x[:,None], y[:,None]], axis=1)
#print(angles.shape)
print('~ PHI PSI Angle Array Created ~')

print('\nDo you want the Ramachandran plotted according to KDE?(y/n)')
RAMA_PLOT=input()
if RAMA_PLOT == 'y' or RAMA_PLOT == 'Y':
    print('---------- Plotting Ramachandran ----------')
    # Ramachandran Plot (https://en.wikipedia.org/wiki/Ramachandran_plot)
    # Plotting the Ramachandran Plot with KDE/contour 
    xmin=x.min()
    xmax=x.max()
    ymin=y.min()
    ymax=y.max()

    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    values = np.vstack([x,y])
    kernel = gaussian_kde(values)

    z = np.reshape(kernel(positions).T, X.shape)

    kde_fig, ax = plt.subplots()
    plt.suptitle('Ramachandran Plot', fontsize=22)
    plt.title('{0}'.format(SETUP))
    plt.xlabel(r'$\phi$', fontsize=22, fontweight='bold')
    plt.ylabel(r'$\psi$', fontsize=22, fontweight='bold')

    my_im=ax.imshow(np.rot90(z), cmap='jet', extent=[xmin, xmax, ymin, ymax])

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    kde_fig.colorbar(my_im, cax=cax, orientation='vertical')

    plt.savefig('rama-{0}-kde.png'.format(SETUP),dpi=300)
    print('~ Ramachandran KDE Plotted ~')

##################################################################################################################################################
print('~ Create a Folder for Saving ~')

print('\nInput the Number of KMeans you want to check - Saved to N')
print('   *Currently, you can check from between 3 and 9 centroids')
N=int(input())
print('N={0}'.format(N))
# Make a New directory to keep files organized
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'Kmeans-{0}n'.format(N))
if not os.path.exists(final_directory):
   os.makedirs(final_directory)
os.chdir(final_directory)

##################################################################################################################################################
print('---------- Cluster Backbone Angles w/ Centroid Based Clustering ----------')
# Since we are testing number of KMeans...
# Calculate the KMeans Clustering
KM=KMeans(n_clusters=N,  init='k-means++', n_init=200, random_state=STATE)
KM.fit(angles)
Y_kmeans=KM.predict(angles)
KM_Labels=KM.labels_

# Extract Labels
PHI_PSI_KMEANS=np.concatenate([x[:,None], y[:,None], KM_Labels[:,None]], axis=1)
savetxt('kmeans_{0}n_{1}_labels.csv'.format(N, STATE), PHI_PSI_KMEANS, fmt='%d', delimiter=',')

# Extract Centroids
PHI_CENT=np.array(KM.cluster_centers_[:, 0])
PSI_CENT=np.array(KM.cluster_centers_[:, 1])
CENTROIDS=np.concatenate([PHI_CENT[:,None], PSI_CENT[:,None]], axis=1)
savetxt('kmeans_{0}n_{1}_centroids.csv'.format(N, STATE), CENTROIDS, fmt='%d', delimiter=',')

# Extract KMeans Labels 
rows=len(x)
columns=3

if N==3:
    KMEANS_n1, KMEANS_n2, KMEANS_n3 = [], [], []
elif N==4:
    KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4 = [], [], [], []
elif N==5:
    KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5 = [], [], [], [], []
elif N==6:
    KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6 = [], [], [], [], [], []
elif N==7:
    KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6, KMEANS_n7 = [], [], [], [], [], [], []
elif N==8:
    KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6, KMEANS_n7, KMEANS_n8 = [], [], [], [], [], [], [], []
elif N==9:
    KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6, KMEANS_n7, KMEANS_n8, KMEANS_n9 = [], [], [], [], [], [], [], [], []

for i in range(rows):
    if (PHI_PSI_KMEANS[i][2])==0.0:
        KMEANS_n1.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==1.0:
        KMEANS_n2.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==2.0:
        KMEANS_n3.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==3.0:
        KMEANS_n4.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==4.0:
        KMEANS_n5.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==5.0:
        KMEANS_n6.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==6.0:
        KMEANS_n7.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==7.0:
        KMEANS_n8.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    elif (PHI_PSI_KMEANS[i][2])==8.0:
        KMEANS_n9.append([PHI_PSI_KMEANS[i][0], PHI_PSI_KMEANS[i][1]])
    else:
        print('Cluster Error')

# Set Lists to Arrays and List of Labels
if N==3:
    KMEANS_n1=np.array(KMEANS_n1)
    KMEANS_n2=np.array(KMEANS_n2)
    KMEANS_n3=np.array(KMEANS_n3)
    NAME=[KMEANS_n1, KMEANS_n2, KMEANS_n3]
    CLUSTER_LABELS=['0', '1', '2']
elif N==4:
    KMEANS_n1=np.array(KMEANS_n1)
    KMEANS_n2=np.array(KMEANS_n2)
    KMEANS_n3=np.array(KMEANS_n3)
    KMEANS_n4=np.array(KMEANS_n4) 
    NAME=[KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4]
    CLUSTER_LABELS=['0', '1', '2', '3']
elif N==5:
    KMEANS_n1=np.array(KMEANS_n1)
    KMEANS_n2=np.array(KMEANS_n2)
    KMEANS_n3=np.array(KMEANS_n3)
    KMEANS_n4=np.array(KMEANS_n4) 
    KMEANS_n5=np.array(KMEANS_n5)
    NAME=[KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5]
    CLUSTER_LABELS=['0', '1', '2', '3', '4']
elif N==6:
    KMEANS_n1=np.array(KMEANS_n1)
    KMEANS_n2=np.array(KMEANS_n2)
    KMEANS_n3=np.array(KMEANS_n3)
    KMEANS_n4=np.array(KMEANS_n4) 
    KMEANS_n5=np.array(KMEANS_n5)
    KMEANS_n6=np.array(KMEANS_n6)
    NAME=[KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6]
    CLUSTER_LABELS=['0', '1', '2', '3', '4', '5']
elif N==7:
    KMEANS_n1=np.array(KMEANS_n1)
    KMEANS_n2=np.array(KMEANS_n2)
    KMEANS_n3=np.array(KMEANS_n3)
    KMEANS_n4=np.array(KMEANS_n4) 
    KMEANS_n5=np.array(KMEANS_n5)
    KMEANS_n6=np.array(KMEANS_n6)
    KMEANS_n7=np.array(KMEANS_n7)
    NAME=[KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6, KMEANS_n7]
    CLUSTER_LABELS=['0', '1', '2', '3', '4', '5', '6']
elif N==8:
    KMEANS_n1=np.array(KMEANS_n1)
    KMEANS_n2=np.array(KMEANS_n2)
    KMEANS_n3=np.array(KMEANS_n3)
    KMEANS_n4=np.array(KMEANS_n4) 
    KMEANS_n5=np.array(KMEANS_n5)
    KMEANS_n6=np.array(KMEANS_n6)
    KMEANS_n7=np.array(KMEANS_n7)
    KMEANS_n8=np.array(KMEANS_n8)
    NAME=[KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6, KMEANS_n7, KMEANS_n8]
    CLUSTER_LABELS=['0', '1', '2', '3', '4', '5', '6', '7']
elif N==9:
    KMEANS_n1=np.array(KMEANS_n1)
    KMEANS_n2=np.array(KMEANS_n2)
    KMEANS_n3=np.array(KMEANS_n3)
    KMEANS_n4=np.array(KMEANS_n4) 
    KMEANS_n5=np.array(KMEANS_n5)
    KMEANS_n6=np.array(KMEANS_n6)
    KMEANS_n7=np.array(KMEANS_n7)
    KMEANS_n8=np.array(KMEANS_n8)
    KMEANS_n9=np.array(KMEANS_n9)
    NAME=[KMEANS_n1, KMEANS_n2, KMEANS_n3, KMEANS_n4, KMEANS_n5, KMEANS_n6, KMEANS_n7, KMEANS_n8, KMEANS_n8]
    CLUSTER_LABELS=['0', '1', '2', '3', '4', '5', '6', '7', '8']

###################################################################################################
# Show within clusters sum of squares (WCSS) to Evaluate the KMeans
WCSS=[]
for i in range(1,20):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(angles)
    WCSS.append(kmeans.inertia_)
wcss_fig=plt.figure(figsize=(8,8))
plt.scatter(range(1,20), WCSS, color='red')
plt.plot(range(1,20), WCSS, color='black', alpha=0.6)
plt.xlabel('Number of Clusters', fontsize=16)
plt.ylabel('Within Cluser Sum of Squares', fontsize=16)
plt.grid(visible=bool, which='major', axis='both', color='black', linestyle='solid', alpha=0.05)
plt.savefig('WCSS-KM-{0}.png'.format(STATE),dpi=300)
print('~ WCSS Computed ~')

# Color the Ramachandran Plot with K-Means clusters
# Set Up Distinct Color Map
COLORS = ['red', 'darkorange', 'gold', 'green', 'turquoise', 'blue', 'limegreen', 'purple', 'palevioletred']  
n_bin = N  # No. of Clusters
cmap_name = 'my_list'

cm = LinearSegmentedColormap.from_list('my_list', COLORS, N=n_bin)

kmeans_fig=plt.figure(figsize=(8,8))
plt.scatter(angles[:, 0], angles[:, 1], c=Y_kmeans, s=25, cmap=cm)
for i in range(0,N):
    plt.text(KM.cluster_centers_[i, 0], KM.cluster_centers_[i, 1], CLUSTER_LABELS[i], color='black', fontweight='bold', fontsize=25)
    plt.text(((KM.cluster_centers_[i, 0])-40), ((KM.cluster_centers_[i, 1])-15), (round(KM.cluster_centers_[i, 0], 1)), color='black', fontweight='bold', fontsize=16)
    plt.text(((KM.cluster_centers_[i, 0])+15), ((KM.cluster_centers_[i, 1])-15), (round(KM.cluster_centers_[i, 1], 1)), color='black', fontweight='bold', fontsize=16)
plt.xlabel(r'$\phi$ Angles', fontsize=16)
plt.ylabel(r'$\psi$ Angles', fontsize=16)
plt.savefig('Cluster_Labels_{0}n_kmean++_{1}.png'.format(N, STATE),dpi=300)
print('~ Plotted the Angles with KMeans Cluster Labels & Angles ~')

print('\nNOTE: Check the Cluster_Labels.png to check if your Kmeans clusters make sense.')
###################################################################################################
print('\n---------- Identify the Secondary Structure Regions ----------')
print('StELa is looking for Alpha Helices and Beta Strands. This means the Kmeans chosen should reflect a single cluster ID for those two regions as described in the Lit. If the Alpha/Beta regions are split by the KMeans clusters, you should reduce the number of KMeans clusters.')

print('\nInput the Value of the Alpha Region - Saved to ALPHA')
ALPHA=float(input())
print('ALPHA={0}'.format(ALPHA))

print('\nInput the Value of the Beta Region - Saved to BETA')
BETA=float(input())
print('BETA={0}'.format(BETA))

print('\nInput the Value of either the Bridge Region between the Alpha/Beta regimes, or an alternative region with similar PSI character - Saved to BRIDGE')
BRIDGE=float(input())
print('BRIDGE={0}'.format(BRIDGE))
##################################################################################################################################################
print('\n---------- Adjust According to Biochemistry Rules ----------')
FRAME_CLUSTERS=[]
FRAME_CLUSTERS_NOSPLIT=[]
FRAME_CLUSTERS_ADJUST=[]

for frame in range(0, TRAJ_FRAMES):
    #print('--------------------- Frame No. : ', frame, ' ---------------------')
    OG_VECTOR=[]
    for i in range(NTERM,CTERM):
        K_ID=(PHI_PSI_KMEANS[i][2])
        OG_VECTOR.append(K_ID)
    #print(OG_VECTOR)
    FRAME_CLUSTERS.append(OG_VECTOR[:])
    VECTOR=OG_VECTOR[:]
    
    ##### ALPHA CORRECTION & BIAS #####
    # Need to account for the Biochemistry (For it to be a true helix, need 4 alpha bois in a row
    res=0
    for index in range(0,TOTAL_ANGLES):   
        if res+2==TOTAL_ANGLES:   
            break

        #print('Res #: ', res)
        #print('Vector Element: ', VECTOR[res])

        if res+2==(TOTAL_ANGLES-1):   

            #print('Res #: ', res+1)
            #print('Vector Element: ', VECTOR[res+1])
            #print('Res #: ', res+2)
            #print('Vector Element: ', VECTOR[res+2])

            if VECTOR[res+2]==ALPHA or VECTOR[res+1]==ALPHA or VECTOR[res]==ALPHA:
                if VECTOR[res+2]==ALPHA:   
                    if VECTOR[res+1]==ALPHA and VECTOR[res]==ALPHA and VECTOR[res-1]==ALPHA:
                        res=res+1
                        continue
                    else:
                        VECTOR[res+2]=BRIDGE
                        if VECTOR[res+1]==ALPHA:   
                            VECTOR[res+1]=BRIDGE
                        if VECTOR[res]==ALPHA and VECTOR[res-1]!=ALPHA:   
                            VECTOR[res]=BRIDGE
                        res=res+1
                        continue
                elif VECTOR[res+1]==ALPHA:   
                    if VECTOR[res]==ALPHA and VECTOR[res-1]==ALPHA and VECTOR[res-2]==ALPHA:
                        res=res+1
                        continue
                    if VECTOR[res+2]==ALPHA and VECTOR[res]==ALPHA and VECTOR[res-1]==ALPHA:
                        res=res+1
                        continue
                    else:
                        VECTOR[res+1]=BRIDGE
                        if VECTOR[res]==ALPHA and VECTOR[res-1]!=ALPHA:   
                            VECTOR[res]=BRIDGE
                        res=res+1
                        continue
                elif VECTOR[res]==ALPHA:   
                    if VECTOR[res-1]==ALPHA and VECTOR[res-2]==ALPHA and VECTOR[res-3]==ALPHA:
                        res=res+1
                        continue
                    elif VECTOR[res+1]==ALPHA and VECTOR[res-1]==ALPHA and VECTOR[res-2]==ALPHA:
                        res=res+1
                        continue
                    elif VECTOR[res+2]==ALPHA and VECTOR[res+1]==ALPHA and VECTOR[res-1]==ALPHA:
                        res=res+1
                        continue
                    else:
                        VECTOR[res]=BRIDGE
                        if VECTOR[res+1]==ALPHA:  
                            VECTOR[res+1]=BRIDGE
                        if VECTOR[res+2]==ALPHA:   
                            VECTOR[res+2]=BRIDGE
                        res=res+1
                        continue
            else:
                res=res+1
            continue 
        elif VECTOR[res]==ALPHA:
            if res==0:    
                if VECTOR[res+1]==ALPHA and VECTOR[res+2]==ALPHA and VECTOR[res+3]==ALPHA:
                    res=res+1
                    continue
                else:
                    VECTOR[res]=BRIDGE
                    if VECTOR[res+1]==ALPHA:
                        VECTOR[res+1]=BRIDGE
                    if VECTOR[res+2]==ALPHA:
                        VECTOR[res+2]=BRIDGE
                    if VECTOR[res+3]==ALPHA:
                        VECTOR[res+3]=BRIDGE

                    res=res+1
                    continue
            elif res != 0:
                if VECTOR[res+1]==ALPHA and VECTOR[res+2]==ALPHA and VECTOR[res+3]==ALPHA:
                    res=res+1
                    continue
                if VECTOR[res+2]==ALPHA and VECTOR[res+1]==ALPHA and VECTOR[res-1]==ALPHA:
                    res=res+1
                    continue
                if VECTOR[res+1]==ALPHA and VECTOR[res-1]==ALPHA and VECTOR[res-2]==ALPHA:
                    res=res+1
                    continue
                if VECTOR[res-1]==ALPHA and VECTOR[res-2]==ALPHA and VECTOR[res-3]==ALPHA:
                    res=res+1
                    continue
                else:
                    VECTOR[res]=BRIDGE
                    res=res+1
                    continue
        elif VECTOR[res]!=ALPHA:
            if res==(TOTAL_ANGLES-5):   
                if VECTOR[res+1]==ALPHA:    
                    if VECTOR[res-4]==ALPHA and VECTOR[res-3]==ALPHA and VECTOR[res-2]==ALPHA and VECTOR[res-1]==ALPHA:   
                        VECTOR[res+1]=BRIDGE
                        res=res+1
                        continue
                    else:
                        res=res+1
                        continue
                else: 
                    res=res+1
                    continue
            elif res==(TOTAL_ANGLES-4) or res==(TOTAL_ANGLES-3) or res==(TOTAL_ANGLES-2) or res==(TOTAL_ANGLES-1):
                res=res+1
                continue
            elif VECTOR[res-4]==ALPHA and VECTOR[res-3]==ALPHA and VECTOR[res-2]==ALPHA and VECTOR[res-1]==ALPHA and VECTOR[res+1]==ALPHA and VECTOR[res+2]==ALPHA and VECTOR[res+3]==ALPHA and VECTOR[res+4]==ALPHA:
                VECTOR[res+1]=BRIDGE
                res=res+1
                continue
            else:
                res=res+1
                continue
        else:
            res=res+1

    ##### BETA BIAS #####
    res=0
    for index in range(0,TOTAL_ANGLES):   
        if res+1==TOTAL_ANGLES:
            break

        if res+1==(TOTAL_ANGLES-1): 

            if VECTOR[res+1]==BETA or VECTOR[res]==BETA:
                if VECTOR[res+1]==BETA:   
                    if VECTOR[res]==BETA and VECTOR[res-1]==BETA:
                        res=res+1
                        continue
                    else:
                        VECTOR[res+1]=BRIDGE
                        if VECTOR[res]==BETA and VECTOR[res-1]!=BETA:
                            VECTOR[res]=BRIDGE
                        res=res+1
                        continue
                elif VECTOR[res]==BETA:   
                    if VECTOR[res-1]==BETA and VECTOR[res-2]==BETA:
                        res=res+1
                        continue
                    elif VECTOR[res+1]==BETA and VECTOR[res-1]==BETA:
                        res=res+1
                        continue
                    else:
                        VECTOR[res]=BRIDGE
                        if VECTOR[res+1]==BETA:
                            VECTOR[res+1]=BRIDGE
                        res=res+1
                        continue
            else:
                res=res+1
            continue 
        elif VECTOR[res]==BETA:
            if res==0:  
                if VECTOR[res+1]==BETA and VECTOR[res+2]==BETA:
                    res=res+1
                    continue
                else:
                    VECTOR[res]=BRIDGE
                    res=res+1
                    continue
            elif res != 0:
                if VECTOR[res+1]==BETA and VECTOR[res+2]==BETA:
                    res=res+1
                    continue
                if VECTOR[res+1]==BETA and VECTOR[res-1]==BETA:
                    res=res+1
                    continue
                if VECTOR[res-2]==BETA and VECTOR[res-1]==BETA:
                    res=res+1
                    continue
                else:
                    VECTOR[res]=BRIDGE
                    res=res+1
                    continue
        else:
            res=res+1
    #print(VECTOR)    
    VECTOR_NOSPLIT=VECTOR[:]
    
    # Bias the heirarchical clustering
    element=0
    for index in range(0,TOTAL_ANGLES):   
        # Change the ALPHA elements to 10.0
        if VECTOR_NOSPLIT[element]==ALPHA:
            VECTOR_NOSPLIT[element]=10.0
        # Change the BETA elements to 15.0
        if VECTOR_NOSPLIT[element]==BETA:
            VECTOR_NOSPLIT[element]=15.0
        element=element+1
    #print(VECTOR_NOSPLIT)
    # Save Vector to List for Exporting
    FRAME_CLUSTERS_ADJUST.append(VECTOR_NOSPLIT[:])
    NTERM=NTERM+TOTAL_ANGLES
    CTERM=CTERM+TOTAL_ANGLES
    
    print('### Progress {}% ###'.format(round((frame/TRAJ_FRAMES)*100,1)))

print('~ Get the Cluster Vector Per Frame ~')

# Save the Original Vectors to a .csv
frame_list=list(range(0,TRAJ_FRAMES))
DF1=pd.DataFrame(FRAME_CLUSTERS, columns=PROTEIN_HEAD)
DF1.insert(0, "FRAME_INDEX", frame_list)
DF1.to_csv('./OG-frame-vectors-{0}-{1}n.csv'.format(STATE, N), index=False)
print('~ Original Vectors are Saved to OG.csv Files ~')

# Save the Adjusted Vectors to a .csv 
DF2=pd.DataFrame(FRAME_CLUSTERS_ADJUST, columns=PROTEIN_HEAD)
DF2.insert(0, "FRAME_INDEX", frame_list)
DF2.to_csv('./ADJUSTED-frame-vectors-{0}-{1}n.csv'.format(STATE, N), index=False)
print('~ Adjusted Vectors are Saved to ADJUSTED.csv Files ~')

##################################################################################################################################################
print('\n---------- Evaluating the No. of Clusters for AHC ----------')
# Set the Agglomerative Heirarchical Clustering Method being used ('complete', 'average')
# 'single' linkage method = Minimum Distance
# 'complete' linkage method = Maximum Distance
# 'average' linkage method = Inter-Cluster Dissimilarity
METHOD='complete'

plt.rcParams['lines.linewidth'] = 3
# Plot the Dendrogram
dend_fig1=plt.figure(figsize=(18,12))
Z=(SCH.linkage(FRAME_CLUSTERS_ADJUST, method=METHOD))

dendrogram = SCH.dendrogram(Z, truncate_mode='lastp', p=40)   # color_threshold=40
# Uncomment the horizonal lines to add them if you need help looking at euclidean distances
#plt.hlines(25, 0, 20500, color='black', lw=2, linestyle='--')   

#plt.title('{0} Linkage Dendrogram'.format(METHOD), fontsize=22)
plt.ylabel('Euclidean Distance', fontsize=26, fontweight='bold')
plt.yticks(fontsize=20)
plt.xticks(fontsize=16)
plt.savefig('{0}-Dendrogram-{1}n-{2}.png'.format(METHOD, N, STATE),dpi=300)
print('~ Dendogram Plotted ~')

##################################################################################################################################################
plt.rcParams['lines.linewidth'] = 2
# Plotting the Calinski Harabasz Score Scores
pSF=[]
for number_clust in range(2, 30):
    cluster=AgglomerativeClustering(n_clusters=number_clust, affinity='euclidean', linkage=METHOD)
    y=cluster.fit_predict(FRAME_CLUSTERS_ADJUST)
    labels=cluster.labels_
    pSF.append(metrics.calinski_harabasz_score(FRAME_CLUSTERS_ADJUST, labels))

psf_fig=plt.figure(figsize=(12,8))
plt.plot(range(2,30), pSF, color='black', alpha=0.75)
plt.scatter(range(2,30), pSF, s=75, color='deeppink')
#plt.vlines(11, np.min(pSF), np.max(pSF), colors='black', linestyles='dashed' )
plt.xlabel('Number of Clusters', fontsize=22, fontweight='bold')
plt.xticks(fontsize=16)
plt.ylabel('Calinski Harabasz Score', fontsize=22, fontweight='bold')
plt.yticks(fontsize=16)
plt.savefig('CalinskiHarabaszScore-{0}.png'.format(METHOD),dpi=300)
print('~ pSF Scores Plotted ~')

# Plotting the Silhouette Scores 
SIL=[]
for number_clust in range(2,30):
    cluster=AgglomerativeClustering(n_clusters=number_clust, affinity='euclidean', linkage=METHOD)
    y=cluster.fit_predict(FRAME_CLUSTERS_ADJUST)
    labels=cluster.labels_
    SIL.append(metrics.silhouette_score(FRAME_CLUSTERS_ADJUST, labels, metric='euclidean'))

sil_fig=plt.figure(figsize=(12,8))
plt.plot(range(2,30), SIL, color='black', alpha=0.75)
plt.scatter(range(2,30), SIL, s=75, color='goldenrod')
#plt.vlines(11, np.min(SIL), np.max(SIL), colors='black', linestyles='dashed' )
plt.xlabel('Number of Clusters', fontsize=22, fontweight='bold')
plt.xticks(fontsize=16)
plt.ylabel('Silhouette Score', fontsize=22, fontweight='bold')
plt.yticks(fontsize=16)
plt.savefig('SilhouetteScores-{0}.png'.format(METHOD),dpi=300)
print('~ Silhouette Scores Plotted ~')

##################################################################################################################################################
print('---------- Clustering the Representative Vectors ----------')
print('Input the Number of Agglomerative Heirarchical Clusters - Saved to AHC_N')
AHC_N=int(input())
print('AHC_N={0}'.format(AHC_N))

# Set which cluster you are extracting
C=list(range(0,AHC_N))

clusters=AgglomerativeClustering(n_clusters=AHC_N, affinity='euclidean', linkage=METHOD)
AHC=clusters.fit_predict(FRAME_CLUSTERS_ADJUST)

##################################################################################################################################################
# Replot the dendrogram with the chosen number of clusters
plt.rcParams['lines.linewidth'] = 4
# Plot the Dendrogram
dend_fig1=plt.figure(figsize=(18,12))
Z=(SCH.linkage(FRAME_CLUSTERS_ADJUST, method=METHOD))

dendrogram = SCH.dendrogram(Z, truncate_mode='lastp', p=AHC_N)
# Uncomment the horizonal lines to add them if you need help looking at euclidean distances
#plt.hlines(21.0, 0, 20500, color='black')   

#plt.title('{0} Linkage Dendrogram'.format(METHOD), fontsize=22)
plt.ylabel('Euclidean Distance', fontsize=26, fontweight='bold')
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig('{0}-Dendrogram-{1}n-{2}.png'.format(METHOD, N, STATE),dpi=300)

##################################################################################################################################################
print('---------- Analyzing Individual Cluster Populations ----------')
ALL_MATCHING_FRAMES=[]
ALL_MATCHING_FRAMES_TXT=[]
# In case of rerunning the same script multiple times, remove the file if it exists
if os.path.exists('StELa_Clusters.txt'):
    os.remove('StELa_Clusters.txt')
CLUSTER_INFO=open(r'StELa_Clusters.txt', 'a')

for clus in range(0,len(C)):
    LIST_OF_MATCHING_FRAMES=[]
    CLUSTER=[]
    for frame in range(0,len(AHC)):
        if AHC[frame]==clus:
            CLUSTER.append(DF2.loc[frame])
    CLUSTER_pd=pd.DataFrame(CLUSTER, columns=PROTEIN_HEAD_FRAME)

    ### Analyzing Each Cluster ###
    COUNTING_LEN=len(CLUSTER_pd)
    # Adjusted Cluster Labels ALPHA = 10.0 & BETA = 15.0 for proper identification
    ValuesList=[float(i) for i in CLUSTER_LABELS]
    element=0
    for index in range(0, N):   
        if ValuesList[element]==ALPHA:
            ValuesList[element]=10.0
        if ValuesList[element]==BETA:
            ValuesList[element]=15.0
        element=element+1

    # Calculate the Probablity Table 
    for angle_counter in range(1,TOTAL_ANGLES+1):   # Starts at 1 because in the File, the first column (0) is for the Frame Numbers
        PROB_LIST=[]    # Have to make a new list for every residue
        for rama_label in range(0,len(ValuesList)):
            try:
                COUNTING_INSTANCES=(CLUSTER_pd['{0}'.format(PROTEIN_HEAD_FRAME[angle_counter])].value_counts()[ValuesList[rama_label]])

                PROBABILITY=round((COUNTING_INSTANCES/COUNTING_LEN),4)
                PROB_LIST.append(PROBABILITY)
            except:
                PROBABILITY=0.0
                PROB_LIST.append(PROBABILITY)
        if angle_counter==1:    # Make a DF to hold the List of Prob with the Residue Headers
            DF4=pd.DataFrame([PROB_LIST], columns=None, index=['{0}'.format(PROTEIN_HEAD_FRAME[angle_counter])])
            DF4=DF4.T
        else:   # Make a DF for hold the list of Prob with the Residue Headers & then append it to create the full Probability table
            DF5=pd.DataFrame([PROB_LIST], columns=None, index=['{0}'.format(PROTEIN_HEAD_FRAME[angle_counter])])
            DF5=DF5.T
            DF4['{0}'.format(PROTEIN_HEAD_FRAME[angle_counter])]=DF5

    VALUES_COLUMN=pd.DataFrame(ValuesList, columns=['FRAME_INDEX'])

    DF4.insert(loc=0, column='FRAME_INDEX', value=VALUES_COLUMN)

    # Identify the Most Probable Vector
    PROB_VECTOR=[]
    for angle_counter in range(1,TOTAL_ANGLES+1):
        COL=DF4['{0}'.format(PROTEIN_HEAD_FRAME[angle_counter])]
        COL_MAX=COL.idxmax()
        RAMA_ID=ValuesList[COL_MAX]
        PROB_VECTOR.append(RAMA_ID)

    # Prep to Add the most Probable Vector to the Data Frame
    PROB_VECTOR_DF=pd.DataFrame(PROB_VECTOR)
    PROB_VECTOR_ID=PROB_VECTOR_DF.T
    PROB_VECTOR_DF=PROB_VECTOR_DF.T
    PROB_VECTOR_DF.columns=PROTEIN_HEAD
    PROB_VECTOR_DF.insert(loc=0, column='FRAME_INDEX', value='PROB VECTOR')

    # Combine the Cluster Data with the Probability Table and the Most Probable Vector
    DF_LIST=[CLUSTER_pd, DF4, PROB_VECTOR_DF]
    CLUSTER_PROB_DF=pd.concat(DF_LIST, ignore_index=True)

    # Choose the Most Probably Vector, or a Manual Selection
    # Check each available frame to see if it matches the Most Probable Vector
    for frame in range(0, COUNTING_LEN):
        for angle_counter in range(1,TOTAL_ANGLES+1):
            if CLUSTER_PROB_DF.iloc[frame, angle_counter]!=PROB_VECTOR[angle_counter-1]:
                # Cannot use this Frame
                break
            if angle_counter == TOTAL_ANGLES:
                LIST_OF_MATCHING_FRAMES.append(CLUSTER_PROB_DF.iloc[frame, 0])   
    pd.DataFrame(CLUSTER_PROB_DF).to_csv('./CHECKcluster-vector-{0}-{1}ahc.csv'.format(STATE, AHC_N), index=False)

    print('\n---------- Cluster {0} ----------'. format(clus))
    POP=(round((len(CLUSTER)/TRAJ_FRAMES)*100, 4))
    print('Population: {0}%'.format(POP))
    print('\nThe Calculated Probability Table:')
    print(DF4)
    print('Most Probable Vector: ', PROB_VECTOR)
    print('\nThe Following Frame Indices Match the Probable Vector:')
    print(LIST_OF_MATCHING_FRAMES)
    # Save the Matching Frames for the Indicated Vector to the Text File.
    ALL_MATCHING_FRAMES.append(LIST_OF_MATCHING_FRAMES)    
    CLUSTER_INFO.write('\n---------- Cluster {0} ---------- \n'.format(clus))
    CLUSTER_INFO.write('Population: {0} \n'.format(POP))
    CLUSTER_INFO.write('Probable Vector: {0} \n'.format(PROB_VECTOR))
    CLUSTER_INFO.write('Possible Frames for the Representative Structure:\n {0} \n'.format(LIST_OF_MATCHING_FRAMES))
    VECTOR_CHECK=input('\nWould you like to use the Most Probable Vector? (y/n): ')

    while VECTOR_CHECK == 'n' or VECTOR_CHECK == 'N':   # Will Repeat Until Vector is Suitable
        CHOSEN_VECTOR=[]
        print('You will need to manually type the vector you would like to check the cluster with.')
        print('The Probability Table for this Cluster can be found in CHECKcluster-vector-...-.csv\n')

        s = input(f"Enter alternative vector to check [a, b, ... ]: ")
        s = s[1:-1]
        CHOSEN_VECTOR = [float(entry[:-1]) for entry in s.split()[:-1]]
        CHOSEN_VECTOR.append(float(s.split()[-1]))
        print(CHOSEN_VECTOR)
        
        for frame in range(0, COUNTING_LEN):
            for angle_counter in range(1,TOTAL_ANGLES+1):
                if CLUSTER_PROB_DF.iloc[frame, angle_counter]!=CHOSEN_VECTOR[angle_counter-1]:
                    break
                if angle_counter == TOTAL_ANGLES:
                    LIST_OF_MATCHING_FRAMES.append(CLUSTER_PROB_DF.iloc[frame, 0])   
        print('The Following Frame Indices Match the Indicated Vector for Cluster {0}'.format(clus))
        print(LIST_OF_MATCHING_FRAMES)
        # Check if this Vector is present
        VECTOR_CHECK=input('\nWould you like to use the Listed Vector? (y/n): ')
        if VECTOR_CHECK == 'y' or VECTOR_CHECK == 'Y':
            CHOSEN_VECTOR_DF=pd.DataFrame(CHOSEN_VECTOR)
            CHOSEN_VECTOR_DF=CHOSEN_VECTOR_DF.T
            CHOSEN_VECTOR_DF.columns=PROTEIN_HEAD
            CHOSEN_VECTOR_DF.insert(loc=0, column='FRAME_INDEX', value='CHOSEN VECTOR')
            DF_LIST=[CLUSTER_PROB_DF, CHOSEN_VECTOR_DF]
            CLUSTER_PROB_DF=pd.concat(DF_LIST, ignore_index=True)
            CLUSTER_INFO.write('PROB VECTOR cannot be used for CLUSTER {0} \n'.format(clus))
            CLUSTER_INFO.write('Vector Used for this Cluster: \n {0} \n'.format(CHOSEN_VECTOR))
            CLUSTER_INFO.write('Possible Frames for the Representative Structure:\n {0} \n'.format(LIST_OF_MATCHING_FRAMES))

    # Save the Cluster Information (the frames, the prob table, the prob vector, & the chosen vector if necessary) to a .csv File
    pd.DataFrame(CLUSTER_PROB_DF).to_csv('./cluster{0}-vectors-{1}-{2}ahc.csv'.format(clus, STATE, AHC_N), index=False)
    print('Next Cluster')

print('\n~ Each Cluster is Saved to an Individual .csv File ~')
CLUSTER_INFO.close()
print('\n\nAll Vectors of Interest and Frame info is saved to StELa_Clusters.txt')
# Save the Adjusted Vectors & AHC Labels to a Master List .csv 

SINGLE_TRAJ=int((TRAJ_FRAMES/NO_TRAJ))
single_traj_frames=list(range(0,SINGLE_TRAJ))
single_traj_frames_list=single_traj_frames*NO_TRAJ

##################################################################################################################################################
frame_list=list(range(0,TRAJ_FRAMES))
DF3=pd.DataFrame(FRAME_CLUSTERS_ADJUST, columns=PROTEIN_HEAD)
DF3.insert(0, "AHC_LABEL", AHC)
DF3.insert(0, "FRAME_INDEX", frame_list)
DF3.insert(0, "TRAJ_FRAME", single_traj_frames_list)
DF3.to_csv('./master-list-vectors-{0}-{1}n.csv'.format(STATE, AHC_N), index=False)
##################################################################################################################################################
print('\n~ Done :) ~')
print('\nThank you for using StELa for your secondary structure needs!')
print('With Love, \nStELa')


