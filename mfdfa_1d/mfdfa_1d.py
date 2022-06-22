"""Main module."""
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio

dem_ds = rio.open(r'E:\IITGN_postdoc\SRIP_2022\data\output_SRTMGL1.tif')
dem_data = dem_ds.read()[0]
plt.figure()
plt.imshow(dem_data)

# data = r'E:\IITGN_postdoc\SRIP_2022\data\IU.CASY.10.BHZ.M.2022.141.072018.SAC.xy'

# data = np.loadtxt(data, dtype=float)

data = dem_data[750, :]

# E = data[:, 1]
# X = data[:, 0]
E = data
X=np.arange(1, len(E) + 1)
mean_E = np.mean(E)
# Y = E - mean_E
Y = np.cumsum(E-np.mean(E))       # cumulative sum of "mean removed"

fig, ax = plt.subplots()
ax.plot(X, E, 'b')
ax1 = ax.twinx()
ax1.plot(X, Y, 'r')
ax.set_ylabel('E')
ax1.set_ylabel('Y')
plt.tight_layout()

# %%
import pandas as pd
from scipy.stats import linregress

N = len(E) # Length of the sequential series

# Definig scale

i = int(N / 4) # Maximum size for scale (Dutta, 2017)
Scale = [i]
while i > 30: # At least 30 points needed to caculate the variance
    i = int(0.9 * i)
    Scale.append(i)

list_q = np.round(np.arange(-3, 3.1, 0.1), 1)

df = pd.DataFrame(columns = ['s'])
df['s'] = Scale

df_hq = pd.DataFrame(columns = ['q', 'h(q)', 'r2'])

# q = -3
# list_q = [0]


indx1 = 0
for q in list_q:
    list_FqS = []
    NS_list = []
    NS_new_list = []
    k = 0
    for s in Scale:
        Ns = int(np.floor(N/s))
        NS_new = 0
        FqS = 0
        
        # ar = np.zeros((Ns, 1))
        for v in range(Ns):
            y = Y[v * s : (v + 1) * s]
            x = X[v * s : (v + 1) * s]
            yT = np.polyval(np.polyfit(x, y, 1), x)
            F2S = 0
            for i in range(s):
                F2S += (1 / s) * ((y[i] - yT[i]) ** 2)
            
            # ar[v, 0] = F2S
            if (F2S > 1e-4) & (q != 0):
                FqS += F2S ** (q / 2)
                NS_new += 1
            elif (F2S > 1e-4) & (q == 0):
                FqS += np.log(F2S)
                NS_new += 1
            
        d_FqS_51 = []
        for v in range(Ns):
            y = Y[N - (v + 1) * s : N - v * s]
            x = X[N - (v + 1) * s : N - v * s]
            yT = np.polyval(np.polyfit(x, y, 1), x)
            F2S = 0
            for i in range(s):
                F2S += (1 / s) * ((y[i] - yT[i]) ** 2)
            
            if (F2S > 1e-4) & (q != 0):
                FqS += F2S ** (q / 2)
                NS_new += 1
            elif (F2S > 1e-4) & (q == 0):
                FqS += np.log(F2S)
                NS_new += 1
                
            d_FqS_51.append(FqS)
            
        NS_list.append(2 * Ns)
        NS_new_list.append(NS_new)
        
        if q != 0:
            FqS = ((1 / (2 * Ns)) * FqS) ** (1 / q)
            # FqS = ((1 / NS_new) * FqS) ** (1 / q)
        else:
            FqS = np.exp((1 / (4 * Ns)) * FqS)
        
        list_FqS.append(FqS)
        
        
        df.loc[k, 'q=' + '%.1f' %q] = FqS
        k += 1
        
    x = Scale
    y = list_FqS
    res = linregress(np.log(x), np.log(y))
    
    df_hq.loc[indx1, 'q'] = q
    df_hq.loc[indx1, 'h(q)'] = res.slope
    df_hq.loc[indx1, 'r2'] = res.rvalue ** 2
    
    indx1 += 1
    
    print('%.1f' %q)

plt.figure()
select_list = [-3.0, 0.0, 3.0]
import seaborn as sns
c_list = palette = sns.color_palette(None, len(select_list))
for i in range(len(select_list)):
    plt.scatter(df['s'], df['q=' + str(select_list[i])], 
                marker = '*', s = 50,
                facecolors = 'none', edgecolors = c_list[i], 
                lw = 0.5, label = 'q=' + str(select_list[i]))
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.figure()
plt.scatter(df_hq['q'], df_hq['h(q)'], marker = 'o', s = 10, 
            facecolors = 'none', edgecolors = 'k', lw = 0.5)

# Only valid scenario is where the sprectrum is montonically decreasing
if df_hq['h(q)'].is_monotonic_decreasing:
    # Calculate alpha and f-alpha spectrum
    # alpha = h(q) + qh'(q)
    alpha = df_hq['h(q)'] + (df_hq['q'] * np.gradient(df_hq['h(q)'], df_hq['q']))
    # f(alpha) = q[alpha - h(q)] + 1
    f_alpha = df_hq['q'] * (alpha - df_hq['h(q)']) + 1
    
    assert np.max(f_alpha) == 1, "f_alpha can't exceed 1. Please check."
    
    alpha_0 = alpha[np.argmax(f_alpha)]
    x_alpha = (alpha - alpha_0).astype('float')
    f_alpha = f_alpha.astype('float')
    
    # calculate the multifractal width by fitting a 2nd order polynominal to the f-alpha spectrum and finding the roots of the polynomial
    res = np.polyfit(x_alpha, f_alpha, 2)
    roots = np.roots(res)
    mf_width = roots[1] - roots[0]
    
    y_alpha = np.polyval(res, x_alpha)
    plt.figure()
    plt.scatter(alpha, f_alpha, marker = '^', s = 10, 
                facecolors = 'none', edgecolors = 'b', lw = 0.5)
    plt.plot(x_alpha + alpha_0, y_alpha, 'k')
    
    
    df_alpha = pd.DataFrame(columns = ['q', 'h(q)', 'alpha', 'f_alpha'])
    df_alpha['q'] = df_hq['q']
    df_alpha['h(q)'] = df_hq['h(q)']
    df_alpha['alpha'] = alpha
    df_alpha['f_alpha'] = f_alpha
    
    df_alpha.to_excel(r'E:\IITGN_postdoc\SRIP_2022\data\transect_500.xlsx')
