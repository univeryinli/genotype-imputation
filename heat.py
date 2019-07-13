import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

region = ['Azerbaijan', 'Bahamas', 'Bangladesh', 'Belize', 'Bhutan',
          'Cambodia', 'Cameroon', 'Cape Verde', 'Chile', 'China']  # 10个

kind = ['Afforestation & reforestation', 'Biofuels', 'Biogas', 'Biomass', 'Cement']  # 5个

np.random.seed(20180316)
arr_region = np.random.choice(region, size=(200,))
list_region = list(arr_region)

arr_kind = np.random.choice(kind, size=(200,))
list_kind = list(arr_kind)

values = np.random.randint(100, 200, 200)
list_values = list(values)

df = pd.DataFrame({'region': list_region, 'kind': list_kind, 'values': list_values})

pt = df.pivot_table(index='kind', columns='region', values='values', aggfunc=np.sum)  # 数据透视表

print(pt)

# index是行，columns是列，values是表中展示的数据，aggfunc是表中展示每组数据使用的运算
f, (ax1, ax2) = plt.subplots(figsize=(8, 8), ncols=2)

# cmap用cubehelix map颜色
cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
sns.heatmap(pt, linewidths=0.05, ax=ax1, cmap=cmap, square=True)
ax1.set_title('cubehelix map')
ax1.set_xlabel('')
# ax1.set_xticklabels([]) #设置x轴图例为空值
ax1.set_ylabel('kind')

# cmap用matplotlib colormap
sns.heatmap(pt, linewidths=0.05, ax=ax2, vmax=900, vmin=0, cmap='rainbow', square=True)
# rainbow为 matplotlib 的colormap名称
ax2.set_title('matplotlib colormap')
ax2.set_xlabel('region')
ax2.set_yticklabels([])
ax2.set_ylabel('kind')

plt.show()
