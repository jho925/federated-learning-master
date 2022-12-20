import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab

accuracy_nothing = [0,0.513,0.754,0.464,0.794]
auc_nothing = [0,0.721,0.729,0.751 ,0.760]

plt.plot([0,1,2,3,4], accuracy_nothing,label='Accuracy')
plt.plot([0,1,2,3,4], auc_nothing,label='AUC')

plt.title('Central Hosting', fontdict = {'fontsize':20})
plt.ylabel('Performance', fontdict = {'fontsize': 18})
plt.xlabel('Round', fontdict = {'fontsize': 18})
plt.legend(loc="upper left")
plt.tick_params(labelsize=14)



plt.xticks(np.arange(0, 2, 1.0))
plt.tight_layout()
plt.savefig('nothing_rounds.svg',dpi =300,saveformat = 'svg')
plt.show()