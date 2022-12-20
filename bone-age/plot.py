import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab

central_hosting_test = [31.219476897681016  ]
central_hosting_male = [32.42201222243704 ]
central_hosting_female = [ 29.571103336663334 ]


round_homo_test = [64.69494778937295 ,31.874211248074467,31.529237414460802 ,30.435360648415305 ]
round_homo_male = [75.7765946818012 ,33.983675346865674 , 31.560727180841145 ,30.889992136812005 ]
round_homo_female = [57.58960940234962 ,29.04946909279659 ,31.487070083618164 ,29.82657250590708 ]
round_hetero_test= [59.51238164034757 , 34.340500890183506 ,29.62976868088181 ,30.492829934384957 ]
round_hetero_male = [67.31858489237118 ,37.52547248545634 ,29.676325343709134 ,33.431128767938574 ]
round_hetero_female = [49.05924740056882 ,30.0755676620308 ,29.567425563417633 ,26.55821138140799 ]
site_hetero_test= [28.759410370772827 ]
site_hetero_male = [28.71065257649565 ]
site_hetero_female = [28.82470098035089]
round_binary_test= [ 68.8833399967128 ,32.12705085435722 ,30.69453531401163 , 29.84692602602797 ]
round_binary_male = [76.64376256394284 , 36.14001879057659 ,28.345196981798427 ,32.5859261623268 ]
round_binary_female = [58.491509777376024 ,26.753363905281855 ,33.84048836806725 ,26.17918446420253]
site_binary_test= [31.83919584897578 ]
site_binary_male = [33.370665288278474  ]
site_binary_female = [29.788435048070447 ]


plt.plot([0,1,2,3,4], [126.5458146494216] + round_binary_test ,label='Overall Error')
plt.plot([0,1,2,3,4], [134.11142419767864] + round_binary_male,label='Male Error',color='mediumseagreen')
plt.plot([0,1,2,3,4], [116.4137807723199] + round_binary_female,label='Female Error',color='salmon')



plt.title('Round Dynamic Class Incremental', fontdict = {'fontsize':20})
plt.ylabel('Performance', fontdict = {'fontsize': 18})
plt.xlabel('Round', fontdict = {'fontsize': 18})
plt.legend(loc="upper left")
plt.tick_params(labelsize=14)



plt.xticks(np.arange(0, 5, 1.0))

# plt.savefig('round_binary2.svg',dpi =300,saveformat = 'svg')
plt.show()