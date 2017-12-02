# Non-proportional venn diagram for all correctly assigned
from matplotlib import pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn3_circles, venn3_unweighted
plt.figure(figsize=(10, 10))
vd = venn3_unweighted(subsets=(61, 37, 39, 6, 41, 31, 1158))
for text in vd.subset_labels:
    text.set_weight('bold')
for text in vd.subset_labels:
    text.set_color('black')
for text in vd.subset_labels: 
    text.set_fontsize(20)
vd.get_label_by_id('A').set_text('Support Vector Machines')
vd.get_label_by_id('A').set_fontsize(20)
vd.get_label_by_id('A').set_weight('bold')
vd.get_label_by_id('B').set_text('Random Forest')
vd.get_label_by_id('B').set_fontsize(20)
vd.get_label_by_id('B').set_weight('bold')
vd.get_label_by_id('C').set_text('Bayes Net')
vd.get_label_by_id('C').set_fontsize(20)
vd.get_label_by_id('C').set_weight('bold')
vd.get_patch_by_id('100').set_alpha(1.0)
vd.get_patch_by_id('100').set_color('#594046')
vd.get_patch_by_id('110').set_alpha(1.0)
vd.get_patch_by_id('110').set_color('#DE646B')
vd.get_patch_by_id('010').set_alpha(1.0)
vd.get_patch_by_id('010').set_color('#D74A50')  
vd.get_patch_by_id('101').set_alpha(1.0)
vd.get_patch_by_id('101').set_color('#FCB064')
vd.get_patch_by_id('111').set_alpha(1.0)
vd.get_patch_by_id('111').set_color('#F7DADC')
vd.get_patch_by_id('011').set_alpha(1.0)    
vd.get_patch_by_id('011').set_color('#DD646B') 
vd.get_patch_by_id('001').set_alpha(1.0)
vd.get_patch_by_id('001').set_color('#FAA451')         
plt.text(-1.1,-0.75, 'Number of correctly assigned observations: 1363', fontsize=14, fontweight='bold')
plt.text(-1.1,-0.8, 'Total number of observations: 1470', fontsize=14, fontweight='bold')
plt.show()
