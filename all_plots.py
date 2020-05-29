import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def plot_cost(cost_vector, cost_func, xticks, scale):
    ''' this function plots the data in 'cost_vector' in the form of box plots. The vectors in the 
    'cost_vector' can be of different length'''
      
    fig, ax = plt.subplots()
    bp = ax.boxplot(np.array(cost_vector), notch=0, sym='', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title(f'{scale} {cost_func}', fontsize = 14)
    ax.set_xlabel('dataset', fontsize = 14)
    ax.set_ylabel('values', fontsize = 14)

    # Setting xlimit, ylimit and xticklabels
    #ax.set_xlim(0.5, len(cost_func)+0.5)
    #ax.set_ylim(min(np.concatenate(np.array(cost_vector))), max(np.concatenate(np.array(cost_vector))))
    ax.set_xticklabels(xticks, rotation=45, fontsize=8)

    # Now fill the boxes with desired colors
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(cost_vector)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax.plot(np.average(med.get_xdata()), np.average(cost_vector[i]), color='w', marker='*', markeredgecolor='k')

    plt.show() # will show the figure
