import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import os



def show_output(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
    

    
def main(item):
   img_bgr = cv2.imread(item)
   height, width, channel = img_bgr.shape 
   img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
   no_points = 8;
   radius = 1;
   lbp = local_binary_pattern(img_gray, no_points, radius, method='default')
   print(lbp.shape)
   x = itemfreq(lbp.ravel())
   # Normalize the histogram
   hist = x[:, 1]/sum(x[:, 1])
   X_test.append(hist)
   lbp_feat=np.array(X_test)
   output_list = []
   output_list.append({
        "img": img_gray,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "Gray Image",
        "type": "gray"        
   })
   output_list.append({
        "img": lbp,
        "xlabel": "",
        "ylabel": "",
        "xtick": [],
        "ytick": [],
        "title": "LBP Image",
        "type": "gray"
   })    
   output_list.append({
        "img": hist,
        "xlabel": "Bins",
        "ylabel": "Number of pixels",
        "xtick": None,
        "ytick": None,
        "title": "Histogram(LBP)",
        "type": "histogram"
   })

   show_output(output_list)
                             
    
X_name = []
X_test = []
#items = os.listdir('/content/drive/My Drive/Training Images')
#for each_image in items:
'''
if each_image.endswith(".jpg"):
print (each_image)
full_path = "/content/drive/My Drive/Training Images/" + each_image
X_name.append(full_path)
print (full_path)
main(full_path)
'''
main("/content/lenna (1).jpg")
