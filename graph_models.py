"""
FUNCTION: graph_models.py
-------------------------
Graphs the models used in correction.
"""

import sys
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from settings import FIGURE_DIRECTORY, RECOG_EVAL_DIRECTORY

def main():
   graph("accuracy-scatter", "basic")
   graph("accuracy-scatter", "size-color")
   graph("regression", "basic")
   graph("flat-change", "basic")
   graph("flat-change", "size-color")


class SizeBucket:
   """
   This class defines a bucket, which is used to group text files based on their word count.
   """
   def __init__(self, low, high, color):
      self.low = low  # lower bound of number of words
      self.high = high  # upper bound of number of words
      self.color = color  # color in plot
      self.contents = []  # arr containing data of files in group


def graph(graphType, graphLevel="basic") -> None:
   """
   Graphs evaluated models. To evaluate models, use accuracy_counter.py

   Args:
      graphType: String, "accuracy" or "change"
      graphLevel: String, "basic" or "advanced"
   """
   checkParams(graphType, graphLevel)

   plt.clf()
   fig = plt.figure(linewidth=5, edgecolor="#04253a")
   plt.rcParams['legend.handlelength'] = 0.8
   plt.rcParams['legend.handleheight'] = 1
   plt.rcParams['legend.numpoints'] = 1

   # prepares data from pickle file
   with open(f"{RECOG_EVAL_DIRECTORY}model_v2-times-random-sample.pkl", "rb") as f:
      d = pickle.load(f)

   total_files = d["TOTAL"]
   orig_percent = np.array(d["ORIG"]) * 100
   pred_percent = np.array(d["PRED"]) * 100
   zip_percent = np.array(d["ZIP"]) * (100, 100)
   flat_change = np.array(d["FLAT"]) * 100
   perc_change = np.array(d["PERC"]) * 100
   file_size = d["SIZE"]

   # groups files by word count
   xs_files = SizeBucket(0, 249, "pink")
   small_files = SizeBucket(250, 499, "red")
   med_files = SizeBucket(500, 999, "yellow")
   large_files = SizeBucket(1000, 2499, "green")
   xl_files = SizeBucket(2500, max(file_size), "blue")

   ordered_buckets = [xs_files, small_files, med_files, large_files, xl_files]

   for i in range(total_files):
      for bucket in ordered_buckets:
         if (file_size[i] < bucket.high):
            bucket.contents.append(zip_percent[i])
            break


   # Scatter plot of orig accuracy vs pred accuracy
   if (graphType == "accuracy-scatter"):
      # sets up graph and labels
      plt.plot([-100, 200], [-100, 200], color="#000000", linestyle="dashed", linewidth="0.5")
      plt.xlabel("Original File Readability (%)")
      plt.ylabel("Predicted File Readability (%)")
      plt.title(f"Seq2Seq OCR Prediction Accuracies (n={total_files})")
      plt.xlim(0, 100), plt.ylim(0, 100)
      plt.xticks([0, 20, 40, 60, 80, 100])
      plt.yticks([0, 20, 40, 60, 80, 100])

      if (graphLevel == "basic"):
         plt.scatter(*zip(*zip_percent), c="black", s=2)

      elif(graphLevel == "size-color"):
         for bucket in ordered_buckets:
            plt.scatter(*zip(*bucket.contents), c=bucket.color, s=1)
            plt.plot([], c=bucket.color, marker="s", markersize=10, label=f"{bucket.low}-{bucket.high} words (n={len(bucket.contents)})")

         legend = plt.legend(loc="lower right", title="File Size", fancybox=True)
         legend.get_title().set_fontsize('12')  # adjusts legend fontsizes
         plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')
   

   # Linear regressions of pred accuracy as a function of orig accuracy
   elif (graphType == "regression"):
      plt.plot([-100, 200], [-100, 200], color="#000000", linestyle="dashed", linewidth="0.5")
      plt.xlabel("Original File Readability (%)")
      plt.ylabel("Predicted File Readability (%)")
      plt.title(f"Seq2Seq Predicted Accuracy Regressions (n={total_files})")
      plt.xlim(0, 100), plt.ylim(0, 100)
      plt.xticks([0, 20, 40, 60, 80, 100])
      plt.yticks([0, 20, 40, 60, 80, 100])

      m, b = np.polyfit(*zip(*zip_percent), 1)
      plt.plot(np.arange(0, 100, 0.1), m * np.arange(0, 100, 0.1) + b, color="black")
      plt.plot([], c="black", marker="s", markersize=10, label=f"All Files ({'y={:.2f}x+{:.2f}'.format(m, b)})")
      legend = plt.legend(loc="lower right", fancybox=True)
      
      for bucket in ordered_buckets:
         m, b = np.polyfit(*zip(*bucket.contents), 1)
         plt.plot(np.arange(0, 100, 0.1), m * np.arange(0, 100, 0.1) + b, color=bucket.color, linestyle="dashed")
         plt.plot([], c=bucket.color, marker="s", markersize=10, label=f"{bucket.low}-{bucket.high} words ({'y={:.2f}x+{:.2f}'.format(m, b)})")
      legend = plt.legend(loc="lower right", title="File Size", fancybox=True)
      legend.get_title().set_fontsize('12')  # adjusts legend fontsizes

      
      plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')

   
   # Scatter plot of orig accuracy vs flat change in pred accuracy (units: %)
   elif (graphType == "flat-change"):
      plt.plot([-100, 200], [0, 0], color="#000000", linestyle="dashed", linewidth="0.5")
      plt.xlabel("Original File Readability (%)")
      plt.ylabel("Flat Change (%)")
      plt.title(f"Seq2Seq Recognizability Improvement (n={total_files})")
      plt.xlim(0, 100)
      plt.xticks([0, 20, 40, 60, 80, 100])

      if (graphLevel == "basic"):
         plt.scatter(orig_percent, flat_change, c="black", s=2)

      elif (graphLevel == "size-color"):
         for bucket in ordered_buckets:
            bucket.contents = [(a, b - a) for a, b in bucket.contents]
            plt.scatter(*zip(*bucket.contents), c=bucket.color, s=1)
            plt.plot([], c=bucket.color, marker="s", markersize=10, label=f"{bucket.low}-{bucket.high} words (n={len(bucket.contents)})")

         legend = plt.legend(loc="upper right", title="File Size", fancybox=True)
         legend.get_title().set_fontsize('12')  # adjusts legend fontsizes
         plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')

   # saves graph to file
   if (graphType == "regression"):
      plt.savefig(f"{FIGURE_DIRECTORY}{graphType}.png", bbox_inches="tight", edgecolor=fig.get_edgecolor())
   else:
      fig.savefig(f"{FIGURE_DIRECTORY}{graphType}-{graphLevel}.png", bbox_inches="tight", edgecolor=fig.get_edgecolor())
   plt.show()
   return


def measure_improvement():
   with open(f"{RECOG_EVAL_DIRECTORY}model_v2.pkl", "rb") as f:
      d = pickle.load(f)

   total_files = d["TOTAL"]
   orig_percent = np.array(d["ORIG"]) * 100
   pred_percent = np.array(d["PRED"]) * 100
   zip_percent = np.array(d["ZIP"]) * (100, 100)
   flat_change = np.array(d["FLAT"]) * 100
   perc_change = np.array(d["PERC"]) * 100
   file_size = d["SIZE"]
   print("AVERAGE IMPROVEMENT: ", (sum(flat_change) / len(flat_change)))

   total = 0
   for i in range(total_files):
      if flat_change[i] < 0:
         total += 100
      total += flat_change[i]
   print("IMPROVEMENT EXCLUDING BAD: ", total / len(flat_change))


def checkParams(graphType, graphLevel):
   """
   Makes sure the params passed into the graph() function are valid.
   """
   possTypes = ["accuracy-scatter", "regression", "flat-change"]
   possLevels = ["basic", "size-color"]
   if graphType not in possTypes:
      print("For the graph type, please specify one of the following: ", ",".join(possTypes))
      sys.exit()
   if  graphLevel not in possLevels:
      print("For the graph level, please specify one of the following: ", ",".join(possLevels))
      sys.exit()



if __name__ == "__main__":
    main()