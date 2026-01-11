import matplotlib.pyplot as plt
import scripts.others.util as util
import pandas as pd



def plotResults(dataframe, savePath=None, showPlot=True, showMm=False, showConfidence=True, title="Pupil Diameter Over Time"):
    # clear previous plots
    plt.clf()
    # plot data based on dataframe
    plt.figure("Showing results for " + str(savePath),figsize=(12, 6))
    plt.subplot(2, 1, 1)
    if showMm:
        plt.plot(dataframe['timestamp'], dataframe['diameter_mm'], label='Pupil Diameter (mm)', color='blue')
        plt.ylabel('Diameter (mm)')
    else:
        plt.plot(dataframe['timestamp'], dataframe['diameter'], label='Pupil Diameter (pixels)', color='blue')
        plt.ylabel('Diameter (pixels)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.legend()
    
    if showConfidence:
        plt.subplot(2, 1, 2)
        plt.plot(dataframe['timestamp'], dataframe['confidence'], label='Confidence', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence')
        plt.title('Detection Confidence Over Time')
        plt.legend()
    
    plt.tight_layout()
    if savePath:
        plt.savefig(savePath)
        util.dprint(f"Plot saved to '{savePath}'")
    if showPlot:
        plt.show()
