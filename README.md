# Child Mind Institute - Detect Sleep States - Kaggle Competition

Sleep is vital in regulating mood, emotions, and behavior across all age groups, especially in children. Accurately detecting sleep and wakefulness using wrist-worn accelerometer data enables researchers to better understand sleep patterns and disturbances in children. These predictions could have significant implications, particularly for children and youth experiencing mood and behavioral difficulties.

## Downloading the Data

This project uses **Git Large File Storage (Git LFS)** to manage large files, such as training and testing datasets. Please ensure that **Git LFS** is installed on your machine before cloning the repository to correctly download the data files.

### Steps to Download the Data:

1. **Install Git LFS**:  
   If you haven't installed Git LFS yet, run the following command to set it up on your machine:
   ```bash
   git lfs install
   ```

2. **Clone the Repository**:  
   Clone the repository to download both the code and the large data files tracked by Git LFS:
   ```bash
   git clone https://github.com/zhukovanadezhda/sleep-states.git
   ```

3. **Verify Large File Download**:  
   Git LFS should automatically download the large files (e.g., dataset files) when you clone the repository. If, for some reason, the large files are not downloaded, you can manually trigger the download by running:
   ```bash
   git lfs pull
   ```

4. **Unzip the archive**:
    After cloning the repository, you will find a ZIP file containing the dataset in the `data` folder. To extract the data, run the following command:
    ```bash
    unzip data/child-mind-institute-detect-sleep-states.zip -d data/
    ```
    This will extract the contents of the ZIP file into the `data/` directory, making the dataset available for analysis.

After completing these steps, you should have access to both the code and the dataset files needed for the project.
