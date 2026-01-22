# SyncG

SyncG is a tool designed for generating synthetic analogy gauge datasets. This repository provides both pre-built datasets and the original scripts to generate your own customized data.

## 💾 Pre-built Dataset

A pre-built dataset is available for immediate use. You can download all the necessary files from the Hugging Face repository:

*   **[Download the dataset](https://huggingface.co/datasets/YihengDeng/syncG/tree/main)**

## ⚙️ Custom Dataset Generation

For those who wish to create their own datasets tailored to specific needs, we provide the original generation scripts. The entire process is designed to run headlessly.

Here is a step-by-step guide to get you started.

### 1. Prerequisites: Blender Environment

We recommend using Docker for a consistent and clean environment.

* **Pull the Blender Docker image:**

  ```bash
  docker pull linuxserver/blender:4.1.1
  ```

* **Install Python dependencies:**
  Once the container is running, use the following command to install the required Python libraries within Blender's Python environment:

  ```bash
  /blender/4.1/python/bin/python3.11 -m pip install opencv-python
  ```

### 2. HDRI Files

High-Dynamic Range Image (HDRI) files are used for scene lighting.

*   Upload your HDRI files (which typically have a `.hdr` extension) to the `scene_file` folder.
*   A collection of pre-selected HDRIs is available for download at the same [Hugging Face repository](https://huggingface.co/datasets/YihengDeng/syncG/tree/main).

### 3. Configuration (`data_config.json`)

The `data_config.json` file contains the primary parameters for the generation process. Pay close attention to the following arguments:

*   `project_root`: The absolute path to the main project folder. Ensure the `scene_file` and `blender_project` folders are located under this root directory.
*   `dataset_name`: A unique name for your generated dataset.

### 4. Script Setup

You'll need to configure the paths within the generation scripts.

* **In `generate_syncG.py`:**
  On line 896, update the `config_path` variable to point to the actual location of your `data_config.json` file.

* **In `generate.sh`:**
  Update the paths for the Blender project file (`/syncG/blender_project/fakemeter_oil_new.blend`) and the script file (`/syncG/scripts/scripts_clean.py`) to reflect their actual locations on your system.

* **Link the script:**
  It is recommended to link (or move) the `generate.sh` script to the `/blender` directory inside your container for easier execution.

  ```bash
  # Create a symbolic link
  ln -s /path/to/your/generate.sh /blender/generate.sh
  ```

### 5. Running the Generation

Once everything is configured, you can start the generation process.

* **Navigate to the Blender directory:**

  ```bash
  cd /blender
  ```

* **Execute the script:**

  ```bash
  bash generate.sh
  ```

***

### Appendix 1. Data Card

 A reference table listing all annotation keys, their descriptions, and data types.


| **Field Name**            | **Description**                                              | **Data Type**      |
| ------------------------- | ------------------------------------------------------------ | ------------------ |
| **file_name**             | **Unique identifier prefix for the image  file.**            | **String**         |
| **gauge_type**            | **The category or type of the gauge  (e.g., pressure, voltmeter).** | **String**         |
| **width**                 | **Image width in pixels.**                                   | **Integer**        |
| **height**                | **Image height in pixels.**                                  | **Integer**        |
| **ground_truth**          | **The precise ground truth reading value  of the gauge.**    | **Float**          |
| **long_interval_degree**  | **Angle (in degrees) between adjacent  major scale marks relative to the dial center.** | **Float**          |
| **long_interval_value**   | **The numerical difference in value  between adjacent major scale marks.** | **Integer**        |
| **long_num**              | **Total number of major scale marks on  the dial.**          | **Integer**        |
| **start_value**           | **The initial (minimum) numerical value  of the gauge scale.** | **Integer**        |
| **small_interval**        | **Number of minor scale subdivisions  between two adjacent major marks.** | **Integer**        |
| **text_interval**         | **Frequency of label occurrence (number  of major marks per labeled value).** | **Integer**        |
| **pointer_rotate_degree** | **Deflection angle of the pointer from  its initial position to the current reading.** | **Float**          |
| **num_scalemark_before**  | **Count of major scale marks preceding  the current pointer position.** | **Integer**        |
| **min_scalemark**         | **The reading value of the major scale  mark immediately preceding the pointer.** | **Integer**        |
| **closest_scalemark**     | **The reading value of the major scale  mark spatially closest to the pointer.** | **Integer**        |
| **ann_camera_location**   | **The spatial coordinates [x, y, z] of  the camera in the 3D scene.** | **List [Float]**   |
| **homo_matrix**           | **3×3 matrix parameters for image  perspective correction (**homography matrix) | **List [Float]**   |
| **dial_bbox_annotations** | **Bounding box coordinates [x_min,y_min,x_max,y_max] for the gauge dial.** | **List [Integer]** |
| **text_bbox_annotations** | **List of objects containing bounding  boxes and values for numerical text labels.** | **List [Dict]**    |
| **seg_annotations**       | **Segmentation data (masks/polygons) for  scale marks and pointers.** | **List [Dict]**    |
| **keypoints_annotations** | **Keypoint**  **coordinates for pointer (origin/tip) and scale marks.** | **List [Dict]**  |
| **scene_name**            | **Filename of the HDRI map used for  background/lighting rendering.** | **String**         |
| **seed**                  | **Random seed used for synthetic  generation logic.**        | **Integer**        |
