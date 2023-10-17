# Google Cloud Vision API

## Setup

Prepare a **Google Cloud project ID** and a **Google Cloud API key**.

(Optional) Create a new virtual environment and activate it:
```bash
python -m venv .env
source .env/bin/activate
```

Ensure that the local Python version is >= 3.8. Install the required libraries:
```bash
pip install -r requirements.txt
```

## Try it in Jupyter Notebook

Install and start the Jupyter Lab:
```bash
pip install jupyterlab
jupyter lab
```

Open `gcloud_obj_detect_example.ipynb`, fill in the Google Cloud API key and project ID in the 5th cell, and run the cells one by one.

## Example in SQL

```sql
CREATE OR REPLACE FUNCTION gvision_obj_detect
IMPL 'google_cloud_vision_object_detector.py';

DROP TABLE IF EXISTS MyImage;
LOAD IMAGE 'imgs/bicycle_example.png' INTO MyImage;
LOAD IMAGE 'imgs/example2.jpeg' INTO MyImage;
LOAD IMAGE 'imgs/example3.jpg' INTO MyImage;

SELECT gvision_obj_detect(data) from MyImage;
```