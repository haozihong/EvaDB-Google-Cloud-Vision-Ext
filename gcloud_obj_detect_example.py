import evadb
import os
import cv2
from matplotlib import pyplot as plt

def main():
    cursor = evadb.connect().cursor()

    # Create an UDF for the Google Cloud Vision Object Detector
    cursor.query("CREATE OR REPLACE FUNCTION gvision_obj_detect IMPL  'google_cloud_vision_object_detector.py';").df()

    # Load some images into a table
    cursor.query("DROP TABLE IF EXISTS MyImage;").df()
    cursor.query("LOAD IMAGE 'imgs/bicycle_example.png' INTO MyImage;").df()
    cursor.query("LOAD IMAGE 'imgs/example2.jpeg' INTO MyImage;").df()
    cursor.query("LOAD IMAGE 'imgs/example3.jpg' INTO MyImage;").df()

    # Set Google Cloud API key and project ID in environment variables
    # Or set them here:
    os.environ["GOOGLE_CLOUD_API_KEY"] = "YOUR_GOOGLE_CLOUD_API_KEY"
    os.environ["GOOGLE_CLOUD_PROJECT_ID"] = "YOUR_GOOGLE_CLOUD_PROJECT_ID"

    # Call the UDF to do the object detection
    obj_loc = cursor.query("SELECT gvision_obj_detect(data) from MyImage;").df()

    def annotate_images(detections, input_images_paths):
        '''A helper function marking the detected objects on the original images'''
        color1=(207, 248, 64)
        color2=(255, 49, 49)
        thickness=3

        for df, path in zip(detections[['name', 'bounds']].values, input_images_paths):
            img = cv2.imread(path)
            for label, bbox in zip(df[0], df[1]):
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # object bbox
                img=cv2.rectangle(img, (x1, y1), (x2, y2), color1, thickness) 
                # object label
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color2, thickness) 

            path_sep = os.path.splitext(path)
            plt.imsave(f'{path_sep[0]}_annotated{path_sep[1]}', img)


    # Mark the detected objects on the original images
    img_paths = cursor.query("select name from MyImage;").df().values.tolist()
    img_paths = [e[0] for e in img_paths]
    annotate_images(obj_loc, img_paths)

if __name__ == "__main__":
    main()