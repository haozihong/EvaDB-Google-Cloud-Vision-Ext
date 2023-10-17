# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.configuration.configuration_manager import ConfigurationManager
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe


class GoogleCloudVisionObjectDetector(AbstractFunction):
    """
    Input Signatures:
        data (str) : image data in NumPy array format

    Output Signatures:
        columns=["name", "score", "bounds"],
        name (str)   : Labels of the detected objects
        score (str)  : Likelihood of the detected objects
        bounds (str) : Bounding box (x1, y1, x2, y2) of the objects.

    Example Usage:
        Create an UDF with:
            CREATE OR REPLACE FUNCTION gvision_obj_detect
            IMPL  'evadb/functions/google_cloud_vision_object_detector.py';

        Load any images into a table:
            LOAD IMAGE 'bicycle_example.png' INTO MyImage;
            LOAD IMAGE 'example2.jpeg' INTO MyImage;

        Get the results by calling the UDF:
            SELECT gvision_obj_detect(data) from MyImage;
    """

    @property
    def name(self) -> str:
        return "GoogleCloudVisionObjectDetector"

    @setup(cacheable=True, function_type="object_detection", batchable=True)
    def setup(self) -> None:
        pass

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["name", "score", "bounds"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,)],
            )
        ],
    )
    def forward(self, frames: pd.DataFrame):
        # Register API key, try configuration manager first
        # Commented out because the yml config file is not configurable with the pip-installed EvaDB
        # api_key_string = ConfigurationManager().get_value("third_party", "GOOGLE_CLOUD_API_KEY")
        # quota_project_id = ConfigurationManager().get_value("third_party", "GOOGLE_CLOUD_PROJECT_ID")
        api_key_string, quota_project_id = None, None

        # If not found, try OS Environment Variable
        if api_key_string is None:
            api_key_string = os.environ.get("GOOGLE_CLOUD_API_KEY", None)
        if quota_project_id is None:
            quota_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", None)
        assert (
            api_key_string is not None and quota_project_id is not None
        ), "Please set your Google Cloud API key and Project ID in evadb.yml file "\
            "(third_party, GOOGLE_CLOUD_API_KEY) and (third_party, GOOGLE_CLOUD_PROJECT_ID) "\
            "or environment variable (GOOGLE_CLOUD_API_KEY) and (GOOGLE_CLOUD_PROJECT_ID)"

        from google.cloud import vision
        import cv2

        client = vision.ImageAnnotatorClient(
            client_options={"api_key": api_key_string, "quota_project_id": quota_project_id}
        )

        res = []
        imgs = np.ravel(frames.to_numpy())
        for img in imgs:
            img_h, img_w = img.shape[0], img.shape[1]
            success, encoded_image = cv2.imencode('.png', img)
            image = vision.Image(content=encoded_image.tobytes())

            objects = client.object_localization(image=image).localized_object_annotations
            names, scores, bounds = [], [], []
            for o in objects:
                names.append(o.name)
                scores.append(o.score)
                bounds.append([
                    o.bounding_poly.normalized_vertices[0].x * img_w,
                    o.bounding_poly.normalized_vertices[0].y * img_h,
                    o.bounding_poly.normalized_vertices[2].x * img_w,
                    o.bounding_poly.normalized_vertices[2].y * img_h,
                ])
            res.append({"name": names, "score": scores, "bounds": bounds})
        
        return pd.DataFrame(
            res,
            columns=["name", "score", "bounds"],
        )
