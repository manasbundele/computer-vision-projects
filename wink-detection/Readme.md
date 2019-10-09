# Wink Detection using Haar Cascades

Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, “Rapid Object Detection using a Boosted Cascade of Simple Features” in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.*

In this project, we have to detect a winking face and mark face with blue bounding box, otherwise just detect a face with a green bounding box around it. This implementation can detect winks in both images as well as live video. 

For wink detection in images:

> python DetectWink.py /path/to/folder

For live video:

> python DetectWink.py


*Ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
