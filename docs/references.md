# References

[Drive](https://drive.google.com/drive/folders/1RvwVxlfmq1SzwZ_TeHPp65TRBNNIr7VY?usp=share_link)

## Ball Tracking

- [TrackNet (Huang et al., 2019)](https://arxiv.org/abs/1907.03698) — heatmap-based deep learning network for tracking small, fast-moving objects. Takes consecutive frames as input, outputs Gaussian heatmap. Foundational architecture for our ball tracking module.
- [Tennis-Ball-Tracker (Nikhil Grad, 2024)](https://github.com/nikhilgrad/Tennis-Ball-Tracker) — YOLOv8 + interpolation approach using the same Roboflow dataset we use for detection.
- [Tennis Ball Detection Dataset (Viren Dhanwani, 2022)](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection) — Roboflow dataset used for ball detection baseline.

## Court Detection

- [Tennis Analysis using Deep Learning and Machine Learning (Kosolapov, 2023)](https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee) — end-to-end pipeline covering court keypoint detection (14 keypoints), ball tracking, bounce detection and player tracking from a single camera. Blueprint for our full pipeline.

## Player Tracking

- [tennis-tracking (ArtLabss, 2021)](https://github.com/ArtLabss/tennis-tracking) — open-source pipeline combining TrackNet for ball tracking, YOLOv3 + ResNet50 for player detection, homography-based court detection and bounce detection. Covers nearly all our MVP tasks.

## Architecture & Training

- [Dropout in CNNs](https://arxiv.org/abs/1801.05134) — discusses use of dropout in CNNs. Relevant to whether dropout should be added alongside batch normalization in our architecture (BatchNorm already provides regularization, so dropout may be redundant or even harmful).