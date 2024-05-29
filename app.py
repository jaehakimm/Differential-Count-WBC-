from flask import Flask, render_template, request
import os
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import cv2
import matplotlib.pyplot as plt
from collections import Counter

app = Flask(__name__)

def configure_detectron2():
    setup_logger()
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cuda'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.RETINANET.NUM_CLASSES = 8
    cfg.MODEL.WEIGHTS = "modules\model_final.pth"
    cfg.DATASETS.TEST = ("my_dataset_test", )
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    return cfg

def load_image(image_path):
    return cv2.imread(image_path)

def predict_image(predictor, image):
    outputs = predictor(image)
    return outputs

def visualize_prediction(image, outputs, metadata):
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cfg = configure_detectron2()
    predictor = DefaultPredictor(cfg)
    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")

    image_data = {}  # Store image names and predictions
    total_results = []  # Store results for all images

    for uploaded_file in request.files.getlist('files[]'):
        if uploaded_file.filename != '':
            image_path = os.path.join('static/uploads', uploaded_file.filename)
            uploaded_file.save(image_path)
            image = load_image(image_path)
            outputs = predict_image(predictor, image)
            image_with_predictions = visualize_prediction(image, outputs, my_dataset_train_metadata)

            # Construct predicted image filename
            predicted_image_filename = 'predicted_' + uploaded_file.filename
            predicted_image_path = os.path.join('static/uploads', predicted_image_filename)

            # Save the predicted image in the 'static/uploads' directory
            plt.imsave(predicted_image_path, image_with_predictions)

            image_data[uploaded_file.filename] = predicted_image_filename

            detected_classes = outputs["instances"].pred_classes.tolist()
            class_names = [my_dataset_train_metadata.thing_classes[i] for i in detected_classes]
            total_results.append(class_names)

    print("Total results:", total_results)

    # Count occurrences of each class
    class_counts = Counter(class_name for result in total_results for class_name in result)

    print("Class counts:", class_counts)

    # Calculate percentages excluding "RBC"
    class_percentages = calculate_percentages(class_counts)

    print("Class percentages:", class_percentages)

    return render_template('index.html', image_data=image_data, total_results=total_results, class_counts=class_counts, class_percentages=class_percentages)

def calculate_percentages(class_counts):
    # Exclude "RBC" class from the calculation
    class_counts_without_rbc = class_counts.copy()
    rbc_count = class_counts_without_rbc.pop("RBC", None)

    # Calculate percentages
    total_predictions = sum(class_counts_without_rbc.values())
    class_percentages = {class_name: count / total_predictions * 100 for class_name, count in class_counts_without_rbc.items()}

    return class_percentages

def main():
    class_names = ["","Basophil", "Lymphocyte", "Monocyte", "Neutrophil", "Platelet", "RBC", "eosinophil"]
    MetadataCatalog.get("my_dataset_train").set(thing_classes=class_names)
    print(MetadataCatalog.get("my_dataset_train"))

if __name__ == "__main__":
    main()
    app.run(debug=True)
