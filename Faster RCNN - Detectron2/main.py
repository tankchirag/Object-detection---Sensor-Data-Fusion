from pathlib import Path
import requests, json, shutil, random
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tensorboard import program
import pickle, os, cv2

def download_dataset(url, target_directory):
    # Make a URL request 
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for errors
    
    # Open a file with write-binary mode
    with open(target_directory, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):  # Download the file in chunks
            file.write(chunk)

    print("Download completed!")

class COCOConverter:
    def __init__(self, dataset_path, coco_dir_path, validation_set_ratio=0.3):
        self.dataset_path = dataset_path
        self.coco_dir_path = coco_dir_path
        self.validation_set_ratio = validation_set_ratio

    def create_dir_structure(self):

        base_dir = Path(self.coco_dir_path)

        # Create base directory if it does not exist
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (base_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / "train" / "annotations").mkdir(parents=True, exist_ok=True)
        (base_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (base_dir / "val" / "annotations").mkdir(parents=True, exist_ok=True)

    def move_dataset(self):
        random.seed(42)

        self.train_img_path = self.coco_dir_path + "/train/images"
        self.train_ann_path = self.coco_dir_path + "/train/annotations"

        self.val_img_path = self.coco_dir_path + "/val/images"
        self.val_ann_path = self.coco_dir_path + "/val/annotations"

        for scene in Path(self.dataset_path).iterdir():
            self.images_dir = scene / 'camera_01/camera_01__data'
            self.jsons_dir = scene / 'camera_01/camera_01__annotation'

            self.image_paths = sorted(self.images_dir.glob('*.png'))
            self.json_paths = sorted(self.jsons_dir.glob('*.json'))

            paired_paths = list(zip(self.image_paths, self.json_paths))
            random.shuffle(paired_paths)

            split_index = int(self.validation_set_ratio * len(paired_paths))

            train_pairs = paired_paths[:split_index]
            val_pairs = paired_paths[split_index:]

            for train_img, train_json in train_pairs:
                destination_path_to_copy_image = Path(self.train_img_path + f'/{train_img.stem}.png')
                destination_path_to_copy_json = Path(self.train_ann_path + f'/{train_json.stem}.json')

                if not destination_path_to_copy_image.exists() and not destination_path_to_copy_json.exists():
                    shutil.copy(train_img, destination_path_to_copy_image)
                    shutil.copy(train_json, destination_path_to_copy_json)

            for val_img, val_json in val_pairs:
                destination_path_to_copy_image = Path(self.val_img_path + f'/{val_img.stem}.png')
                destination_path_to_copy_json = Path(self.val_ann_path + f'/{val_json.stem}.json')

                if not destination_path_to_copy_image.exists() and not destination_path_to_copy_json.exists():
                    shutil.copy(val_img, destination_path_to_copy_image)
                    shutil.copy(val_json, destination_path_to_copy_json)
            
        # print stats
        self.total_train_img = list(Path(self.train_img_path).glob('*'))
        self.total_train_ann = list(Path(self.train_ann_path).glob('*'))
        self.total_val_img = list(Path(self.val_img_path).glob('*'))
        self.total_val_ann = list(Path(self.val_ann_path).glob('*'))

        print("There are ", len(self.total_train_img), " training images.")
        print("There are", len(self.total_train_ann), " training annotations.\n")
        print("There are ", len(self.total_val_img), " validation images.")
        print("There are ", len(self.total_val_ann), " validation annotations.\n")
        print("There are ", len(self.total_val_ann) + len(self.total_train_ann), " images and annotations in total\n")


    def read_json_file(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def write_json_file(self, data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    def combine_json_files(self, directory):
        path = Path(directory)
        json_files = path.glob('*.json')

        combined_data = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        categories_set = set()
        for json_file in json_files:
            data = self.read_json_file(json_file)

            # Info and licenses - copy from first file as it's likely the same for all
            if not combined_data["info"]:
                combined_data["info"] = data["info"]
                combined_data["licenses"] = data["licenses"]

            # Images
            combined_data["images"].append(data["image"])

            # Categories - avoid duplicates by using a set of tuples
            for category in data["categories"]:
                category_tuple = (category["category_id"], category["name"], category["supercategory"])
                if category_tuple not in categories_set:
                    categories_set.add(category_tuple)
                    combined_data["categories"].append({
                        "id": int(category["category_id"]),
                        "name": category["name"],
                        "supercategory": category["supercategory"]
                    })

            # Annotations
            for annotation in data["annotations"]:
                combined_data["annotations"].append({
                    "id": annotation["det_id"],
                    "image_id": annotation["image_id"],
                    "category_id": int(annotation["category_id"]),
                    "bbox": annotation["bbox"],
                    "area": annotation["bbox"][2] * annotation["bbox"][3],
                    "iscrowd": 0
                })

        return combined_data

    def convert_to_coco(self):
        self.create_dir_structure()

        self.move_dataset()

        # Combine into one COCO-format JSON
        train_combined_annotations = self.combine_json_files(self.train_ann_path)
        val_combined_annotations = self.combine_json_files(self.val_ann_path)

        # Set COCO-format target path
        self.train_coco_annotation_path = self.train_ann_path + '/train_coco_annotations.json'
        self.val_coco_annotation_path = self.val_ann_path + '/val_coco_annotations.json'

        # Write the COCO-format JSON files
        self.write_json_file(train_combined_annotations, self.train_coco_annotation_path)
        self.write_json_file(val_combined_annotations, self.val_coco_annotation_path)

class DetectronModel():

    def __init__(self, coco_dir_path):
        self.coco_dir_path = coco_dir_path
        self.train_cfg = get_cfg()
        self.train_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.train_cfg.DATALOADER.NUM_WORKERS = 4
        self.predict_cfg = get_cfg()
        self.predict_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.predict_cfg.DATALOADER.NUM_WORKERS = 4
        register_coco_instances("faster_train", {}, (coco_dir_path + "/train/annotations/train_coco_annotations.json"), (coco_dir_path + "/train/images"))
        register_coco_instances("faster_val", {}, (coco_dir_path + "/val/annotations/val_coco_annotations.json"), (coco_dir_path + "/val/images"))

    def create_train_configs(self, output_dir_path, epochs, classes, training_images_len, batch_size=4, lr=0.00025, checkpoint_path=0):
        self.train_cfg.DATASETS.TRAIN = ("faster_train", )
        self.train_cfg.OUTPUT_DIR = output_dir_path
        self.train_cfg.DATASETS.TEST = ("faster_val", )
        if checkpoint_path == 0:
            self.train_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from train_model zoo
        else:
            self.train_cfg.MODEL.WEIGHTS = checkpoint_path  # Let training initialize from train_model zoo
        self.train_cfg.SOLVER.IMS_PER_BATCH = batch_size  # This is the real "batch size" commonly known to deep learning people
        self.train_cfg.SOLVER.BASE_LR = lr  # pick a good LR

        # Number of iterations per epoch = Total Number of Training Data Points (1728) / Batch Size
        # Total Iterations = Iterations per Epoch × Number of Epochs
        iter_per_epoch = training_images_len / batch_size
        iterations = epochs * iter_per_epoch
        self.train_cfg.SOLVER.MAX_ITER = iterations    # = 43200 itersaations / 100 epochs

        self.train_cfg.SOLVER.STEPS = []        # do not decay learning rate
        self.train_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
        self.train_cfg.MODEL.ROI_HEADS.NUM_CLASSES = classes # 5 
        self.train_cfg.TEST.EVAL_PERIOD = iterations / epochs  # Perform evaluation every epoch
        self.train_cfg.SOLVER.CHECKPOINT_PERIOD = iterations / epochs  # Save a checkpoint every epoch, or choose a different interval

        with open('train_config.pkl', 'wb') as f:
            pickle.dump(self.cfg, f)
        
        self.trainer = DefaultTrainer(self.train_cfg)

        print("Training Configurations have been saved.")

    def create_predict_configs(self, model_path):
        self.predict_cfg.MODEL.WEIGHTS = model_path  # Let training initialize from train_model zoo

        with open('predict_config.pkl', 'wb') as f:
            pickle.dump(self.predict_cfg, f)
        
        self.predictor = DefaultPredictor(self.predict_cfg)
        print("Prediction Configurations have been saved.")

    def train(self):
        os.makedirs(self.train_cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer.resume_or_load(resume=True)
        self.trainer.train()

    def get_train_diagnostics(self):
        # Create a TensorBoard program object
        tb = program.TensorBoard()

        # Configure the TensorBoard
        tb.configure(argv=[None, '--logdir', self.train_cfg.OUTPUT_DIR, '--bind_all'])  # 'output' is the directory with training logs

        # Start TensorBoard
        url = tb.launch()

        print(f"TensorBoard is running at {url}")

        return url
    
    # Function to update the dictionary
    def update_counter(self, class_id):
        if class_id in self.class_counters:
            self.class_counters_list[0][class_id] += 1
        else:
            self.class_counters_list[0][class_id] = 1

    def predict(self, test_images_path, test_json_path):
        test_metadata = MetadataCatalog.get("faster_val")

        self.class_counters = {}
        self.class_counters_list = []
        self.class_counters_list.append(self.class_counters)

        # Path to the COCO JSON annotation file
        json_file_path = test_metadata.json_file

        paths = []
        for file in os.listdir(test_images_path):
            # Construct full file path
            file_path = os.path.join(test_images_path, file)
            # Check if its a file
            if os.path.isfile(file_path):
                paths.append(file_path)

        # Load JSON data from the file
        with open(json_file_path, 'r') as file:
            coco_data = json.load(file)

        # Extract categories and create a dictionary mapping IDs to labels
        prediction_classes = {category['id']: category['name'] for category in coco_data['categories']}

        # Print the dictionary to verify
        print("The prediction classes are: \n", prediction_classes, "\n")

        output_json = self.coco_dir_path + "/retrain_fasterrcnn_count.json"

        with open(output_json, 'w') as file:
            fasterrcnn_count = []
            for p in paths:
                img_data = {}

                img_data['img_path'] = p # Append the image test_path to the dictionary

                img = cv2.imread(p)
                output = self.predictor(img)
                v = Visualizer(img[:, :, ::-1],
                        metadata=test_metadata,
                        scale=0.8
                        )
                out = v.draw_instance_predictions(output["instances"].to("cpu"))

                box = output['instances'].pred_boxes # Boxes(tensor([[]]))
                box_tensor = box.tensor # tensor([])
                box_values = box_tensor.to('cpu').tolist() # []

                img_data['bounding_boxes'] = box_values # Append the bounding boxes coordinates to the dictionary

                pclass_tensor = output['instances'].pred_classes # tensor()
                pclass = pclass_tensor.tolist() # list
                pclass_names = []

                for i in pclass:
                    pclass_name = prediction_classes[i] # string
                    self.update_counter(pclass_name) # Increment the instance
                    pclass_names.append(pclass_name)

                img_data['classes'] = pclass_names # Append the class labels to the dictionary

                cv2.imshow("output", out.get_image()[:, :, ::-1])
                cv2.waitKey(1)

                fasterrcnn_count.append(img_data) # Append the image data to the json list

            fasterrcnn_count.append(self.class_counters) # Append the classes count to the json list

            json.dump(fasterrcnn_count, file, indent=4) # Append the content to the json file
                
    def evaluate_test_dataset(self):
        evaluator = COCOEvaluator("faster_val", output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, "faster_val")
        print(inference_on_dataset(self.predictor.train_model, val_loader, evaluator))


def main():
    # Download the dataset
    # URL of the dataset
    url = 'https://fordatis.fraunhofer.de/bitstream/fordatis/355/26/INFRA-3DRC-Dataset.zip'

    # Path where you want to save the dataset
    save_directory = Path('/mnt/c/Users/alhasan/Documents/Python Scripts/detectron/final')
    save_path = save_directory / 'INFRA-3DRC-Dataset.zip'

    # Download the dataset
    download_dataset(url, str(save_path))

    # Convert the dataset to the COCO format
    coco_dir_path = '/mnt/c/Users/alhasan/Documents/Python Scripts/detectron/final/faster_retrain'
    dataset_path = '/mnt/c/Users/alhasan/Documents/Python Scripts/detectron/final/INFRA-3DRC-Dataset'
    output_dir_path = coco_dir_path + "/output"

    converter = COCOConverter(dataset_path, coco_dir_path)
    converter.convert_to_coco()

    # Create a trainer
    # train_model = DetectronModel(coco_dir_path)
    # train_model.create_train_configs(output_dir_path, 25, 5, converter.total_train_img)
    # # Start the training
    # train_model.train()
    # # Get training diagnostics
    # train_model.get_train_diagnostics()

    # Create a predictor
    model_path = "/mnt/c/Users/alhasan/Documents/Python Scripts/detectron/final/model_0010799.pth"
    predict_model = DetectronModel(coco_dir_path)
    predict_model.create_predict_configs(model_path)
    # Start the prediction
    test_images_path = "/mnt/c/Users/alhasan/Documents/Python Scripts/detectron/final/faster_retrain/val/images"
    predict_model.predict(test_images_path, converter.val_coco_annotation_path)
    # Evaluate the prediction results
    predict_model.evaluate_test_dataset()

if __name__ == "__main__":
    main()