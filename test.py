import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

TEST_DATASET_PATH = "./intel-image-classification/seg_test/seg_test"
MODEL_PATH = "best_model_tf17.h5"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

model = load_model(MODEL_PATH)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    TEST_DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse"
)


test_loss, test_accuracy = model.evaluate(test_ds)

print(f"Test Accuracy: {test_accuracy:.2%}")
