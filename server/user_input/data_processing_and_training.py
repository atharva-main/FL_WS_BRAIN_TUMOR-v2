from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def data_processing(dataset_path):

    target_size = (64, 64)
    batch_size = 32
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Normalize pixel values to [0,1]
        rotation_range=20,         # Random rotation
        width_shift_range=0.2,     # Horizontal shift
        height_shift_range=0.2,    # Vertical shift
        shear_range=0.2,           # Shearing
        zoom_range=0.2,            # Random zoom
        horizontal_flip=True,      # Flip images horizontally
        fill_mode='nearest'        # Fill mode for image resizing
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        f'{dataset_path}/Train',    # Path to the training dataset directory
        target_size=target_size,    # Resize images to target_size
        batch_size=batch_size,      # Batch size for training
        class_mode='categorical'    # Use 'binary' for binary classification, 'categorical' for more than 2 classes
    )

    test_generator = test_datagen.flow_from_directory(
        f'{dataset_path}/Test',     # Path to the testing dataset directory
        target_size=target_size,    # Resize images to target_size
        batch_size=batch_size,      # Batch size for testing
        class_mode='categorical'    # Use 'binary' for binary classification, 'categorical' for more than 2 classes
    )
    
    return train_generator, test_generator

def train_model(model, train_generator, test_generator):

    epochs = 3
    history = model.fit(
        train_generator,                          # Training data
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=test_generator,           # Validation data
        validation_steps=test_generator.samples // test_generator.batch_size,
        epochs=epochs
    )
    
    return history


