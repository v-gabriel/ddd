import logging
import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.class_weight import compute_class_weight

from src.config import ExperimentConfig

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, applications, mixed_precision
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

    TF_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ModelTrainer(ABC):
    def __init__(self, config: ExperimentConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None): pass

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError(f"{self.model_name} not trained.")
        return self.model.predict(X_test)

    @abstractmethod
    def save(self, fold_num=None): pass


# TODO: implement or remove
class DecisionTreeRegressorTrainer(ModelTrainer):
    def __init__(self, config: ExperimentConfig, model_type: str):
        super().__init__(config, model_type)
        self.model = DecisionTreeRegressor(
            max_depth=config.DT_MAX_DEPTH,
            min_samples_split=config.DT_MIN_SAMPLES_SPLIT,
            random_state=config.RANDOM_STATE
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save(self, fold_num=None):
        filename = self.model_name.lower() + (f"_fold_{fold_num}" if fold_num is not None else "")
        filepath = os.path.join(self.config.OUTPUT_DIR, f"{filename}.pkl")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {self.model_name} to {filepath}")


# TODO: implement or remove
class RandomForestTreeRegressorTrainer(ModelTrainer):
    def __init__(self, config: ExperimentConfig, model_type: str):
        super().__init__(config, model_type)
        self.model = RandomForestRegressor(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.DT_MAX_DEPTH,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save(self, fold_num=None):
        filename = self.model_name.lower() + (f"_fold_{fold_num}" if fold_num is not None else "")
        filepath = os.path.join(self.config.OUTPUT_DIR, f"{filename}.pkl")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {self.model_name} to {filepath}")


class SVMTrainer(ModelTrainer):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, 'SVM')

        if config.SVM_TYPE == 'linear':
            self.model = LinearSVC(
                C=config.SVM_C,
                class_weight=config.SVM_CLASS_WEIGHT,
                random_state=config.RANDOM_STATE,
                max_iter=10000,
                dual=False,
                tol=1e-4,
                loss='squared_hinge'
            )
            if config.SETUP_HYPERPARAMETER_TUNING:
                logger.info(f"Using LinearSVC with hyperparameter tuning")
            else:
                logger.info(f"Using LinearSVC with "
                            f"c={self.config.SVM_C}, "
                            f"weights={config.SVM_CLASS_WEIGHT}")
        elif config.SVM_TYPE == 'rbf':
            self.model = SVC(
                C=config.SVM_C,
                gamma=config.SVM_GAMMA,
                kernel='rbf',
                class_weight=config.SVM_CLASS_WEIGHT,
                random_state=config.RANDOM_STATE,
                max_iter=10000,
                probability=False,
                tol=1e-4
            )
            if config.SETUP_HYPERPARAMETER_TUNING:
                logger.info(f"Using SVC kernel with hyperparameter tuning")
            else:
                logger.info(f"Using SVC with "
                            f"kernel={self.model.kernel}, "
                            f"gamma={config.SVM_GAMMA}, "
                            f"c={config.SVM_C}, "
                            f"weights={config.SVM_CLASS_WEIGHT}")
        else:
            raise ImportError("TensorFlow not available for CNNTrainer.")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"SVM Training class distribution: {dict(zip(unique, counts))}")

        if len(unique) > 1:
            class_weights = compute_class_weight('balanced', classes=unique, y=y_train)
            logger.info(f"SVM Computed class weights: {dict(zip(unique, class_weights))}")

            imbalance_ratio = max(counts) / min(counts)
            logger.info(f"SVM Class imbalance ratio: {imbalance_ratio:.2f}")

            if imbalance_ratio > 10:
                logger.warning(f"High class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                               "Consider additional balancing techniques.")
        else:
            logger.error("Only one class found in training data!")
            return self

        self.model.fit(X_train, y_train)

        try:
            relevant_params = {k: v for k, v in self.model.get_params().items()
                               if k in ['C', 'gamma', 'kernel', 'class_weight', 'max_iter']}
            logger.info(f"{type(self.model).__name__} Final parameters: {relevant_params}")
        except Exception as e:
            logger.warning(f"Could not log model parameters: {e}")

        return self

    def save(self, fold_num=None):
        filename = self.model_name.lower() + (f"_fold_{fold_num}" if fold_num is not None else "")
        filepath = os.path.join(self.config.OUTPUT_DIR, f"{filename}.pkl")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {self.model_name} to {filepath}")


class KNNTrainer(ModelTrainer):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, 'KNN')

        self.model = KNeighborsClassifier(
            n_neighbors=config.KNN_NEIGHBORS,
            weights=config.KNN_WEIGHTS,
            algorithm=config.KNN_ALGORITHM,
            metric=config.KNN_METRIC,
            leaf_size=50,
            n_jobs=-1
        )
        if config.SETUP_HYPERPARAMETER_TUNING:
            logger.info(f"KNN initialized with hyperparameter tuning.")
        else:
            logger.info(f"KNN initialized with "
                        f"k={config.KNN_NEIGHBORS},"
                        f" weights={config.KNN_WEIGHTS},"
                        f" metric={config.KNN_METRIC},"
                        f" algorithm={config.KNN_ALGORITHM}")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"KNN Training class distribution: {dict(zip(unique, counts))}")

        if len(unique) > 1:
            imbalance_ratio = max(counts) / min(counts)
            logger.info(f"KNN Class imbalance ratio: {imbalance_ratio:.2f}")

            if imbalance_ratio > 5:
                logger.warning(f"KNN may struggle with class imbalance (ratio: {imbalance_ratio:.2f}).")

        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]
        if n_features > n_samples / 2:
            logger.warning(f"KNN: High dimensionality ({n_features} features, {n_samples} samples). "
                           "Consider feature selection.")

        self.model.fit(X_train, y_train)
        logger.info(f"KNN trained with {self.model.n_neighbors} neighbors")
        return self

    def save(self, fold_num=None):
        filename = self.model_name.lower() + (f"_fold_{fold_num}" if fold_num is not None else "")
        filepath = os.path.join(self.config.OUTPUT_DIR, f"{filename}.pkl")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved {self.model_name} to {filepath}")


class CNNTrainer(ModelTrainer):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config, f'{config.CNN_MODEL_TYPE.upper()}')
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for CNNTrainer.")

        mixed_precision.set_global_policy('mixed_float16')
        self.history = None
        self.base_model = None

    def build_cnn_model(self, input_shape, num_classes):
        logger.info(f"Building optimized CNN model: {self.config.CNN_MODEL_TYPE}")

        if len(input_shape) == 2:
            corrected_input_shape = (*input_shape, 1)
        else:
            corrected_input_shape = input_shape

        input_shape = corrected_input_shape
        inputs = layers.Input(shape=input_shape)

        # strict overfitting fix: try to force generalization
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.08)(x)
        x = layers.RandomZoom(0.08)(x)
        x = layers.RandomBrightness(0.1)(x)

        model_configs = {
            'mobilenetv2': {
                'base_class': applications.MobileNetV2,
                'preprocessing': applications.mobilenet_v2.preprocess_input
            },
            'efficientnetv2': {
                'base_class': applications.EfficientNetV2B0,
                'preprocessing': applications.efficientnet_v2.preprocess_input
            },
            'resnet50v2': {
                'base_class': applications.ResNet50V2,
                'preprocessing': applications.resnet_v2.preprocess_input
            }
        }

        if self.config.CNN_MODEL_TYPE in model_configs:
            config_info = model_configs[self.config.CNN_MODEL_TYPE]

            if input_shape[-1] == 1:
                logger.info("Adapting 1-channel grayscale input to 3 channels.")
                x = layers.Concatenate(axis=-1)([x, x, x])
            elif input_shape[-1] > 3:  # Handles 9-channel stacked ROIs, etc.
                logger.info(f"Adapting {input_shape[-1]}-channel input to 3 channels with a Conv2D adapter.")
                x = layers.Conv2D(3, (1, 1), padding='same', name='input_adapter')(x)
            x = config_info['preprocessing'](x)

            base_model = config_info['base_class'](
                input_shape=(input_shape[0], input_shape[1], 3),
                include_top=False,
                weights='imagenet'
            )
            self.base_model = base_model
            base_model.trainable = False

            x = base_model(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)

            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
        else:
            logger.info(f"Using custom CNN. {self.config.CNN_MODEL_TYPE} not found in {model_configs}")

            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)

            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)

            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
        return models.Model(inputs, outputs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y_train))

        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"CNN Training class distribution: {dict(zip(unique, counts))}")

        self.model = self.build_cnn_model(input_shape, num_classes)

        initial_lr = 0.001
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            LearningRateScheduler(
                lambda epoch: initial_lr * (0.9 ** (epoch // 10)),
                verbose=1
            )
        ]

        weights = compute_class_weight('balanced', classes=unique, y=y_train)
        weights = dict(enumerate(weights))
        if len(weights) == 2:
            weights[1] = weights[1] # TODO: consider (x2) to add extra penalty for missing drowsiness

        logger.info(f"CNN class weights: {weights}")

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.CNN_EPOCHS,
            batch_size=self.config.CNN_BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=weights,
            verbose=1
        )

        if self.base_model and self.config.CNN_FINE_TUNE:
            logger.info("Starting safety-optimized fine-tuning")
            self.base_model.trainable = True

            for layer in self.base_model.layers[:-30]:
                layer.trainable = False

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy']
            )

            self.model.fit(
                X_train, y_train,
                epochs=self.config.CNN_EPOCHS,
                batch_size=self.config.CNN_BATCH_SIZE,
                validation_split=0.2,
                callbacks=callbacks,
                class_weight=weights,
                verbose=1
            )

        return self

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError(f"{self.model_name} not trained.")
        predictions = self.model.predict(X_test, batch_size=self.config.CNN_BATCH_SIZE * 2)
        return np.argmax(predictions, axis=1)

    def save(self, fold_num=None):
        filename = self.model_name.lower() + (f"_fold_{fold_num}" if fold_num is not None else "")
        filepath = os.path.join(self.config.OUTPUT_DIR, f"{filename}.keras")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        self.model.save(filepath, save_format='keras')
        logger.info(f"Saved {self.model_name} to {filepath}")


class LSTMTrainer(ModelTrainer):

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, 'LSTM')
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTMTrainer.")

        # Enable mixed precision for speedup on compatible GPUs
        mixed_precision.set_global_policy('mixed_float16')
        self.history = None

    def build_simplified_lstm_model(self, input_shape, num_classes):
        logger.info("Building Simplified LSTM model with enhanced regularization")
        inputs = layers.Input(shape=input_shape, name='input_sequences')
        x = layers.Masking(mask_value=0.0, name='masking')(inputs)

        x = layers.LSTM(
            64,
            return_sequences=False,
            name='lstm_layer',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)

        x = layers.Dense(64, name='dense1', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  #
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Activation('relu')(x)

        x = layers.Dropout(0.5, name='dropout1')(x)

        outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)
        return models.Model(inputs, outputs, name='SimplifiedLSTM_Regularized')

    # TODO: test, use?
    def build_hierarchical_lstm_model(self, input_shape, num_classes):

        logger.info("Building Hierarchical LSTM model with enhanced regularization")
        inputs = layers.Input(shape=input_shape, name='input_sequences')
        x = layers.Masking(mask_value=0.0, name='masking')(inputs)

        x = layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            name='short_term_lstm',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = layers.BatchNormalization(name='bn1')(x)

        x = layers.LSTM(
            96,
            return_sequences=True,
            dropout=0.3,  # Povećan dropout
            name='medium_term_lstm',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = layers.BatchNormalization(name='bn2')(x)

        x = layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.3,
            name='long_term_lstm',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 Regularizacija
        )(x)
        x = layers.BatchNormalization(name='bn3')(x)

        x = layers.Dense(128, activation='relu', name='dense1', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5, name='dropout1')(x)  # Povećano s 0.4 na 0.5

        x = layers.Dense(64, activation='relu', name='dense2', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.Dropout(0.4, name='dropout2')(x)  # Povećano s 0.3 na 0.4

        outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)
        return models.Model(inputs, outputs, name='HierarchicalLSTM_Regularized')

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        input_shape = X_train.shape[1:]
        num_classes = len(np.unique(y_train))

        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"LSTM Training class distribution: {dict(zip(unique, counts))}")

        model_type = getattr(self.config, 'LSTM_MODEL_TYPE', 'simplified')
        if model_type == 'hierarchical':
            self.model = self.build_hierarchical_lstm_model(input_shape, num_classes)
        else:
            self.model = self.build_simplified_lstm_model(input_shape, num_classes)

        self.model.summary(print_fn=logger.info)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                min_delta=0.001,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        weights = compute_class_weight('balanced', classes=unique, y=y_train)
        class_weights = dict(enumerate(weights))
        logger.info(f"LSTM Class weights: {class_weights}")

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        if validation_data is None:
            logger.warning("No validation data.")

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config.CNN_EPOCHS,
            batch_size=self.config.CNN_BATCH_SIZE,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return self

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError(f"{self.model_name} not trained.")

        # Use a larger batch size for prediction for speed
        predictions = self.model.predict(X_test, batch_size=self.config.CNN_BATCH_SIZE * 2, verbose=0)

        # Return class labels
        return np.argmax(predictions, axis=1)

    def save(self, fold_num=None):
        model_arch = getattr(self.config, 'LSTM_MODEL_TYPE', 'simplified')
        filename = f"{self.model_name.lower()}_{model_arch}" + (f"_fold_{fold_num}" if fold_num is not None else "")
        filepath = os.path.join(self.config.OUTPUT_DIR, f"{filename}.keras")
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        self.model.save(filepath, save_format='keras')
        logger.info(f"Saved {self.model_name} model to {filepath}")


def get_model_trainer(model_type: str, config: ExperimentConfig) -> ModelTrainer:
    trainers = {
        'svm': SVMTrainer,
        'svm_deep': SVMTrainer,
        'knn': KNNTrainer,
        'cnn': CNNTrainer,
        'lstm': LSTMTrainer,
        # 'random_forest_reg': RandomForestTreeRegressorTrainer,
        # 'decision_tree_reg': DecisionTreeRegressorTrainer,
    }

    if model_type not in trainers:
        raise ValueError(f"Unsupported model type: {model_type}")

    return trainers[model_type](config)
