from __future__ import annotations
import tensorflow as tf

def focal_loss(alpha: float = 0.75, gamma: float = 2.0):
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true_f, 1.0), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(tf.equal(y_true_f, 1.0), alpha, 1.0 - alpha)
        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss

def build_model(window_size: int, n_features: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_size, n_features), name="sensor_window")
    x = tf.keras.layers.Conv1D(64, 5, padding="causal", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48, return_sequences=True))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="failure_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="predictive_maintenance_cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss(),
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )
    return model
