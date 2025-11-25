import tensorflow as tf
from util import load_and_preprocess_data
from model import create_model_1, create_model_2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Create directory for saving charts if it doesn't exist
CHARTS_DIR = os.path.join(".", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def plot_history(history, model_name):
    """Plots the training and validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.plot(history.history["accuracy"], label="Training Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Training and Validation Accuracy for {model_name}")
    ax1.legend()

    ax2.plot(history.history["loss"], label="Training Loss")
    ax2.plot(history.history["val_loss"], label="Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Training and Validation Loss for {model_name}")
    ax2.legend()

    plt.tight_layout()

    # Save figure to charts directory
    filename = os.path.join(
        CHARTS_DIR, f"{model_name.replace(' ', '_')}_training_history.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved training history plot to {filename}")

    plt.show()


def evaluate_model(model, ds_test, ds_info, model_name):
    """Evaluates the model and returns detailed metrics."""
    print(f"\n--- Evaluating {model_name} ---")

    # Regular evaluation
    test_loss, test_acc = model.evaluate(ds_test)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")

    # Top-5 accuracy calculation
    top5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")
    for images, labels in ds_test:
        predictions = model.predict(images)
        top5_acc.update_state(labels, predictions)

    print(f"{model_name} Top-5 Accuracy: {top5_acc.result().numpy():.4f}")

    # Get predictions
    y_pred_probs = model.predict(ds_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([y for x, y in ds_test], axis=0)
    y_true = np.argmax(y_true, axis=1)

    # Confusion matrix and class-wise accuracy
    class_names = ds_info.features["label"].names
    confusion_mtx = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        confusion_mtx,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=["" for _ in range(len(class_names))],
        yticklabels=["" for _ in range(len(class_names))],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.tight_layout()

    # Save confusion matrix to charts directory
    conf_matrix_filename = os.path.join(
        CHARTS_DIR, f"{model_name.replace(' ', '_')}_confusion_matrix.png"
    )
    plt.savefig(conf_matrix_filename, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix to {conf_matrix_filename}")

    plt.show()

    # Class-wise accuracy
    class_accuracy = np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    top_5_classes = np.argsort(class_accuracy)[-5:][::-1]
    worst_5_classes = np.argsort(class_accuracy)[:5]

    print("\nTop-5 best recognized classes:")
    for i in top_5_classes:
        print(f"{class_names[i]}: {class_accuracy[i]:.4f}")

    print("\nWorst-5 recognized classes:")
    for i in worst_5_classes:
        print(f"{class_names[i]}: {class_accuracy[i]:.4f}")

    # Plot and save class accuracies
    plt.figure(figsize=(12, 8))
    # Top 5 classes
    plt.subplot(2, 1, 1)
    plt.bar(range(len(top_5_classes)), [class_accuracy[i] for i in top_5_classes])
    plt.xticks(
        range(len(top_5_classes)), [class_names[i] for i in top_5_classes], rotation=45
    )
    plt.title(f"Top 5 Best Recognized Classes - {model_name}")
    plt.ylabel("Accuracy")

    # Worst 5 classes
    plt.subplot(2, 1, 2)
    plt.bar(range(len(worst_5_classes)), [class_accuracy[i] for i in worst_5_classes])
    plt.xticks(
        range(len(worst_5_classes)),
        [class_names[i] for i in worst_5_classes],
        rotation=45,
    )
    plt.title(f"Worst 5 Recognized Classes - {model_name}")
    plt.ylabel("Accuracy")

    plt.tight_layout()

    # Save class accuracy plot
    class_acc_filename = os.path.join(
        CHARTS_DIR, f"{model_name.replace(' ', '_')}_class_accuracy.png"
    )
    plt.savefig(class_acc_filename, dpi=300, bbox_inches="tight")
    print(f"Saved class accuracy plot to {class_acc_filename}")

    plt.show()

    return test_acc, top5_acc.result().numpy()


def train_model(
    model, ds_train, ds_validation, model_name, optimizer, epochs=200, patience=10
):
    print(f"\n--- Training {model_name} with {optimizer.__class__.__name__} ---")

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )

    # Train the model
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_validation,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    plot_history(history, f"{model_name}")
    return model, history


def main():
    """Main function to run the training and evaluation."""
    # Load data
    ds_train, ds_validation, ds_test, ds_info = load_and_preprocess_data()
    input_shape = ds_info.features["image"].shape
    num_classes = ds_info.features["label"].num_classes

    results = []

    # --- Experiment with Model 1 - Configuration 1 ---
    model1_1 = create_model_1(input_shape, num_classes)
    optimizer1_1 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model1_1, history1_1 = train_model(
        model1_1, ds_train, ds_validation, "Model 1 - Adam(lr=1e-3)", optimizer1_1
    )
    acc1_1, top5_acc1_1 = evaluate_model(
        model1_1, ds_test, ds_info, "Model 1 - Config 1"
    )
    results.append(("Model 1 - Adam(lr=1e-3)", acc1_1, top5_acc1_1))

    # --- Experiment with Model 1 - Configuration 2 ---
    model1_2 = create_model_1(input_shape, num_classes)
    optimizer1_2 = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model1_2, history1_2 = train_model(
        model1_2, ds_train, ds_validation, "Model 1 - Adam(lr=5e-4)", optimizer1_2
    )
    acc1_2, top5_acc1_2 = evaluate_model(
        model1_2, ds_test, ds_info, "Model 1 - Config 2"
    )
    results.append(("Model 1 - Adam(lr=5e-4)", acc1_2, top5_acc1_2))

    # Model 1 with SGD
    model1_3 = create_model_1(input_shape, num_classes)
    optimizer1_3 = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model1_3, history1_3 = train_model(
        model1_3, ds_train, ds_validation, "Model 1 - SGD(lr=0.01)", optimizer1_3
    )
    acc1_3, top5_acc1_3 = evaluate_model(
        model1_3, ds_test, ds_info, "Model 1 - Config 3"
    )
    results.append(("Model 1 - SGD(lr=0.01)", acc1_3, top5_acc1_3))

    # Now for something completely different - RMSprop on Model 1
    model1_4 = create_model_1(input_shape, num_classes)
    optimizer1_4 = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
    model1_4, history1_4 = train_model(
        model1_4, ds_train, ds_validation, "Model 1 - RMSprop(lr=1e-4)", optimizer1_4
    )
    acc1_4, top5_acc1_4 = evaluate_model(
        model1_4, ds_test, ds_info, "Model 1 - Config 4"
    )
    results.append(("Model 1 - RMSprop(lr=1e-4)", acc1_4, top5_acc1_4))

    # model2 with SGD
    model2_1 = create_model_2(input_shape, num_classes)
    optimizer2_1 = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model2_1, history2_1 = train_model(
        model2_1, ds_train, ds_validation, "Model 2 - SGD(lr=0.01)", optimizer2_1
    )
    acc2_1, top5_acc2_1 = evaluate_model(
        model2_1, ds_test, ds_info, "Model 2 - Config 1"
    )
    results.append(("Model 2 - SGD(lr=0.01)", acc2_1, top5_acc2_1))

    # Model 2 - Configuration 2
    model2_2 = create_model_2(input_shape, num_classes)
    optimizer2_2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model2_2, history2_2 = train_model(
        model2_2, ds_train, ds_validation, "Model 2 - Adam(lr=1e-4)", optimizer2_2
    )
    acc2_2, top5_acc2_2 = evaluate_model(
        model2_2, ds_test, ds_info, "Model 2 - Config 2"
    )
    results.append(("Model 2 - Adam(lr=1e-4)", acc2_2, top5_acc2_2))

    # Model 2 - Configuration 3
    model2_3 = create_model_2(input_shape, num_classes)
    optimizer2_3 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model2_3, history2_3 = train_model(
        model2_3, ds_train, ds_validation, "Model 2 - Adam(lr=1e-3)", optimizer2_3
    )
    acc2_3, top5_acc2_3 = evaluate_model(
        model2_3, ds_test, ds_info, "Model 2 - Config 3"
    )
    results.append(("Model 2 - Adam(lr=1e-3)", acc2_3, top5_acc2_3))

    # Experiment with Model 2 - Configuration 4 
    model2_4 = create_model_2(input_shape, num_classes)
    optimizer2_4 = tf.keras.optimizers.RMSprop(learning_rate=5e-4)
    model2_4, history2_4 = train_model(
        model2_4, ds_train, ds_validation, "Model 2 - RMSprop(lr=5e-4)", optimizer2_4
    )
    acc2_4, top5_acc2_4 = evaluate_model(
        model2_4, ds_test, ds_info, "Model 2 - Config 4"
    )
    results.append(("Model 2 - RMSprop(lr=5e-4)", acc2_4, top5_acc2_4))

    # Compare all results
    print("\n--- Overall Results Comparison ---")
    print("Model Configuration | Test Accuracy | Top-5 Accuracy")
    print("-" * 60)
    for name, acc, top5_acc in results:
        print(f"{name:<22} | {acc:.4f}       | {top5_acc:.4f}")

    # Find best model based on test accuracy
    best_idx = np.argmax([acc for _, acc, _ in results])
    best_model_name = results[best_idx][0]
    print(
        f"\nBest model based on test accuracy: {best_model_name} with {results[best_idx][1]:.4f} accuracy"
    )

    # Find best model based on top-5 accuracy
    best_top5_idx = np.argmax([top5_acc for _, _, top5_acc in results])
    best_top5_model_name = results[best_top5_idx][0]
    print(
        f"Best model based on top-5 accuracy: {best_top5_model_name} with {results[best_top5_idx][2]:.4f} top-5 accuracy"
    )

    # Create and save comparison chart - update for 8 configurations
    plt.figure(figsize=(14, 8))

    model_names = [name for name, _, _ in results]
    accuracies = [acc for _, acc, _ in results]
    top5_accuracies = [top5 for _, _, top5 in results]

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width / 2, accuracies, width, label="Top-1 Accuracy")
    plt.bar(x + width / 2, top5_accuracies, width, label="Top-5 Accuracy")

    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.xticks(
        x,
        [name.replace("Model ", "M").replace("Config", "C") for name in model_names],
        rotation=45,
    )
    plt.legend()

    plt.tight_layout()

    # Save comparison chart
    comparison_filename = os.path.join(CHARTS_DIR, "model_comparison.png")
    plt.savefig(comparison_filename, dpi=300, bbox_inches="tight")
    print(f"Saved model comparison chart to {comparison_filename}")

    plt.show()

    # Create separate charts for Model 1 and Model 2 for better readability
    # Model 1 comparison
    plt.figure(figsize=(10, 6))
    model1_indices = [i for i, name in enumerate(model_names) if "Model 1" in name]
    m1_names = [model_names[i] for i in model1_indices]
    m1_acc = [accuracies[i] for i in model1_indices]
    m1_top5 = [top5_accuracies[i] for i in model1_indices]

    x = np.arange(len(m1_names))

    plt.bar(x - width / 2, m1_acc, width, label="Top-1 Accuracy")
    plt.bar(x + width / 2, m1_top5, width, label="Top-5 Accuracy")

    plt.ylabel("Accuracy")
    plt.title("Model 1 - Configuration Comparison")
    plt.xticks(x, [name.replace("Model 1 - ", "") for name in m1_names], rotation=45)
    plt.legend()
    plt.tight_layout()

    m1_comparison = os.path.join(CHARTS_DIR, "model1_comparison.png")
    plt.savefig(m1_comparison, dpi=300, bbox_inches="tight")
    print(f"Saved Model 1 comparison chart to {m1_comparison}")

    plt.show()

    # Model 2 comparison
    plt.figure(figsize=(10, 6))
    model2_indices = [i for i, name in enumerate(model_names) if "Model 2" in name]
    m2_names = [model_names[i] for i in model2_indices]
    m2_acc = [accuracies[i] for i in model2_indices]
    m2_top5 = [top5_accuracies[i] for i in model2_indices]

    x = np.arange(len(m2_names))

    plt.bar(x - width / 2, m2_acc, width, label="Top-1 Accuracy")
    plt.bar(x + width / 2, m2_top5, width, label="Top-5 Accuracy")

    plt.ylabel("Accuracy")
    plt.title("Model 2 - Configuration Comparison")
    plt.xticks(x, [name.replace("Model 2 - ", "") for name in m2_names], rotation=45)
    plt.legend()
    plt.tight_layout()

    m2_comparison = os.path.join(CHARTS_DIR, "model2_comparison.png")
    plt.savefig(m2_comparison, dpi=300, bbox_inches="tight")
    print(f"Saved Model 2 comparison chart to {m2_comparison}")

    plt.show()

    # Save results to CSV for easy reporting
    import csv

    csv_filename = os.path.join(CHARTS_DIR, "model_results.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model", "Test Accuracy", "Top-5 Accuracy"])
        for name, acc, top5_acc in results:
            writer.writerow([name, f"{acc:.4f}", f"{top5_acc:.4f}"])

    print(f"Saved model results to {csv_filename}")


if __name__ == "__main__":
    main()