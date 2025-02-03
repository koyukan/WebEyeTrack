import tensorflow as tf

# Define hyperparameters
meta_epochs = 100  # Number of meta-training iterations
tasks_per_batch = 4  # Number of user-specific tasks per meta-update

def maml_inner_loop(model, x_support, y_support, learning_rate=1e-3):
    """
    Performs one inner-loop update on a single task.
    
    Args:
        model: The BlazeGaze model.
        x_support: Small support set (user-specific training data).
        y_support: Corresponding gaze vectors.
        learning_rate: Learning rate for inner-loop update.

    Returns:
        Adapted model parameters after one task update.
    """
    with tf.GradientTape() as tape:
        y_pred = model(x_support, training=True)
        loss = angular_loss(y_support, y_pred)  # Compute gaze loss

    # Compute gradients w.r.t. model parameters
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Apply task-specific gradients (inner update)
    updated_params = [w - learning_rate * g for w, g in zip(model.trainable_variables, grads)]
    
    return updated_params

def maml_outer_loop(model, tasks, meta_learning_rate=1e-3, inner_learning_rate=1e-3):
    """
    Performs the outer-loop update for MAML.
    
    Args:
        model: The BlazeGaze model.
        tasks: List of tasks, each with (support_set, query_set).
        meta_learning_rate: Learning rate for meta-update.
        inner_learning_rate: Learning rate for inner updates.

    Returns:
        Meta-optimized model parameters.
    """
    meta_grads = [tf.zeros_like(w) for w in model.trainable_variables]

    for task in tasks:
        x_support, y_support = task["support_set"]
        x_query, y_query = task["query_set"]

        # Perform inner-loop adaptation
        adapted_params = maml_inner_loop(model, x_support, y_support, inner_learning_rate)

        # Compute loss on the query set using adapted parameters
        with tf.GradientTape() as meta_tape:
            y_pred = model(x_query, training=True)
            meta_loss = angular_loss(y_query, y_pred)

        # Compute gradients w.r.t. the original model parameters
        task_grads = meta_tape.gradient(meta_loss, model.trainable_variables)

        # Accumulate gradients across tasks
        meta_grads = [m + t for m, t in zip(meta_grads, task_grads)]

    # Apply meta-updates to the original model
    optimizer = tf.keras.optimizers.Adam(meta_learning_rate)
    optimizer.apply_gradients(zip(meta_grads, model.trainable_variables))

    return model

# Train the model using MAML
for epoch in range(meta_epochs):
    task_batch = sample_tasks(tasks_per_batch)  # Sample a batch of user tasks
    model = maml_outer_loop(model, task_batch, meta_learning_rate=1e-4, inner_learning_rate=1e-3)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Meta-learning step completed.")
