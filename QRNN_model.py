# dqn_offline_train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import os
from data_reader import load_winners_dataset
tf.random.set_seed(0)
np.random.seed(0)

action_map = [
    (1, -1), (1, 1),
    (0, 0),
    (-1, -1), (-1, 1)
]

def encode_action(pair):
    return action_map.index(tuple(pair))

def build_q_model(state_shape, look_back=3, n_actions=5, hidden_units=(5, 5), learning_rate=1e-3):
    state_dim = np.prod(state_shape)
    inp = layers.Input(shape=(look_back, state_dim))
    x = inp

    for i, u in enumerate(hidden_units):
        return_seq = i < len(hidden_units) - 1
        x = layers.SimpleRNN(u, activation="relu", return_sequences=return_seq)(x)

    out = layers.Dense(n_actions, activation="linear")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                  loss=losses.MeanSquaredError())
    return model



def prepare_arrays(states, actions, rewards, next_states, dones):
    """
    Ensure numpy arrays and shapes are correct.
    Expects:
      - states, next_states : numpy arrays with same leading dim
      - actions, rewards, dones : 1D arrays of length N
    """
    states = np.asarray(states)
    next_states = np.asarray(next_states)
    actions = np.asarray(actions, dtype=np.int32)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)  # 1.0 if done else 0.0

    assert states.shape[0] == actions.shape[0] == rewards.shape[0] == next_states.shape[0] == dones.shape[0], \
        "All arrays must have same first dimension"
    return states, actions, rewards, next_states, dones


def train_from_dataset(model,
                       states,
                       actions,
                       rewards,
                       next_states,
                       dones,
                       gamma=0.9,
                       batch_size=64,
                       epochs=10,
                       shuffle=True,
                       verbose=1,
                       target_network=None,
                       target_update_freq=5):
    """
    Offline training loop using the collected dataset.
    If target_network is provided, uses that to compute target Q-values for stability.
    Training updates the model's Q-value predictions to match:
      target_for_action = reward + (1 - done) * gamma * max_a Q_next(next_state, a)
    Implementation details:
    - Compute model.predict(states) as base target matrix (this keeps untouched Q-values for other actions).
    - For the action actually taken, replace with computed target scalar.
    """

    # Arrays are already prepared & actions are already encoded
    states, actions, rewards, next_states, dones = prepare_arrays(
        states, actions, rewards, next_states, dones
    )

    n_samples = states.shape[0]
    n_actions = model.output_shape[-1]

    # If target network provided, use it; else use model for bootstrapping
    target_model = target_network if target_network is not None else model

    # Precompute next-state Q-values in batches to avoid mem blow-ups
    def compute_max_next_q(arr):
        return np.max(target_model.predict(arr, verbose=0), axis=1)

    # Training
    for epoch in range(epochs):
        if shuffle:
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            states_sh = states[idx]
            actions_sh = actions[idx]
            rewards_sh = rewards[idx]
            next_states_sh = next_states[idx]
            dones_sh = dones[idx]
        else:
            states_sh = states
            actions_sh = actions
            rewards_sh = rewards
            next_states_sh = next_states
            dones_sh = dones

        losses = []
        # Mini-batch training
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            s_batch = states_sh[start:end]
            a_batch = actions_sh[start:end]
            r_batch = rewards_sh[start:end]
            ns_batch = next_states_sh[start:end]
            d_batch = dones_sh[start:end]

            actual_batch_size = len(s_batch)

            # Current Q predictions (shape: actual_batch_size x n_actions)
            q_preds = model.predict(s_batch, verbose=0)

            # Compute max Q for next states using the target model
            max_next_q = compute_max_next_q(ns_batch)

            # Build target for the taken actions
            # target_q[action_index] = reward + (1 - done) * gamma * max_next_q
            targets = q_preds.copy()
            # vectorized update:
            targets[np.arange(actual_batch_size), a_batch] = r_batch + (1.0 - d_batch) * gamma * max_next_q


            # Train one step
            history = model.fit(s_batch, targets, epochs=1, batch_size=targets.shape[0], verbose=0)
            losses.append(history.history['loss'][0])

        # Optionally update target network
        if target_network is not None and (epoch + 1) % target_update_freq == 0:
            target_network.set_weights(model.get_weights())

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} — loss: {np.mean(losses):.6f}")

    return model


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"Saved model to {path}")


def load_model(path):
    return tf.keras.models.load_model(path)


# Example usage (edit to your filenames / arrays):
if __name__ == "__main__":
    # Load dataset
    states, actions, rewards, next_states, dones = load_winners_dataset("Alex_data")
    
    # Convert raw motor-pair actions to indices
    actions = np.array([encode_action(a) for a in actions], dtype=np.int32)
    
    # Prepare arrays
    states, actions, rewards, next_states, dones = prepare_arrays(
        states, actions, rewards, next_states, dones
    )
    
    n_samples = states.shape[0]
    n_actions = len(action_map)   # ✅ instead of model.output_shape
    
    # Build Q-model
    example_state_shape = (3,)  # because you reduced 5 → 3 sensors
    if './model_savings/QRNN.keras':
        q_model = load_model("./model_savings/QRNN.keras")
        target_q = load_model("./model_savings/QRNN.keras")
        target_q.set_weights(q_model.get_weights())
    else:
        q_model = build_q_model(state_shape=example_state_shape,
                                    n_actions=n_actions,
                                    hidden_units=(5, 5),
                                    learning_rate=1e-3)
        # Target network (recommended)
        target_q = build_q_model(state_shape=example_state_shape,
                                 n_actions=n_actions,
                                 hidden_units=(5, 5),
                                 learning_rate=1e-3)
        target_q.set_weights(q_model.get_weights())

    # Train offline
    q_model = train_from_dataset(q_model,
                                 states,
                                 actions,
                                 rewards,
                                 next_states,
                                 dones,
                                 gamma=0.9,
                                 batch_size=32,
                                 epochs=1,
                                 verbose=1,
                                 target_network=target_q,
                                 target_update_freq=2)

    # Save
    save_model(q_model, "./model_savings/QRNN.keras")
