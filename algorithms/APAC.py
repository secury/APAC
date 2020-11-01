import pickle
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from evaluate import evaluate_policy


class ReplayBuffer:

    def __init__(self, max_action, buffer_size):
        self.obs = [None] * buffer_size
        self.action = [None] * buffer_size
        self.reward = [None] * buffer_size
        self.next_obs = [None] * buffer_size
        self.done = [None] * buffer_size
        self.max_action = max_action
        self.N = 0
        self.i = 0
        self.buffer_size = buffer_size

    def __len__(self):
        return self.N

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.i] = obs
        self.action[self.i] = action
        self.reward[self.i] = reward
        self.next_obs[self.i] = next_obs
        self.done[self.i] = done
        self.N = self.N + 1 if self.N < self.buffer_size else self.buffer_size
        self.i = (self.i + 1) % self.buffer_size

    def can_sample(self, batch_size):
        return self.N >= batch_size

    def sample(self, batch_size):
        """
        Return samples (action is normalized)
        """
        obs, action, reward, next_obs, done = [], [], [], [], []
        for _ in range(batch_size):
            idx = np.random.randint(1, self.N)
            obs.append(self.obs[idx])
            action.append(self.action[idx] / self.max_action)
            reward.append(self.reward[idx])
            next_obs.append(self.next_obs[idx])
            done.append(self.done[idx])
        return np.array(obs), np.array(action), np.array(reward)[:, None], np.array(next_obs), np.array(done)[:, None]


def apply_squashing_func(sample, logp):
    """
    Squash the ouput of the gaussian distribution and account for that in the log probability.

    :param sample: (tf.Tensor) Action sampled from Gaussian distribution
    :param logp: (tf.Tensor) Log probability before squashing
    """
    # Squash the output
    squashed_action = tf.tanh(sample)
    squashed_action_logp = logp - tf.reduce_sum(tf.log(1 - squashed_action ** 2 + 1e-6), axis=1, keepdims=True)  # incurred by change of variable
    return squashed_action, squashed_action_logp


class Actor(tf.keras.layers.Layer):

    def __init__(self, c, action_dim, hidden_dim=100):
        super(Actor, self).__init__()
        self.c = c
        self.lcm_c = np.lcm.reduce(c)
        self.action_dim = action_dim

        # Actor parameters
        self.l0 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='actor/f0')
        self.l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='actor/f1')
        self.l2_mu = tf.keras.layers.Dense(self.lcm_c * action_dim, name='actor/f2_mu')
        self.l2_log_std = tf.keras.layers.Dense(self.lcm_c * action_dim, name='actor/f2_log_std')
        self.l2_reshape = tf.keras.layers.Reshape((self.lcm_c, action_dim), name='actor/reshape')

    def call(self, inputs, **kwargs):
        timestep, action_old, obs = inputs
        h = self.l0(tf.concat([action_old, obs], axis=1))
        h = self.l1(h)
        means = self.l2_reshape(self.l2_mu(h))  # batch x lcm_c x action_dim
        log_stds = self.l2_reshape(self.l2_log_std(h))

        mean = tf.gather_nd(means, tf.stack([tf.range(tf.shape(obs)[0]), timestep], axis=1))
        log_std = tf.gather_nd(log_stds, tf.stack([tf.range(tf.shape(obs)[0]), timestep], axis=1))
        std = tf.math.softplus(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        dist.shape = mean.shape

        sampled_action = dist.sample()
        sampled_action_logp = dist.log_prob(sampled_action)[:, None]
        squahsed_action, squahsed_action_logp = apply_squashing_func(sampled_action, sampled_action_logp)

        return squahsed_action, squahsed_action_logp, dist, sampled_action_logp, mean, log_std


class QNetwork(tf.keras.layers.Layer):

    def __init__(self, c, hidden_dim=100, num_critics=2):
        super(QNetwork, self).__init__()
        self.c = c
        self.lcm_c = np.lcm.reduce(c)
        self.num_critics = num_critics

        self.qs_l0, self.qs_l1, self.qs_l2 = [], [], []
        for i in range(self.num_critics):
            self.qs_l0.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name='q%d/f0' % i))
            self.qs_l1.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name='q%d/f1' % i))
            self.qs_l2.append(tf.keras.layers.Dense(self.lcm_c * 1, name='q%d/f2' % i))
        self.l2_reshape = tf.keras.layers.Reshape((self.lcm_c, 1), name='q/reshape')

    def call(self, inputs, **kwargs):
        timestep, obs, action = inputs
        batch_size = tf.shape(obs)[0]
        obs_action = tf.concat([obs, action], axis=1)
        qs = []
        for i in range(self.num_critics):
            h = self.qs_l0[i](obs_action)
            h = self.qs_l1[i](h)
            q = self.l2_reshape(self.qs_l2[i](h))  # batch_size x lcm_c x 1

            q = tf.gather_nd(q, tf.stack([tf.range(batch_size), timestep], axis=1))  # batch_size x 1

            qs.append(q)

        return qs


class APAC(tf.keras.layers.Layer):

    def __init__(self, env, c, ent_coef='auto', seed=0, hidden_dim=100):
        super(APAC, self).__init__()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.env = env
        self.max_action = self.env.action_space.high[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.c = np.array(c)  # action_dim
        self.lcm_c = int(np.lcm.reduce(self.c))
        timestep_mask = np.array(np.arange(self.lcm_c)[:, None] % self.c == 0, dtype=np.float)

        self.timestep_ph = tf.keras.layers.Input((), name='timestep', dtype=tf.int32)  # batch_size
        self.obs_ph = tf.keras.layers.Input(self.state_dim, name='obs')
        self.action_ph = tf.keras.layers.Input(self.action_dim, name='action')
        self.reward_ph = tf.keras.layers.Input(1, name='reward')
        self.terminal_ph = tf.keras.layers.Input(1, name='terminal')
        self.next_obs_ph = tf.keras.layers.Input(self.state_dim, name='next_obs')

        batch_size = tf.shape(self.obs_ph)[0]
        timestep = tf.tile(tf.range(self.lcm_c)[None, :], (batch_size, 1))  # batch_size x lcm
        timestep_tile = tf.reshape(timestep, (batch_size * self.lcm_c,))
        obs_tile = tf.reshape(tf.tile(tf.expand_dims(self.obs_ph, axis=1), (1, self.lcm_c, 1)), (batch_size * self.lcm_c, self.state_dim))
        action_tile = tf.reshape(tf.tile(tf.expand_dims(self.action_ph, axis=1), (1, self.lcm_c, 1)), (batch_size * self.lcm_c, self.action_dim))
        reward_tile = tf.reshape(tf.tile(tf.expand_dims(self.reward_ph, axis=1), (1, self.lcm_c, 1)), (batch_size * self.lcm_c, 1))
        terminal_tile = tf.reshape(tf.tile(tf.expand_dims(self.terminal_ph, axis=1), (1, self.lcm_c, 1)), (batch_size * self.lcm_c, 1))
        next_obs_tile = tf.reshape(tf.tile(tf.expand_dims(self.next_obs_ph, axis=1), (1, self.lcm_c, 1)), (batch_size * self.lcm_c, self.state_dim))

        def _project_action(timestep, action_old, action):
            """ Gamma_{timestep, action_old}^{c}(action) """
            action_mask = tf.cast(tf.equal(timestep[:, None] % self.c, 0), tf.float32)  # batch_size x action_dim
            projected_action = action * action_mask + action_old * (1 - action_mask)
            return projected_action

        self.num_critics = 2
        self.gamma = 0.99
        self.tau = 0.005
        self.learning_rate = 3e-4
        self.batch_size = 100
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        self.ent_coef = ent_coef

        self.replay_size = int(1e6)
        self.start_steps = 10000
        self.update_after = 1000

        self.replay_buffer = ReplayBuffer(self.max_action, buffer_size=self.replay_size)

        optimizer_variables = []

        # Entropy coefficient (auto or fixed)
        if isinstance(self.ent_coef, str) and self.ent_coef == 'auto':
            # Default initial value of ent_coef when learned
            init_value = 1.0
            self.log_ent_coef = tf.keras.backend.variable(init_value, dtype=tf.float32, name='log_ent_coef')
            self.ent_coef = tf.exp(self.log_ent_coef)
        else:
            self.ent_coef = tf.constant(self.ent_coef)

        # Actor, Critic
        self.hidden_dim = hidden_dim
        self.actor = Actor(self.c, self.action_dim, hidden_dim=hidden_dim)
        self.q = QNetwork(self.c, hidden_dim=hidden_dim, num_critics=self.num_critics)
        self.q_target = QNetwork(self.c, hidden_dim=hidden_dim, num_critics=self.num_critics)

        # Actor training
        action_pi_tile, logp_pi_tile, dist_tile, logp_raw_pi_tile, dist_mean_tile, dist_log_std_tile = self.actor([timestep_tile, action_tile, next_obs_tile])
        qs_pi = self.q([timestep_tile, next_obs_tile, _project_action(timestep_tile, action_tile, action_pi_tile)])
        actor_loss = tf.reduce_mean(self.ent_coef * logp_pi_tile - tf.reduce_min(qs_pi, axis=0))  # min? idx 0? mean?
        # For action selection
        action_pi, _, _, _, dist_mean, _ = self.actor([self.timestep_ph % self.lcm_c, self.action_ph, self.next_obs_ph])
        self.sampled_action = _project_action(self.timestep_ph, self.action_ph, action_pi)
        self.deterministic_action = _project_action(self.timestep_ph, self.action_ph, tf.tanh(dist_mean))

        actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        actor_train_op = actor_optimizer.minimize(actor_loss, var_list=self.actor.trainable_variables)
        optimizer_variables += actor_optimizer.variables()

        # Critic training (V, Q)
        next_action_pi_tile, next_logp_pi_tile, _, _, _, _ = self.actor([(timestep_tile + 1) % self.lcm_c, action_tile, next_obs_tile])
        v_target_tile = tf.reduce_min(
            self.q_target([(timestep_tile + 1) % self.lcm_c, next_obs_tile, _project_action((timestep_tile + 1) % self.lcm_c, action_tile, next_action_pi_tile)]), axis=0) \
            - self.ent_coef * next_logp_pi_tile
        qs = self.q([timestep_tile, obs_tile, action_tile])
        q_backup = tf.stop_gradient(reward_tile + (1 - terminal_tile) * self.gamma * v_target_tile)  # batch x 1
        q_losses = [tf.losses.mean_squared_error(q_backup, qs[k]) for k in range(self.num_critics)]
        q_loss = tf.reduce_sum(q_losses)

        value_loss = q_loss
        critic_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        critic_train_op = critic_optimizer.minimize(value_loss, var_list=self.q.trainable_variables)
        optimizer_variables += critic_optimizer.variables()

        with tf.control_dependencies([critic_train_op]):
            # Entropy temperature
            if isinstance(ent_coef, str) and ent_coef == 'auto':
                ent_coef_loss = -tf.reduce_mean(self.log_ent_coef * tf.stop_gradient(logp_pi_tile + self.target_entropy))
                entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                entropy_train_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                optimizer_variables += entropy_optimizer.variables()
            else:
                entropy_train_op = tf.no_op('entropy_train_no_op')

            # Update target network
            source_params = self.q.trainable_variables
            target_params = self.q_target.trainable_variables
            target_update_op = [
                tf.assign(target, (1 - self.tau) * target + self.tau * source)
                for target, source in zip(target_params, source_params)
            ]

        # Copy weights to target networks
        self.sess = tf.keras.backend.get_session()
        self.sess.run(tf.variables_initializer(optimizer_variables))
        self.q_target.set_weights(self.q.get_weights())

        self.step_ops = [actor_train_op, critic_train_op, target_update_op, entropy_train_op]
        self.info_ops = [actor_loss, q_loss, qs] + \
                        [self.ent_coef, dist_tile.entropy(), logp_pi_tile]#, dist.mean(), dist.stddev(), action_pi, logp_raw_pi, dist_mean, dist_log_std]
        self.info_labels = ['actor_loss', 'q_loss', 'qs', 'ent_coef', 'entropy', 'logp_pi']#, 'pol_mean', 'pol_stddev', 'pol_action', 'logp_raw_pi', 'dist_mean', 'dist_log_std']

    def validate_action(self, timestep, action_old, action):
        if len(action.shape) == 1:
            return self.validate_action(np.array([timestep]), np.array([action_old]), np.array([action]))[0]
        assert len(action.shape) == 2
        action_mask = np.equal(timestep[:, None] % self.c, 0)
        action_fixed = action_mask * action + (1 - action_mask) * action_old
        return action_fixed

    def _train_step(self):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)  # action is normalized

        _, info_value = self.sess.run([self.step_ops, self.info_ops], feed_dict={
            self.obs_ph: obs,
            self.action_ph: action,
            self.reward_ph: reward,
            self.next_obs_ph: next_obs,
            self.terminal_ph: done
        })
        if not np.all([np.all(np.isfinite(x)) for x in info_value]):
            d = {}
            print('\n============================')
            for i, label in enumerate(self.info_labels):
                d[label] = info_value[i]
                print('%12s: %s' %(label, info_value[i]))
            print('============================\n')

            from IPython import embed; embed()
            exit(1)
        # self.sess.run(, feed_dict={self.obs_ph: obs, self.action_ph: action, self.reward_ph: reward, self.next_obs_ph: next_obs, self.terminal_ph: done})
        return info_value

    def learn(self, vec_env, total_timesteps, log_interval, checkpoint_interval, seed, result_filepath, checkpoint_path, result=None, verbose=1):
        np.random.seed(seed)

        start_time = time.time()
        episode_rewards = [0.0]

        # Start
        start_time = time.time()
        if result is None:
            result = {'eval_timesteps': [], 'evals': [], 'info_values': [], 'log_interval': log_interval}

        timestep = 0
        obs = self.env.reset()
        action_old = self.env.action_space.sample()
        for iteration in tqdm(range(total_timesteps + 1), desc='APAC', ncols=70, mininterval=1):
            # Take an action
            if iteration < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.predict(np.array([timestep], dtype=np.int32), np.array([action_old]), np.array([obs]), deterministic=False)[0].flatten()
            action_mask = timestep % self.c == 0  # action_dim
            action = (action_mask) * action + (1 - action_mask) * action_old
            next_obs, reward, done, info = self.env.step(action)

            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            action_old = action

            episode_rewards[-1] += reward

            if self.replay_buffer.can_sample(self.batch_size) and iteration >= self.update_after:
                info_value = self._train_step()

                if iteration % log_interval == 0:
                    evaluation = evaluate_policy(vec_env, self)
                    # evaluation = np.mean(episode_rewards[-30:])
                    result['eval_timesteps'].append(iteration)
                    result['evals'].append(evaluation)
                    result['info_values'].append({label: np.mean(value) for label, value in zip(self.info_labels, info_value)})
                    result['episode_rewards'] = np.array(episode_rewards)
                    print('t=%d: %f (elapsed_time=%f)' % (iteration, evaluation, time.time() - start_time))
                    print('\n============================')
                    print('%12s: %10.3f' % ('ep_rewmean', np.mean(episode_rewards[-100:])))
                    print('%12s: %10d' % ('rb_size', len(self.replay_buffer)))
                    for label, value in zip(self.info_labels, info_value):
                        print('%12s: %10.3f' % (label, np.mean(value)))
                    print('============================\n', flush=True)

                    if result_filepath:
                        np.save(result_filepath + '.tmp.npy', result)

                if iteration % checkpoint_interval == 0:
                    print('Checkpoint saved: ', checkpoint_path.format(iteration))
                    self.save(checkpoint_path.format(iteration))

            timestep += 1
            if done or info.get('TimeLimit.truncated'):
                obs = self.env.reset()
                episode_rewards.append(0.0)
                timestep = 0

        return result

    def predict(self, timestep, action_old, obs, deterministic=False):
        obs_rank = len(obs.shape)
        if len(obs.shape) == 1:
            return self.predict(np.array([timestep], dtype=np.int32), np.array([action_old]), np.array([obs]))[0]
        assert len(obs.shape) == 2

        if deterministic:
            action = self.sess.run(self.deterministic_action, feed_dict={
                self.timestep_ph: timestep,
                self.action_ph: action_old,
                self.next_obs_ph: obs
            })
        else:
            action = self.sess.run(self.sampled_action, feed_dict={
                self.timestep_ph: timestep,
                self.action_ph: action_old,
                self.next_obs_ph: obs
            })

        rescaled_action = action * self.max_action
        return rescaled_action, None

    def get_parameters(self):
        parameters = []
        weights = self.get_weights()
        for idx, variable in enumerate(self.trainable_variables):
            weight = weights[idx]
            parameters.append((variable.name, weight))
        return parameters

    def load_parameters(self, parameters, exact_match=True):
        assert len(parameters) == len(self.weights)
        weights = []
        for variable, parameter in zip(self.weights, parameters):
            name, value = parameter
            if exact_match:
                if name != variable.name:
                    print(name, variable.name)
                assert name == variable.name
            weights.append(value)
        self.set_weights(weights)

    def save(self, filepath):
        parameters = self.get_parameters()
        with open(filepath, 'wb') as f:
            pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath, env, c, seed, hidden_dim):
        with open(filepath, 'rb') as f:
            parameters = pickle.load(f)

        model = APAC(env, c, seed=seed, hidden_dim=hidden_dim)
        model.load_parameters(parameters)
        return model
