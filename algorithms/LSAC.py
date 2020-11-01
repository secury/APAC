import pickle
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from evaluate import evaluate_policy_lsac


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
            idx = np.random.randint(self.N)
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

    def __init__(self, action_dim, hidden_dim=100):
        super(Actor, self).__init__()
        self.action_dim = action_dim

        # Actor parameters
        self.l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='f0')
        self.l2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='f1')
        self.l3_mu = tf.keras.layers.Dense(action_dim, name='f2_mu')
        self.l3_log_std = tf.keras.layers.Dense(action_dim, name='f2_log_std')

    def call(self, inputs, **kwargs):
        obs, = inputs
        h = self.l1(obs)
        h = self.l2(h)
        mean = self.l3_mu(h)
        log_std = self.l3_log_std(h)
        std = tf.math.softplus(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        dist.shape = mean.shape

        sampled_action = dist.sample()
        sampled_action_logp = dist.log_prob(sampled_action)[:, None]
        squahsed_action, squahsed_action_logp = apply_squashing_func(sampled_action, sampled_action_logp)

        return squahsed_action, squahsed_action_logp, dist, sampled_action_logp, mean, log_std


class VNetwork(tf.keras.layers.Layer):

    def __init__(self, hidden_dim=100, output_dim=1):
        super(VNetwork, self).__init__()

        self.v_l0 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='v/f0')
        self.v_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='v/f1')
        self.v_l2 = tf.keras.layers.Dense(output_dim, name='v/f2')

    def call(self, inputs, **kwargs):
        obs, = inputs
        h = self.v_l0(obs)
        h = self.v_l1(h)
        v = self.v_l2(h)
        return v


class QNetwork(tf.keras.layers.Layer):

    def __init__(self, hidden_dim=100, num_critics=2):
        super(QNetwork, self).__init__()
        self.num_critics = num_critics

        self.qs_l0, self.qs_l1, self.qs_l2 = [], [], []
        for i in range(self.num_critics):
            self.qs_l0.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name='q%d/f0' % i))
            self.qs_l1.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name='q%d/f1' % i))
            self.qs_l2.append(tf.keras.layers.Dense(1, name='q%d/f2' % i))

    def call(self, inputs, **kwargs):
        obs, action = inputs
        obs_action = tf.concat([obs, action], axis=1)
        qs = []
        for i in range(self.num_critics):
            h = self.qs_l0[i](obs_action)
            h = self.qs_l1[i](h)
            q = self.qs_l2[i](h)
            qs.append(q)

        return qs


class LSAC(tf.keras.layers.Layer):

    def __init__(self, env, c, ent_coef='auto', seed=0, hidden_dim=100):
        super(LSAC, self).__init__()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.env = env
        self.max_action = self.env.action_space.high[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.c = np.array(c)  # action_dim
        self.lcm_c = int(np.lcm.reduce(self.c))

        self.obs_ph = tf.keras.layers.Input(self.state_dim + self.lcm_c * (self.action_dim + 1), name='obs')
        self.action_ph = tf.keras.layers.Input(self.action_dim, name='action')
        self.reward_ph = tf.keras.layers.Input(1, name='reward')
        self.terminal_ph = tf.keras.layers.Input(1, name='terminal')
        self.next_obs_ph = tf.keras.layers.Input(self.state_dim + self.lcm_c * (self.action_dim + 1), name='next_obs')

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
        self.actor = Actor(self.action_dim, hidden_dim=hidden_dim)
        self.v = VNetwork(hidden_dim=hidden_dim)
        self.q = QNetwork(hidden_dim=hidden_dim, num_critics=self.num_critics)
        self.v_target = VNetwork(hidden_dim=hidden_dim)

        # Actor training
        action_pi, logp_pi, dist, logp_raw_pi, dist_mean, dist_log_std = self.actor([self.obs_ph])
        qs_pi = self.q([self.obs_ph, action_pi])
        actor_loss = tf.reduce_mean(self.ent_coef * logp_pi - tf.reduce_min(qs_pi, axis=0))  # min? idx 0? mean?
        # For action selection
        self.sampled_action = action_pi
        self.deterministic_action = tf.tanh(dist.mean())

        actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        actor_train_op = actor_optimizer.minimize(actor_loss, var_list=self.actor.trainable_variables)
        optimizer_variables += actor_optimizer.variables()

        with tf.control_dependencies([actor_train_op]):
            # Critic training (V, Q)
            v = self.v([self.obs_ph])
            min_q_pi = tf.reduce_min(qs_pi, axis=0)
            v_backup = tf.stop_gradient(min_q_pi - self.ent_coef * logp_pi)
            v_loss = tf.losses.mean_squared_error(v_backup, v)

            v_target = self.v_target([self.next_obs_ph])
            qs = self.q([self.obs_ph, self.action_ph])
            q_backup = tf.stop_gradient(self.reward_ph + (1 - self.terminal_ph) * self.gamma * v_target)  # batch x 1
            q_losses = [tf.losses.mean_squared_error(q_backup, qs[k]) for k in range(self.num_critics)]
            q_loss = tf.reduce_sum(q_losses)

            value_loss = v_loss + q_loss
            critic_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            critic_train_op = critic_optimizer.minimize(value_loss, var_list=self.v.trainable_variables + self.q.trainable_variables)
            optimizer_variables += critic_optimizer.variables()

            with tf.control_dependencies([critic_train_op]):
                # Entropy temperature
                if isinstance(ent_coef, str) and ent_coef == 'auto':
                    ent_coef_loss = -tf.reduce_mean(self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                    entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    entropy_train_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                    optimizer_variables += entropy_optimizer.variables()
                else:
                    entropy_train_op = tf.no_op('entropy_train_no_op')

                # Update target network
                source_params = self.v.trainable_variables
                target_params = self.v_target.trainable_variables
                target_update_op = [
                    tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    for target, source in zip(target_params, source_params)
                ]

        # Copy weights to target networks
        self.sess = tf.keras.backend.get_session()
        self.sess.run(tf.variables_initializer(optimizer_variables))
        self.v_target.set_weights(self.v.get_weights())

        self.step_ops = [actor_train_op, critic_train_op, target_update_op, entropy_train_op]
        self.info_ops = [actor_loss, v_loss, q_loss, v, qs] + \
                        [self.ent_coef, dist.entropy(), logp_pi]  #, dist.mean(), dist.stddev(), action_pi, logp_raw_pi, dist_mean, dist_log_std]
        self.info_labels = ['actor_loss', 'v_loss', 'q_loss', 'v', 'qs', 'ent_coef', 'entropy', 'logp_pi']  #, 'pol_mean', 'pol_stddev', 'pol_action', 'logp_raw_pi', 'dist_mean', 'dist_log_std']

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
        list_action_old = [self.env.action_space.sample() for t in range(self.lcm_c)]
        one_hot_timestep = np.eye(self.lcm_c)[timestep % self.lcm_c]
        full_obs = np.concatenate([obs, one_hot_timestep] + list_action_old)
        for iteration in tqdm(range(total_timesteps + 1), desc='LSAC', ncols=70, mininterval=1):
            # Take an action
            if iteration < self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.predict(np.array([full_obs]), deterministic=False)[0].flatten()
            next_obs, reward, done, info = self.env.step(action)

            list_action_old = [action] + list_action_old[:-1]
            one_hot_timestep = np.eye(self.lcm_c)[(timestep + 1) % self.lcm_c]
            full_next_obs = np.concatenate([next_obs, one_hot_timestep] + list_action_old)
            # Store transition in the replay buffer.
            self.replay_buffer.add(full_obs, action, reward, full_next_obs, float(done))
            full_obs = full_next_obs

            episode_rewards[-1] += reward
            if done:
                obs = self.env.reset()
                episode_rewards.append(0.0)

            if self.replay_buffer.can_sample(self.batch_size) and iteration >= self.update_after:
                info_value = self._train_step()

                if iteration % log_interval == 0:
                    evaluation = evaluate_policy_lsac(vec_env, self)
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
                list_action_old = [self.env.action_space.sample() for t in range(self.lcm_c)]
                one_hot_timestep = np.eye(self.lcm_c)[timestep % self.lcm_c]
                full_obs = np.concatenate([obs, one_hot_timestep] + list_action_old)

        return result

    def predict(self, obs, deterministic=False, **kwargs):
        obs_rank = len(obs.shape)
        if len(obs.shape) == 1:
            obs = np.array([obs])
        assert len(obs.shape) == 2

        if deterministic:
            action = self.sess.run(self.deterministic_action, feed_dict={self.obs_ph: obs})
        else:
            action = self.sess.run(self.sampled_action, feed_dict={self.obs_ph: obs})

        rescaled_action = action * self.max_action

        if obs_rank == 1:
            return rescaled_action[0], None
        else:
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

        model = LSAC(env, c, seed=seed, hidden_dim=hidden_dim)
        model.load_parameters(parameters)
        return model
